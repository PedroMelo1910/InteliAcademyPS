from __future__ import annotations

import logging
import math
import re
import socket
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from radar.configuracao import (
    DIMENSAO_EMBEDDING_NVIDIA,
    MODELO_EMBEDDING_NVIDIA,
    MODELO_GEMINI,
    MODELO_GROQ,
    MODELO_RERANK_NVIDIA,
    REQUISICOES_GROQ_POR_SEGUNDO,
)
from radar.contratos import (
    BriefingRascunho,
    Classificacao,
    PerfilExtraido,
    PlanoConsulta,
    RecomendacaoRascunho,
)


# O Flash-Lite rejeita parte das restricoes declaradas como JSON Schema mesmo
# quando elas pertencem ao subconjunto documentado pelo endpoint. Enviamos ao
# modelo apenas a forma estrutural; enums e limites continuam obrigatorios nos
# contratos Pydantic e sao conferidos na fronteira, inclusive no retry unico.
_CHAVES_JSON_SCHEMA_GEMINI = frozenset(
    {
        "type",
        "enum",
        "items",
        "prefixItems",
        "anyOf",
        "oneOf",
        "properties",
        "required",
    }
)


logger = logging.getLogger(__name__)


def _schema_json_compativel_gemini(contrato: type[BaseModel]) -> dict[str, Any]:
    """Remove palavras-chave que o Gemini rejeita em ``responseJsonSchema``."""

    original = contrato.model_json_schema()
    definicoes = original.get("$defs", {})

    def limpar(
        schema: dict[str, Any], referencias_em_curso: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        referencia = schema.get("$ref")
        if isinstance(referencia, str):
            nome = referencia.rsplit("/", maxsplit=1)[-1]
            if nome in referencias_em_curso or nome not in definicoes:
                raise ValueError(f"referencia JSON Schema nao suportada: {referencia}")
            combinado = dict(definicoes[nome])
            combinado.update(
                {chave: valor for chave, valor in schema.items() if chave != "$ref"}
            )
            return limpar(combinado, referencias_em_curso | {nome})

        resultado: dict[str, Any] = {}
        for chave, valor in schema.items():
            if chave == "$defs":
                continue
            elif chave not in _CHAVES_JSON_SCHEMA_GEMINI:
                continue
            elif chave == "properties":
                resultado[chave] = {
                    nome: limpar(subschema, referencias_em_curso)
                    for nome, subschema in valor.items()
                }
            elif chave == "items" and isinstance(valor, dict):
                resultado[chave] = limpar(valor, referencias_em_curso)
            elif chave in {"anyOf", "oneOf", "prefixItems"}:
                resultado[chave] = [
                    limpar(subschema, referencias_em_curso) for subschema in valor
                ]
            else:
                resultado[chave] = valor
        return resultado

    return limpar(original)


def _com_saida_estruturada_gemini(
    modelo: ChatGoogleGenerativeAI, contrato: type[BaseModel]
) -> object:
    """Configura JSON nativo sem enfraquecer a validacao Pydantic posterior."""

    return modelo.with_structured_output(
        _schema_json_compativel_gemini(contrato), method="json_schema"
    )


class ProvedorPlanoConsulta(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz uma resposta estruturada ainda sujeita à validação da fronteira."""


class ProvedorGeminiPlanoConsulta:
    """Único contato de rede do fluxo atual."""

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=30,
        )
        self._estruturado = _com_saida_estruturada_gemini(modelo, PlanoConsulta)

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorPerfilExtraido(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz um perfil estruturado ainda sujeito à validação da fronteira."""


class ProvedorGeminiPerfilExtraido:
    """Adaptador de structured output do Extractor; o nó não conhece a rede."""

    def __init__(
        self,
        api_key: str,
        *,
        limitador: BaseRateLimiter | None = None,
    ):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
            rate_limiter=limitador,
        )
        self._estruturado = _com_saida_estruturada_gemini(modelo, PerfilExtraido)

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorClassificacao(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz uma classificação estruturada ainda sujeita à validação da fronteira."""


class ProvedorGeminiClassificacao:
    """Adaptador de structured output do Classifier; o nó não conhece a rede."""

    def __init__(
        self,
        api_key: str,
        *,
        limitador: BaseRateLimiter | None = None,
    ):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
            rate_limiter=limitador,
        )
        self._estruturado = _com_saida_estruturada_gemini(modelo, Classificacao)

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class RascunhosRecomendacao(BaseModel):
    """Envelope de saída estruturada do Recommendation: só rascunhos.

    O LLM produz de 1 a 5 ``RecomendacaoRascunho`` numa única chamada. O modelo
    final ``Recomendacao`` nunca é pedido a ele: prioridade, complexidade e
    fit-score sequer existem neste schema, e as evidências entram como ids que
    o nó resolve depois.
    """

    model_config = ConfigDict(extra="forbid")

    rascunhos: list[RecomendacaoRascunho] = Field(min_length=1, max_length=5)


class ProvedorRecomendacaoRascunho(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz rascunhos ainda sujeitos à validação de proveniência do nó."""


class ProvedorGeminiRecomendacaoRascunho:
    """Adaptador de structured output do Recommendation; o nó não conhece a rede."""

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
        )
        self._estruturado = _com_saida_estruturada_gemini(
            modelo, RascunhosRecomendacao
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorBriefingRascunho(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz as ~6 frases do briefing, ainda sujeitas à conferência do nó."""


class ProvedorGeminiBriefingRascunho:
    """Adaptador de structured output do Briefing; o nó não conhece a rede.

    O schema oferecido ao modelo é o reduzido: tese, síntese e pontos, cada um
    com os ids que ele escolheu. Classe, fit-score, recomendações, fontes,
    avisos e rodapé não existem nele.
    """

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
        )
        self._estruturado = _com_saida_estruturada_gemini(modelo, BriefingRascunho)

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


# ----------------------------------------------------------------------
# Reserva operacional: Groq atrás dos mesmos protocolos
# ----------------------------------------------------------------------
#
# O Gemini continua sendo o provedor primário. O Groq só é chamado quando a
# chamada primária falha por **indisponibilidade** — nunca por erro de
# contrato. A reserva atravessa exatamente a mesma validação Pydantic do nó:
# ela não relaxa contrato, não pula conferência de evidência literal, não
# aumenta teto de retry e não muda a falha segura.


class ErroProvedoresIndisponiveis(RuntimeError):
    """Primário e reserva indisponíveis; nada foi fabricado no lugar."""

    # Lido por ``falha_operacional``: esta é indisponibilidade, não defeito.
    operacional = True


class ErroReservaIncompativel(RuntimeError):
    """A reserva falhou por contrato, schema ou defeito — não por queda.

    Chamar isso de indisponibilidade esconderia um defeito real atrás de um
    rótulo operacional, e o runner de lote passaria a tratá-lo como algo que
    "melhora sozinho".
    """

    operacional = False


# Motivo neutro para a falha segura do nó quando a reserva recusa o pedido
# estruturado. Não nomeia provedor, não carrega código HTTP e não repete
# corpo de resposta de terceiro.
MOTIVO_RECUSA_ESTRUTURADA = (
    "a fronteira de provedores recusou o pedido estruturado"
)


_CODIGOS_SEGUROS_PARA_RELATO = frozenset(
    {400, 401, 402, 403, 404, 408, 409, 422, 425, 429, 500, 502, 503, 504}
)


def _status_seguro(erro: BaseException) -> int | None:
    """Só o código numérico; nunca a mensagem ou o corpo da resposta."""
    candidatos = [getattr(erro, "status_code", None)]
    resposta = getattr(erro, "response", None)
    if resposta is not None:
        candidatos.append(getattr(resposta, "status_code", None))
    for candidato in candidatos:
        try:
            codigo = int(candidato)
        except (TypeError, ValueError):
            continue
        if codigo in _CODIGOS_SEGUROS_PARA_RELATO:
            return codigo
    return None


def _descrever(classe: str, status: int | None) -> str:
    return f"{classe}" if status is None else f"{classe} ({status})"


def _descrever_falha_de_provedor(operacao: str, erro: BaseException) -> str:
    """Mensagem única e segura para qualquer falha vinda de SDK de terceiro.

    A fronteira LLM já usava ``_descrever`` + ``_status_seguro``; os adaptadores
    NVIDIA interpolavam ``str(excecao)`` inteira e encadeavam a original, o que
    levava corpo de resposta, cabeçalho e credencial para a mensagem e para o
    traceback. Aqui sobrevivem só três coisas: a operação do projeto, o nome da
    classe da exceção e um código HTTP de allowlist. Quem levanta deve usar
    ``from None``, para que a causa não recomponha o corpo no traceback comum.
    """
    return f"{operacao}: {_descrever(type(erro).__name__, _status_seguro(erro))}"


_LIMITADOR_GROQ: BaseRateLimiter | None = None


def limitador_groq() -> BaseRateLimiter:
    """Limitador único e conservador, compartilhado por todas as fronteiras.

    Uma queda do Gemini joga todo o lote na reserva de uma vez; sem um teto
    comum, a saída de um provedor viraria estouro de limite no outro.
    """
    global _LIMITADOR_GROQ
    if _LIMITADOR_GROQ is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _LIMITADOR_GROQ = InMemoryRateLimiter(
            requests_per_second=REQUISICOES_GROQ_POR_SEGUNDO,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        )
    return _LIMITADOR_GROQ


def _modelo_groq(api_key: str):
    """Import tardio: teste offline injeta fake e nunca constrói cliente real."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=MODELO_GROQ,
        api_key=api_key,
        temperature=0,
        max_retries=1,
        timeout=60,
        rate_limiter=limitador_groq(),
    )


def _com_saida_estruturada_groq(modelo, contrato: type[BaseModel]) -> object:
    """JSON Schema nativo do contrato; a autoridade final segue sendo o nó.

    Devolve dicionário, igual ao adaptador Gemini, para que o mesmo
    ``model_validate`` do nó continue sendo o único juiz — inclusive do
    ``extra="forbid"``.
    """
    return modelo.with_structured_output(
        contrato.model_json_schema(), method="json_schema"
    )


class ProvedorGroqPlanoConsulta:
    def __init__(self, api_key: str):
        self._estruturado = _com_saida_estruturada_groq(
            _modelo_groq(api_key), PlanoConsulta
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorGroqPerfilExtraido:
    def __init__(self, api_key: str):
        self._estruturado = _com_saida_estruturada_groq(
            _modelo_groq(api_key), PerfilExtraido
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorGroqClassificacao:
    def __init__(self, api_key: str):
        self._estruturado = _com_saida_estruturada_groq(
            _modelo_groq(api_key), Classificacao
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorGroqRecomendacaoRascunho:
    def __init__(self, api_key: str):
        self._estruturado = _com_saida_estruturada_groq(
            _modelo_groq(api_key), RascunhosRecomendacao
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorGroqBriefingRascunho:
    def __init__(self, api_key: str):
        self._estruturado = _com_saida_estruturada_groq(
            _modelo_groq(api_key), BriefingRascunho
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorComFallback:
    """Composição única de primário + reserva, reusada pelas cinco fronteiras.

    Uma tentativa em cada, nesta ordem, sem alternância: se o primário devolve,
    a reserva recebe zero chamadas; se o primário falha por indisponibilidade,
    a reserva é chamada uma vez; se o primário falha por contrato, o erro sobe
    intacto e a reserva não é chamada. Como cada agente já faz no máximo duas
    tentativas de correção estruturada, o teto por fronteira é de **2 chamadas
    ao primário e 2 à reserva**.
    """

    def __init__(self, primario, reserva, *, fronteira: str):
        self.primario = primario
        self.reserva = reserva
        self.fronteira = fronteira

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        try:
            return self.primario.invocar(mensagens)
        except Exception as erro:
            if not falha_operacional(erro):
                # Erro de contrato, de estado ou de programação: quem corrige é
                # o retry do próprio agente, não outro provedor.
                raise
            # Só o que é seguro relatar sai do bloco: classe e código. A
            # exceção original morre aqui e não vira ``__cause__`` nem
            # ``__context__`` do erro final.
            primario_descrito = _descrever(type(erro).__name__, _status_seguro(erro))

        # Observabilidade mínima: fronteira e decisão. Sem prompt, sem
        # documento, sem resposta, sem chave, sem corpo de erro de terceiro.
        logger.warning(
            "fronteira %s: provedor primário indisponível (%s); acionando "
            "provedor reserva",
            self.fronteira,
            primario_descrito,
        )
        try:
            return self.reserva.invocar(mensagens)
        except Exception as erro_reserva:
            operacional = falha_operacional(erro_reserva)
            reserva_descrita = _descrever(
                type(erro_reserva).__name__, _status_seguro(erro_reserva)
            )

        # Levantado FORA de qualquer ``except``: sem exceção ativa, o Python não
        # preenche ``__context__``, e nenhuma mensagem de provedor fica
        # alcançável por log, relatório de lote ou traceback formatado.
        if operacional:
            raise ErroProvedoresIndisponiveis(
                f"a fronteira {self.fronteira} ficou sem provedor: primário "
                f"{primario_descrito} e reserva {reserva_descrita}"
            ) from None
        raise ErroReservaIncompativel(
            f"na fronteira {self.fronteira} o primário caiu por "
            f"{primario_descrito} e a reserva recusou por {reserva_descrita}, "
            "que não é indisponibilidade"
        ) from None


def compor_com_reserva(primario, fabrica_reserva, *, fronteira: str, chave_reserva: str):
    """Devolve o primário puro quando não há chave; senão, a composição.

    Sem ``GROQ_API_KEY`` o comportamento fica idêntico ao de hoje — nenhum
    cliente de reserva é construído.
    """
    if not (chave_reserva or "").strip():
        return primario
    return ProvedorComFallback(
        primario, fabrica_reserva(chave_reserva), fronteira=fronteira
    )


class ProvedorContextoNvidia(Protocol):
    def consultar(self, consulta: str) -> object:
        """Recupera trechos NVIDIA; a validação do contrato fica na fronteira do nó.

        ``ConhecimentoNvidia`` satisfaz este protocolo estruturalmente: o nó do
        grafo depende da forma da chamada, nunca da classe concreta, e por isso
        os testes injetam um consultor determinístico sem tocar a rede.
        """


# --------------------------------------------------------------------------
# Provedores da base de conhecimento NVIDIA (Entregável 2).
# Falha operacional (timeout, indisponibilidade, autenticação, cota, rate
# limit, pagamento) é elegível a fallback; contrato inválido, dimensão errada
# ou invariante quebrada falham alto sem disfarce.
# --------------------------------------------------------------------------

_MARCADORES_FALHA_OPERACIONAL = (
    "timeout",
    "timed out",
    "deadline",
    "connection",
    "connect",
    "unavailable",
    "rate limit",
    "quota",
    "payment",
    "too many requests",
    "unauthorized",
    "permission",
    "api key",
    "exhausted",
)

_CODIGOS_FALHA_OPERACIONAL = {401, 402, 403, 408, 425, 429, 500, 502, 503, 504}
_NOMES_FALHA_OPERACIONAL = (
    "timeout",
    "connection",
    "connecterror",
    "networkerror",
    "ratelimit",
    "authentication",
    "permissiondenied",
    "serviceunavailable",
)


def falha_operacional(excecao: Exception) -> bool:
    """Classifica indisponibilidade sem depender apenas da mensagem humana."""
    atual: BaseException | None = excecao
    vistos: set[int] = set()
    while atual is not None and id(atual) not in vistos:
        vistos.add(id(atual))
        if isinstance(atual, (TimeoutError, ConnectionError, socket.timeout, socket.gaierror)):
            return True
        operacional = getattr(atual, "operacional", None)
        if isinstance(operacional, bool):
            return operacional
        nome_classe = type(atual).__name__.casefold()
        if any(marcador in nome_classe for marcador in _NOMES_FALHA_OPERACIONAL):
            return True

        candidatos_status = [getattr(atual, "status_code", None)]
        resposta = getattr(atual, "response", None)
        if resposta is not None:
            candidatos_status.append(getattr(resposta, "status_code", None))
        for candidato in candidatos_status:
            try:
                if int(candidato) in _CODIGOS_FALHA_OPERACIONAL:
                    return True
            except (TypeError, ValueError):
                pass

        texto = str(atual).casefold()
        if any(marcador in texto for marcador in _MARCADORES_FALHA_OPERACIONAL):
            return True
        if re.search(
            r"\b(?:http|status(?:\s+code)?)\s*[:=]?\s*"
            r"(?:401|402|403|408|425|429|500|502|503|504)\b",
            texto,
        ):
            return True
        atual = atual.__cause__ or atual.__context__
    return False


class ErroProvedorEmbedding(RuntimeError):
    def __init__(self, mensagem: str, *, operacional: bool):
        super().__init__(mensagem)
        self.operacional = operacional


class ErroProvedorRerank(RuntimeError):
    def __init__(self, mensagem: str, *, operacional: bool):
        super().__init__(mensagem)
        self.operacional = operacional


class ErroRerankIndisponivel(ErroProvedorRerank):
    """Falha dupla: nem o reranker primário nem o fallback responderam."""

    def __init__(self, mensagem: str):
        super().__init__(mensagem, operacional=True)


class EmbeddingProvider(Protocol):
    @property
    def dimensao(self) -> int: ...

    @property
    def modelo(self) -> str:
        """Identificador do modelo; compõe a chave do cache de embeddings."""
        ...

    def embutir_passagens(self, textos: list[str]) -> list[list[float]]:
        """Embeddings em modo passage, usado somente na ingestão."""

    def embutir_consulta(self, texto: str) -> list[float]:
        """Embedding em modo query, usado somente na recuperação."""


class RerankProvider(Protocol):
    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        """Scores de relevância alinhados por índice com ``textos``."""


class ProvedorEmbeddingNvidia:
    """Adaptador do endpoint hospedado de embedding da NVIDIA.

    ``embed_documents`` e ``embed_query`` do cliente preservam a distinção
    passage/query do modelo; os dois modos nunca se misturam.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        modelo: str = MODELO_EMBEDDING_NVIDIA,
        dimensao: int = DIMENSAO_EMBEDDING_NVIDIA,
        cliente: object | None = None,
    ):
        if cliente is None:
            if not api_key:
                raise ValueError("api_key é obrigatória sem um cliente injetado")
            try:
                cliente = NVIDIAEmbeddings(
                    model=modelo, nvidia_api_key=api_key, truncate="END"
                )
            except Exception as excecao:
                raise ErroProvedorEmbedding(
                    _descrever_falha_de_provedor(
                        "falha ao inicializar o provedor de embedding", excecao
                    ),
                    operacional=falha_operacional(excecao),
                ) from None
        self._cliente = cliente
        self._dimensao = dimensao
        self._modelo = modelo

    @property
    def dimensao(self) -> int:
        return self._dimensao

    @property
    def modelo(self) -> str:
        return self._modelo

    def _validar(self, vetores: list[list[float]], esperados: int) -> list[list[float]]:
        if len(vetores) != esperados:
            raise ErroProvedorEmbedding(
                f"o provedor devolveu {len(vetores)} vetores para {esperados} textos",
                operacional=False,
            )
        for vetor in vetores:
            if len(vetor) != self._dimensao:
                raise ErroProvedorEmbedding(
                    f"vetor com {len(vetor)} dimensões; esperado {self._dimensao}",
                    operacional=False,
                )
            if not all(math.isfinite(valor) for valor in vetor):
                raise ErroProvedorEmbedding(
                    "vetor com valor não finito na resposta do provedor",
                    operacional=False,
                )
        return vetores

    def embutir_passagens(self, textos: list[str]) -> list[list[float]]:
        try:
            vetores = self._cliente.embed_documents(textos)
        except Exception as excecao:
            raise ErroProvedorEmbedding(
                _descrever_falha_de_provedor(
                    "falha do provedor de embedding em modo passage", excecao
                ),
                operacional=falha_operacional(excecao),
            ) from None
        return self._validar(vetores, len(textos))

    def embutir_consulta(self, texto: str) -> list[float]:
        try:
            vetor = self._cliente.embed_query(texto)
        except Exception as excecao:
            raise ErroProvedorEmbedding(
                _descrever_falha_de_provedor(
                    "falha do provedor de embedding em modo query", excecao
                ),
                operacional=falha_operacional(excecao),
            ) from None
        return self._validar([vetor], 1)[0]


class ProvedorRerankNvidia:
    """Adaptador do reranker hospedado da NVIDIA (primário)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        modelo: str = MODELO_RERANK_NVIDIA,
        cliente: object | None = None,
    ):
        if cliente is None:
            if not api_key:
                raise ValueError("api_key é obrigatória sem um cliente injetado")
            try:
                cliente = NVIDIARerank(model=modelo, nvidia_api_key=api_key)
            except Exception as excecao:
                raise ErroProvedorRerank(
                    _descrever_falha_de_provedor(
                        "falha ao inicializar o reranker NVIDIA", excecao
                    ),
                    operacional=falha_operacional(excecao),
                ) from None
        self._cliente = cliente

    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        documentos = [
            Document(page_content=texto, metadata={"indice": indice})
            for indice, texto in enumerate(textos)
        ]
        try:
            self._cliente.top_n = len(textos)
            resultado = self._cliente.compress_documents(documentos, consulta)
        except Exception as excecao:
            raise ErroProvedorRerank(
                _descrever_falha_de_provedor("falha do reranker NVIDIA", excecao),
                operacional=falha_operacional(excecao),
            ) from None
        scores: dict[int, float] = {}
        try:
            for documento in resultado:
                metadados = documento.metadata
                indice = int(metadados["indice"])
                score = float(metadados["relevance_score"])
                if not math.isfinite(score):
                    raise ValueError("score não finito")
                scores[indice] = score
        except (AttributeError, KeyError, TypeError, ValueError) as erro:
            raise ErroProvedorRerank(
                _descrever_falha_de_provedor(
                    "resposta do reranker viola o contrato de índices e scores",
                    erro,
                ),
                operacional=False,
            ) from None
        if set(scores) != set(range(len(textos))):
            raise ErroProvedorRerank(
                "resposta do reranker não cobre todos os índices enviados",
                operacional=False,
            )
        return [scores[indice] for indice in range(len(textos))]


class OrdenacaoListwise(BaseModel):
    """Saída estruturada do fallback: índices do mais ao menos relevante."""

    model_config = ConfigDict(extra="forbid")

    ordem: list[int]


class ProvedorRerankListwiseGemini:
    """Fallback operacional aprovado: reranking listwise no backbone LLM.

    O score devolvido é uma escala ordinal determinística derivada apenas da
    posição na ordem — existe para satisfazer a ordenação do contrato. Não é
    probabilidade, não é confiança do modelo e não é comparável numericamente
    com os logits do reranker NVIDIA.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        modelo: str = MODELO_GEMINI,
        cliente: object | None = None,
    ):
        if cliente is None:
            if not api_key:
                raise ValueError("api_key é obrigatória sem um cliente injetado")
            base = ChatGoogleGenerativeAI(
                model=modelo,
                api_key=api_key,
                temperature=None,
                retries=1,
                request_timeout=30,
            )
            cliente = _com_saida_estruturada_gemini(base, OrdenacaoListwise)
        self._cliente = cliente

    @staticmethod
    def _montar_mensagens(
        consulta: str, textos: list[str], erro_anterior: str | None
    ) -> list[tuple[str, str]]:
        passagens = "\n\n".join(
            f"[{indice}] {texto}" for indice, texto in enumerate(textos)
        )
        instrucao = (
            "Você é um reranker de passagens. Receberá uma consulta e "
            f"{len(textos)} passagens numeradas de 0 a {len(textos) - 1}. "
            "Responda somente com o campo 'ordem': a lista de todos os índices, "
            "cada um exatamente uma vez, do mais relevante para o menos "
            "relevante em relação à consulta."
        )
        mensagens = [
            ("system", instrucao),
            ("human", f"Consulta: {consulta}\n\nPassagens:\n\n{passagens}"),
        ]
        if erro_anterior:
            mensagens.append(
                (
                    "system",
                    "A resposta anterior violou o contrato. Corrija. "
                    f"Falha: {erro_anterior}",
                )
            )
        return mensagens

    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        total = len(textos)
        erro_anterior: str | None = None
        for tentativa in range(2):
            try:
                bruto = self._cliente.invoke(
                    self._montar_mensagens(consulta, textos, erro_anterior)
                )
            except Exception as excecao:
                raise ErroProvedorRerank(
                    _descrever_falha_de_provedor("falha do fallback listwise", excecao),
                    operacional=falha_operacional(excecao),
                ) from None
            try:
                ordem = list(OrdenacaoListwise.model_validate(bruto).ordem)
            except (ValidationError, TypeError, ValueError) as erro:
                erro_anterior = f"resposta estruturada inválida: {erro}"
                if tentativa == 1:
                    raise ErroProvedorRerank(
                        "o fallback listwise violou o contrato estruturado duas "
                        f"vezes; última falha: {erro_anterior}",
                        operacional=False,
                    ) from erro
                continue
            if sorted(ordem) == list(range(total)):
                scores = [0.0] * total
                for posicao, indice in enumerate(ordem):
                    scores[indice] = (total - posicao) / total
                return scores
            erro_anterior = (
                f"a ordem {ordem} não é uma permutação de 0..{total - 1}"
            )
            if tentativa == 1:
                raise ErroProvedorRerank(
                    "o fallback listwise violou o contrato de ordenação duas "
                    f"vezes; última falha: {erro_anterior}",
                    operacional=False,
                )
        raise AssertionError("laço de reranking terminou em estado impossível")


class RerankComFallback:
    """Composição aprovada: NVIDIA primário, listwise LLM como reserva.

    A reserva só é acionada para falha operacional do primário. Falha dupla
    propaga ``ErroRerankIndisponivel``; a ordem da fusão nunca é devolvida
    em silêncio como se fosse reranking.
    """

    def __init__(self, primario: RerankProvider, reserva: RerankProvider):
        self._primario = primario
        self._reserva = reserva

    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        try:
            return self._primario.reordenar(consulta, textos)
        except ErroProvedorRerank as erro_primario:
            if not erro_primario.operacional:
                raise
            try:
                return self._reserva.reordenar(consulta, textos)
            except ErroProvedorRerank as erro_reserva:
                if not erro_reserva.operacional:
                    raise ErroProvedorRerank(
                        "o reranker primário ficou indisponível, mas a reserva "
                        f"violou seu contrato: {erro_reserva}",
                        operacional=False,
                    ) from erro_reserva
                raise ErroRerankIndisponivel(
                    "reranking indisponível: o primário falhou operacionalmente "
                    f"({erro_primario}) e a reserva também falhou ({erro_reserva})"
                ) from erro_reserva
