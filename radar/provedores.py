from __future__ import annotations

import math
import re
import socket
from typing import Protocol

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from pydantic import BaseModel, ConfigDict, ValidationError

from radar.configuracao import (
    DIMENSAO_EMBEDDING_NVIDIA,
    MODELO_EMBEDDING_NVIDIA,
    MODELO_GEMINI,
    MODELO_RERANK_NVIDIA,
)
from radar.contratos import Classificacao, PerfilExtraido, PlanoConsulta


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
        self._estruturado = modelo.with_structured_output(
            PlanoConsulta, method="json_schema"
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorPerfilExtraido(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz um perfil estruturado ainda sujeito à validação da fronteira."""


class ProvedorGeminiPerfilExtraido:
    """Adaptador de structured output do Extractor; o nó não conhece a rede."""

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
        )
        self._estruturado = modelo.with_structured_output(
            PerfilExtraido, method="json_schema"
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


class ProvedorClassificacao(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz uma classificação estruturada ainda sujeita à validação da fronteira."""


class ProvedorGeminiClassificacao:
    """Adaptador de structured output do Classifier; o nó não conhece a rede."""

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=60,
        )
        self._estruturado = modelo.with_structured_output(
            Classificacao, method="json_schema"
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)


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


def _falha_operacional(excecao: Exception) -> bool:
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
                    f"falha ao inicializar o provedor de embedding: {excecao}",
                    operacional=_falha_operacional(excecao),
                ) from excecao
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
                f"falha do provedor de embedding em modo passage: {excecao}",
                operacional=_falha_operacional(excecao),
            ) from excecao
        return self._validar(vetores, len(textos))

    def embutir_consulta(self, texto: str) -> list[float]:
        try:
            vetor = self._cliente.embed_query(texto)
        except Exception as excecao:
            raise ErroProvedorEmbedding(
                f"falha do provedor de embedding em modo query: {excecao}",
                operacional=_falha_operacional(excecao),
            ) from excecao
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
                    f"falha ao inicializar o reranker NVIDIA: {excecao}",
                    operacional=_falha_operacional(excecao),
                ) from excecao
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
                f"falha do reranker NVIDIA: {excecao}",
                operacional=_falha_operacional(excecao),
            ) from excecao
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
                f"resposta do reranker viola o contrato de índices e scores: {erro}",
                operacional=False,
            ) from erro
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
            cliente = base.with_structured_output(OrdenacaoListwise, method="json_schema")
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
                    f"falha do fallback listwise: {excecao}",
                    operacional=_falha_operacional(excecao),
                ) from excecao
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
