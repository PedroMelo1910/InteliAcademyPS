from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from radar.contratos import (
    MAXIMO_FRASES_JUSTIFICATIVA,
    MINIMO_FRASES_JUSTIFICATIVA,
    Classificacao,
    EstadoRadar,
    PerfilExtraido,
)
from radar.provedores import ProvedorClassificacao


# Campos de estado derivados da classificação. O nó invalida cada um deles a
# cada execução para que uma reclassificação nunca conviva com um downstream
# velho: no laço de reextração, um perfil validado antigo descreveria
# afirmações que já não existem.
CAMPOS_DERIVADOS_DA_CLASSIFICACAO: tuple[str, ...] = (
    "perfil_validado",
    "confianca_perfil",
    "contexto_nvidia",
    "recomendacoes",
    "fit_score",
    "briefing",
)

_PADRAO_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")
_PADRAO_LISTA_AFIRMACOES = re.compile(
    r"\bafirmaç(?:ão|ões)\s+\["
    r"(?P<ids>\d+\s*(?:(?:,|e)\s*\d+\s*)*)\]",
    flags=re.IGNORECASE,
)

# Uma classificação non-AI precisa apontar o que a empresa efetivamente entrega,
# não apenas uma ausência de IA combinada com fatos contextuais sobre time,
# financiamento, escala ou dor operacional.
_CATEGORIAS_EVIDENCIA_DO_PRODUTO = frozenset(
    {
        "workflow_profundo",
        "distribuicao",
        "otimizacao_tecnica",
        "stack_propria",
        "dependencia_api_externa",
    }
)
# ``dados_proprietarios`` demonstra ativo/defensabilidade, mas não descreve por
# si só o produto vendido. ``outro`` é coringa sem semântica suficiente para
# servir como evidência positiva do produto em uma decisão non-AI.


class ErroClassificador(RuntimeError):
    """Falha segura: não inventa classe nem grava classificação fabricada."""


class Classifier:
    """Decide a maturidade de IA da empresa a partir do perfil já extraído.

    A entrada é exclusivamente o ``PerfilExtraido``: o nó não recebe a linha
    curada, o texto integral dos documentos nem ``classe_referencia``, que
    seria justamente a resposta que ele precisa produzir sozinho.
    """

    def __init__(self, provedor: ProvedorClassificacao):
        self.provedor = provedor

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        perfil = self._perfil(estado)
        classificacao = self._classificar_com_validacao(perfil)
        saida: dict[str, Any] = {
            "classificacao": classificacao,
            "trajeto": ["classifier"],
        }
        for campo in CAMPOS_DERIVADOS_DA_CLASSIFICACAO:
            saida[campo] = None
        return saida

    # ------------------------------------------------------------------
    # Pré-condições do nó
    # ------------------------------------------------------------------

    @staticmethod
    def _perfil(estado: EstadoRadar) -> PerfilExtraido:
        bruto = estado.get("perfil_extraido")
        if bruto is None:
            raise ErroClassificador(
                "o Classifier exige um PerfilExtraido no estado"
            )
        try:
            return PerfilExtraido.model_validate(bruto)
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroClassificador(
                f"o perfil recebido não respeita o contrato do Extractor: {erro}"
            ) from erro

    # ------------------------------------------------------------------
    # Fronteira do LLM
    # ------------------------------------------------------------------

    def _classificar_com_validacao(self, perfil: PerfilExtraido) -> Classificacao:
        erro_anterior: str | None = None
        for tentativa in range(2):
            mensagens = self._montar_mensagens(perfil, erro_anterior)
            try:
                bruto = self.provedor.invocar(mensagens)
            except (ValidationError, ValueError, TypeError) as exc:
                # O adaptador de structured output pode validar com Pydantic antes
                # de devolver o objeto; a resposta continua fora do contrato e
                # consome o mesmo retry corretivo.
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroClassificador(self._MENSAGEM_FALHA_DUPLA) from exc
                continue
            except Exception as exc:
                raise ErroClassificador(
                    "O Gemini não respondeu ao Classifier; "
                    "nenhuma classificação foi fabricada."
                ) from exc
            try:
                return self._validar(bruto, perfil)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroClassificador(self._MENSAGEM_FALHA_DUPLA) from exc
        raise AssertionError("laço de validação terminou em estado impossível")

    _MENSAGEM_FALHA_DUPLA = (
        "O Gemini respondeu duas vezes fora do contrato estruturado; "
        "nenhuma classificação foi gravada no estado."
    )

    # ------------------------------------------------------------------
    # Regras de sustentação
    # ------------------------------------------------------------------

    @classmethod
    def _validar(cls, bruto: object, perfil: PerfilExtraido) -> Classificacao:
        classificacao = Classificacao.model_validate(bruto)
        por_id = {
            afirmacao.id_afirmacao: afirmacao for afirmacao in perfil.afirmacoes
        }

        estranhos = [
            id_afirmacao
            for id_afirmacao in classificacao.ids_afirmacoes_suporte
            if id_afirmacao not in por_id
        ]
        if estranhos:
            raise ValueError(
                f"ids_afirmacoes_suporte cita afirmações inexistentes no perfil: "
                f"{estranhos}; ids disponíveis: {sorted(por_id)}"
            )

        suporte = [
            por_id[id_afirmacao]
            for id_afirmacao in classificacao.ids_afirmacoes_suporte
        ]
        evidencia_positiva_do_produto = any(
            afirmacao.polaridade != "ausencia_explicita"
            and afirmacao.categoria in _CATEGORIAS_EVIDENCIA_DO_PRODUTO
            for afirmacao in suporte
        )
        if classificacao.classe == "non-AI" and not evidencia_positiva_do_produto:
            raise ValueError(
                "non-AI exige evidência positiva sobre o produto que a empresa "
                "vende; ausência declarada e fatos apenas contextuais não sustentam "
                "a classe"
            )

        nao_sustentados = cls._numeros_nao_sustentados(classificacao, perfil)
        if nao_sustentados:
            raise ValueError(
                "a justificativa cita números que não aparecem nas afirmações "
                "selecionadas como suporte: "
                f"{nao_sustentados}"
            )
        return classificacao

    @staticmethod
    def _numeros_nao_sustentados(
        classificacao: Classificacao, perfil: PerfilExtraido
    ) -> list[str]:
        """Guarda determinística contra o fato fabricado mais verificável.

        Não é verificação de implicação completa — essa parte fica no prompt —
        mas um número ausente das afirmações selecionadas é sempre um fato sem
        rastreabilidade no suporte declarado.
        """
        por_id = {
            str(afirmacao.id_afirmacao): afirmacao
            for afirmacao in perfil.afirmacoes
        }
        ids_validos = {
            str(id_afirmacao)
            for id_afirmacao in classificacao.ids_afirmacoes_suporte
        }
        sustentados: set[str] = set()
        for id_afirmacao in classificacao.ids_afirmacoes_suporte:
            afirmacao = por_id[str(id_afirmacao)]
            sustentados.update(_PADRAO_NUMERO.findall(afirmacao.texto))
            sustentados.update(_PADRAO_NUMERO.findall(afirmacao.trecho_citado))

        posicoes_de_ids: set[int] = set()
        for referencia in _PADRAO_LISTA_AFIRMACOES.finditer(
            classificacao.justificativa
        ):
            inicio_ids = referencia.start("ids")
            for numero in _PADRAO_NUMERO.finditer(referencia.group("ids")):
                if numero.group() in ids_validos:
                    posicoes_de_ids.add(inicio_ids + numero.start())

        nao_sustentados: list[str] = []
        for numero in _PADRAO_NUMERO.finditer(classificacao.justificativa):
            if (
                numero.group() not in sustentados
                and numero.start() not in posicoes_de_ids
            ):
                nao_sustentados.append(numero.group())
        return nao_sustentados

    @staticmethod
    def _resumir_erro(erro: Exception) -> str:
        if isinstance(erro, ValidationError):
            return "; ".join(
                f"{'.'.join(str(item) for item in falha['loc']) or 'classificacao'}: "
                f"{falha['msg']}"
                for falha in erro.errors()
            )
        return str(erro)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _instrucao(ids_disponiveis: list[int]) -> str:
        return (
            "Você é o Classifier do NVIDIA Startup AI Radar. A partir do perfil "
            "abaixo, decida a maturidade de IA da empresa e produza uma "
            "Classificacao estritamente estruturada.\n"
            "Definições operacionais, nesta ordem de leitura:\n"
            "- AI-native: a IA é o produto; sem os modelos, o produto da empresa "
            "não se sustenta.\n"
            "- AI-enabled: o produto existe sem IA; a IA é uma camada funcional "
            "adicionada sobre ele.\n"
            "- non-AI: não há componente de IA relevante no produto.\n"
            "Regras da resposta:\n"
            "- classe: exatamente um destes três valores: AI-native, AI-enabled, "
            "non-AI.\n"
            f"- justificativa: de {MINIMO_FRASES_JUSTIFICATIVA} a "
            f"{MAXIMO_FRASES_JUSTIFICATIVA} frases terminadas em pontuação.\n"
            "- ids_afirmacoes_suporte: ao menos um id, sem repetição, sempre "
            f"entre os ids deste perfil: {ids_disponiveis}.\n"
            "- Se mencionar ids dentro da justificativa, use somente o formato "
            "delimitado 'afirmação [1]' ou 'afirmações [1 e 2]'; a lista "
            "ids_afirmacoes_suporte continua sendo a referência oficial.\n"
            "- Toda conclusão precisa se apoiar nas afirmações citadas; não "
            "introduza números, nomes ou fatos ausentes do perfil.\n"
            "- Desconhecido não é ausência: se o perfil não menciona um tema, "
            "isso não prova que a empresa não o tem.\n"
            "- non-AI exige evidência positiva sobre o produto que a empresa "
            "realmente vende, e não apenas a falta de menções a IA.\n"
            "- Um jargão de tecnologia isolado não torna a empresa AI-native.\n"
            "- Ferramenta interna ou programa genérico de treinamento em IA para "
            "o time não torna AI-enabled o produto entregue ao cliente.\n"
            "- Não avalie a qualidade das evidências nem produza recomendação; "
            "isso não é tarefa do Classifier."
        )

    @staticmethod
    def _dados(perfil: PerfilExtraido) -> str:
        blocos = "\n\n".join(
            f"[afirmação {afirmacao.id_afirmacao} | categoria: {afirmacao.categoria} "
            f"| polaridade: {afirmacao.polaridade}]\n{afirmacao.texto}\n"
            f'Trecho citado: "{afirmacao.trecho_citado}"'
            for afirmacao in perfil.afirmacoes
        )
        return (
            f"Startup analisada: id {perfil.id_startup}.\n\n"
            f"Resumo do produto: {perfil.resumo_produto}\n\n"
            f"Afirmações do perfil:\n\n{blocos}"
        )

    def _montar_mensagens(
        self, perfil: PerfilExtraido, erro_anterior: str | None
    ) -> list[tuple[str, str]]:
        ids_disponiveis = [afirmacao.id_afirmacao for afirmacao in perfil.afirmacoes]
        mensagens = [
            ("system", self._instrucao(ids_disponiveis)),
            ("human", self._dados(perfil)),
        ]
        if erro_anterior:
            mensagens.append(
                (
                    "system",
                    "A resposta anterior violou o contrato. Corrija sem texto livre. "
                    f"Falha de validação: {erro_anterior}",
                )
            )
        return mensagens
