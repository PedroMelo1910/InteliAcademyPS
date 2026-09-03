from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from radar.contratos import (
    ContextoNvidia,
    EmpresaCandidata,
    EstadoRadar,
    PerfilValidado,
    ResultadoRecuperacao,
)
from radar.provedores import ProvedorContextoNvidia


# O contexto NVIDIA é a entrada da recomendação e do fit-score. Trocá-lo sem
# invalidar o que dele descende deixaria uma recomendação citando chunks de
# outra recuperação — exatamente a proveniência que o marco precisa garantir.
CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA: tuple[str, ...] = (
    "recomendacoes",
    "fit_score",
)

# Só estas duas categorias descrevem dor operacional documentada; as demais
# entram na consulta apenas quando a dimensão correspondente virou gap.
CATEGORIAS_DE_DOR_NA_CONSULTA: tuple[str, ...] = (
    "dependencia_api_externa",
    "escala_e_dor_operacional",
)

# A consulta precisa ser específica sem virar um despejo do perfil inteiro: o
# corte mantém a recuperação comparável entre execuções.
MAXIMO_EVIDENCIAS_NA_CONSULTA = 6


class ErroRagNvidia(RuntimeError):
    """Falha segura: sem contexto válido não há recomendação nem fit-score."""


def _rotulo(identificador: str) -> str:
    """Identificador do contrato em forma legível para busca lexical e embedding."""
    return identificador.replace("_", " ")


def montar_consulta_nvidia(
    perfil: PerfilValidado, empresa: EmpresaCandidata
) -> str:
    """Deriva a consulta apenas do perfil validado e dos dados já aprovados.

    A assinatura é a garantia estrutural mais forte deste nó: não existe
    parâmetro por onde ``classe_referencia`` — o rótulo de curadoria que o
    núcleo nunca pode ler — entraria na consulta. Afirmação derrubada não
    contribui, e dimensão com capacidade confirmada não vira gap.
    """
    gaps = [
        item.dimensao
        for item in perfil.estado_dimensoes_gap
        if item.estado == "gap_confirmado"
    ]
    confirmadas = [
        item
        for item in perfil.afirmacoes_validadas
        if item.situacao == "confirmada"
    ]
    dores = [
        categoria
        for categoria in CATEGORIAS_DE_DOR_NA_CONSULTA
        if any(item.categoria == categoria for item in confirmadas)
    ]
    categorias_relevantes = set(gaps) | set(dores)
    evidencias = [
        item.texto
        for item in sorted(confirmadas, key=lambda item: item.id_afirmacao)
        if item.categoria in categorias_relevantes
    ][:MAXIMO_EVIDENCIAS_NA_CONSULTA]

    partes = [
        f"tecnologias NVIDIA para o setor {empresa.setor}",
        "gaps confirmados: "
        + (", ".join(_rotulo(gap) for gap in gaps) or "nenhum"),
        "dores documentadas: "
        + (", ".join(_rotulo(dor) for dor in dores) or "nenhuma"),
    ]
    if evidencias:
        partes.append("evidências: " + " ".join(evidencias))
    return "; ".join(partes)


class NvidiaRag:
    """Adapta a recuperação NVIDIA existente ao grafo, depois do R3.

    O nó não conhece FTS5, embedding, fusão RRF, reranking nem fallback: tudo
    isso continua dentro de ``ConhecimentoNvidia``, atrás do protocolo
    injetado. Aqui só existem três responsabilidades: montar a consulta a
    partir do estado aprovado, delegar, e recusar qualquer resposta que não
    seja um ``ContextoNvidia`` válido.
    """

    def __init__(self, consultor: ProvedorContextoNvidia):
        self.consultor = consultor

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        perfil = _validar_entrada(
            estado.get("perfil_validado"),
            PerfilValidado,
            "o NVIDIA RAG exige um PerfilValidado no estado",
        )
        recuperacao = _validar_entrada(
            estado.get("resultado_recuperacao"),
            ResultadoRecuperacao,
            "o NVIDIA RAG exige um ResultadoRecuperacao no estado",
        )
        empresa = self._empresa(estado, recuperacao)
        contexto = self._recuperar(montar_consulta_nvidia(perfil, empresa))

        # O checkpointer do LangGraph serializa o estado com msgpack, que não conhece o
        # ``AnyHttpUrl`` do Pydantic: qualquer modelo com URL tipada — ``TrechoNvidia``,
        # ``ContextoNvidia`` e ``Recomendacao`` — quebra a gravação do checkpoint se for
        # gravado como instância. O contrato não muda: o nó grava a forma JSON do mesmo
        # modelo, e todo consumidor continua validando com ``model_validate``. É a
        # mesma escolha que ``DocumentoRecuperado.url_fonte`` já faz ao ser ``str``.
        saida: dict[str, Any] = {
            "contexto_nvidia": contexto.model_dump(mode="json"),
            "trajeto": ["nvidia_rag"],
        }
        for campo in CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA:
            saida[campo] = None
        return saida

    @staticmethod
    def _empresa(
        estado: EstadoRadar, recuperacao: ResultadoRecuperacao
    ) -> EmpresaCandidata:
        selecionada = estado.get("startup_selecionada")
        if selecionada is None:
            raise ErroRagNvidia(
                "o NVIDIA RAG só roda no aprofundamento de uma startup selecionada"
            )
        for empresa in recuperacao.empresas:
            if empresa.id_startup == int(selecionada):
                return empresa
        raise ErroRagNvidia(
            f"a startup {selecionada} não pertence ao conjunto recuperado desta "
            "análise; nenhuma consulta NVIDIA foi montada"
        )

    def _recuperar(self, consulta: str) -> ContextoNvidia:
        try:
            bruto = self.consultor.consultar(consulta)
        except Exception as erro:
            # Indisponibilidade, índice ausente ou provedor fora do ar param a
            # execução: nenhuma passagem NVIDIA é inventada como substituta.
            raise ErroRagNvidia(
                "a recuperação NVIDIA falhou e nenhuma passagem foi fabricada "
                f"em substituição: {erro}"
            ) from erro
        try:
            return ContextoNvidia.model_validate(bruto)
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroRagNvidia(
                "o consultor NVIDIA devolveu um contexto fora do contrato; "
                f"nada foi gravado no estado: {erro}"
            ) from erro


def _validar_entrada(bruto: object, contrato, mensagem: str):
    if bruto is None:
        raise ErroRagNvidia(mensagem)
    try:
        return contrato.model_validate(bruto)
    except (ValidationError, ValueError, TypeError) as erro:
        raise ErroRagNvidia(
            f"{mensagem}; recebido fora do contrato: {erro}"
        ) from erro
