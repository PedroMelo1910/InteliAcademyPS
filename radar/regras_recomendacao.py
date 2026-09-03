"""Regras determinísticas do nó Recommendation: tabela, prioridade e complexidade.

Este módulo não chama LLM, não consulta banco e não lê relógio. Ele existe
separado de ``radar.recomendacao`` para manter intacta a função pura de
fit-score já aprovada: aqui ficam apenas as regras que o nó aplica **depois**
de o LLM ter escolhido gap, tecnologias e ids.

Fontes das regras, nesta ordem: TAPI §5.4/§5.5 (as 16 tecnologias e os
exemplos de recomendação) e a arquitetura §10.1 (tabela gap → candidatas),
§10.2 (escala de prioridade) e §10.3 (escala de complexidade).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import get_args

from radar.contratos import (
    TECNOLOGIAS_NVIDIA,
    ComplexidadeRecomendacao,
    GapEnderecado,
    PerfilValidado,
    PrioridadeRecomendacao,
    TecnologiaNvidia,
)

# ``_pontos_estagio`` é a única normalização de estágio do repositório. A
# prioridade (§10.2) e o pilar Momento (§7.2) usam a mesma janela seed/série A;
# manter duas cópias sutilmente diferentes faria a régua do ranking divergir da
# régua da prioridade sem que nenhum teste percebesse.
from radar.recomendacao import _pontos_estagio


GAPS_ENDERECAVEIS: tuple[str, ...] = get_args(GapEnderecado)
COMPLEXIDADES_RECOMENDACAO: tuple[str, ...] = get_args(ComplexidadeRecomendacao)

# A escada é declarada aqui, e não inferida da ordem do ``Literal``: o ajuste
# do §10.3 sobe um degrau, e essa direção precisa ser um fato do módulo.
ESCADA_COMPLEXIDADE: tuple[ComplexidadeRecomendacao, ...] = ("baixa", "media", "alta")

# Pontuação que ``_pontos_estagio`` atribui exatamente a seed e série A.
PONTOS_ESTAGIO_JANELA_DE_INFLEXAO = 5

# Ponto de corte do ajuste único de complexidade (§10.3): sem sinal de equipe
# técnica de IA própria, o esforço de adoção sobe um degrau.
TETO_CENTRALIDADE_SEM_EQUIPE_DE_IA = 3

# As duas categorias que o §10.2 trata como dor ativa documentada.
CATEGORIAS_DE_DOR: frozenset[str] = frozenset(
    {"dependencia_api_externa", "escala_e_dor_operacional"}
)


class ErroRegraRecomendacao(ValueError):
    """Entrada fora dos catálogos fechados; nenhuma regra é aplicada por padrão."""


# ----------------------------------------------------------------------
# §10.1 — tabela fixa gap → tecnologias candidatas
# ----------------------------------------------------------------------
#
# Derivada dos exemplos do TAPI §5.5 e das quatro colunas estruturais do §1.1.
# O LLM escolhe de 1 a 3 tecnologias **dentro** do conjunto do gap escolhido;
# qualquer escolha fora dele é descartada pelo nó. O programa NVIDIA Inception
# aparece só em ``distribuicao`` porque é alavanca de go-to-market, comunidade
# e créditos (TAPI §5.4) — a dimensão de programa também é endereçada pelo
# catálogo de ``proxima_acao``, e não precisa contaminar os demais gaps.
TECNOLOGIAS_POR_GAP: dict[GapEnderecado, tuple[TecnologiaNvidia, ...]] = {
    "dados_proprietarios": (
        "NVIDIA RAPIDS",
        "cuDF",
        "cuML",
        "NVIDIA NeMo",
        "NVIDIA AI Enterprise",
    ),
    "workflow_profundo": (
        "NVIDIA NIM",
        "NVIDIA NeMo",
        "NeMo Guardrails",
        "NVIDIA Riva",
        "NVIDIA Clara",
        "NVIDIA Isaac",
        "NVIDIA Omniverse",
        "NVIDIA Morpheus",
        "NVIDIA AI Enterprise",
    ),
    "distribuicao": (
        "NVIDIA Inception",
        "NVIDIA AI Enterprise",
    ),
    "otimizacao_tecnica": (
        "NVIDIA Triton Inference Server",
        "TensorRT-LLM",
        "NVIDIA NIM",
        "CUDA",
        "NVIDIA RAPIDS",
    ),
    "dependencia_api_externa": (
        "NVIDIA NIM",
        "NeMo Guardrails",
        "NVIDIA Triton Inference Server",
        "NVIDIA NeMo",
        "NVIDIA AI Enterprise",
    ),
    "escala_e_dor_operacional": (
        "NVIDIA Triton Inference Server",
        "NVIDIA NIM",
        "TensorRT-LLM",
        "NVIDIA RAPIDS",
        "cuDF",
        "cuML",
        "NVIDIA AI Enterprise",
    ),
}


# ----------------------------------------------------------------------
# §10.3 — tabela fixa de complexidade de adoção das 16 tecnologias
# ----------------------------------------------------------------------
COMPLEXIDADE_POR_TECNOLOGIA: dict[TecnologiaNvidia, ComplexidadeRecomendacao] = {
    # Baixa: adoção via programa, sem mudança de infraestrutura.
    "NVIDIA Inception": "baixa",
    # Média: exige GPU ou integração própria, com interface de alto nível.
    "NVIDIA NIM": "media",
    "NeMo Guardrails": "media",
    "NVIDIA Triton Inference Server": "media",
    "NVIDIA RAPIDS": "media",
    "cuDF": "media",
    "cuML": "media",
    "NVIDIA Riva": "media",
    "NVIDIA AI Enterprise": "media",
    # Alta: re-arquitetura, competência nova ou plataforma de domínio.
    "TensorRT-LLM": "alta",
    "NVIDIA NeMo": "alta",
    "CUDA": "alta",
    "NVIDIA Omniverse": "alta",
    "NVIDIA Isaac": "alta",
    "NVIDIA Clara": "alta",
    "NVIDIA Morpheus": "alta",
}


def tecnologias_candidatas(gap: str) -> tuple[TecnologiaNvidia, ...]:
    """Conjunto permitido para o gap; recusa gap fora do enum do contrato."""
    try:
        return TECNOLOGIAS_POR_GAP[gap]  # type: ignore[index]
    except KeyError as erro:
        raise ErroRegraRecomendacao(
            f"gap {gap!r} não pertence ao enum de gaps endereçáveis: "
            f"{list(GAPS_ENDERECAVEIS)}"
        ) from erro


def gaps_sustentados(perfil: PerfilValidado) -> dict[GapEnderecado, frozenset[int]]:
    """Gaps que a evidência confirmada sustenta, com os ids que os sustentam.

    Uma dimensão estrutural só entra quando o Evidence Validator a marcou como
    ``gap_confirmado``. ``desconhecido`` — inclusive o desconhecido que nasce de
    conflito entre presença e ausência confirmadas — e ``capacidade_confirmada``
    nunca viram gap: desconhecimento não é ausência, e capacidade observada é o
    oposto de lacuna.

    As duas categorias de dor não são dimensões do relatório de gap; elas entram
    quando existe ao menos uma afirmação **confirmada** exatamente naquela
    categoria. Afirmação derrubada não sustenta nada.

    A ordem do resultado segue ``GAPS_ENDERECAVEIS``, para que duas execuções
    iguais produzam o mesmo prompt e a mesma mensagem de erro.
    """
    estados = {item.dimensao: item for item in perfil.estado_dimensoes_gap}
    confirmadas = [
        item
        for item in perfil.afirmacoes_validadas
        if item.situacao == "confirmada"
    ]
    sustentados: dict[GapEnderecado, frozenset[int]] = {}
    for gap in GAPS_ENDERECAVEIS:
        estado = estados.get(gap)
        if estado is not None:
            if estado.estado == "gap_confirmado":
                sustentados[gap] = frozenset(estado.ids_evidencias)
            continue
        ids = frozenset(
            item.id_afirmacao for item in confirmadas if item.categoria == gap
        )
        if ids:
            sustentados[gap] = ids
    return sustentados


def conferir_gap_sustentado(
    gap: str,
    ids_citados: Iterable[int],
    sustentados: dict[GapEnderecado, frozenset[int]],
) -> None:
    """Liga a evidência citada ao gap escolhido; sem esse elo não há proveniência.

    Conferir separadamente que a tecnologia pertence ao gap e que cada id é de
    uma afirmação confirmada não basta: as duas conferências passam mesmo quando
    a afirmação citada não diz nada sobre aquele gap. É este elo que impede uma
    recomendação de tomar emprestada a proveniência de uma evidência alheia.
    """
    ids_do_gap = sustentados.get(gap)  # type: ignore[arg-type]
    if not ids_do_gap:
        raise ErroRegraRecomendacao(
            f"o gap {gap!r} não está sustentado por evidência confirmada neste "
            f"perfil; gaps sustentados: {sorted(sustentados)}"
        )
    citados = set(ids_citados)
    if not citados & ids_do_gap:
        raise ErroRegraRecomendacao(
            f"as afirmações {sorted(citados)} não sustentam o gap {gap!r}; "
            f"sustentam-no apenas {sorted(ids_do_gap)}"
        )


def estagio_na_janela_de_inflexao(estagio: str) -> bool:
    """Janela seed/série A do §10.2, lida pela mesma régua do pilar Momento."""
    return _pontos_estagio(estagio) == PONTOS_ESTAGIO_JANELA_DE_INFLEXAO


def calcular_prioridade(
    categorias_citadas: Iterable[str],
    estagio: str,
    gap_confirmado: bool,
) -> PrioridadeRecomendacao:
    """Prioridade do §10.2: urgência para o time NVIDIA, nunca escrita pelo LLM.

    A ordem dos três predicados é o contrato. ``categorias_citadas`` são as
    categorias das afirmações que a própria recomendação cita — não o perfil
    inteiro: prioridade alta exige que a dor esteja na evidência citada.
    """
    tem_dor_documentada = bool(CATEGORIAS_DE_DOR & set(categorias_citadas))
    if tem_dor_documentada:
        return "alta" if estagio_na_janela_de_inflexao(estagio) else "media"
    if gap_confirmado:
        return "media"
    # `baixa` pertence à regra pura (§10.2), mas o nó normal não a alcança: ele
    # só recomenda gap sustentado, e a linha "Média" do §10.2 já cobre gap
    # confirmado sem dor. O nível fica reservado a recomendação de
    # aperfeiçoamento sem gap sustentado, que este marco não produz.
    return "baixa"


def calcular_complexidade(
    tecnologias: Iterable[str],
    pontos_centralidade_ia: int,
) -> ComplexidadeRecomendacao:
    """Complexidade do §10.3: maior degrau do pacote, com ajuste único por P1.

    O ajuste sobe um degrau quando o pilar de Centralidade de IA não passa de
    ``TETO_CENTRALIDADE_SEM_EQUIPE_DE_IA`` — sinal de que não há equipe técnica
    de IA própria para absorver a adoção. Nunca desce.
    """
    pacote = list(tecnologias)
    if not pacote:
        raise ErroRegraRecomendacao(
            "o pacote precisa conter ao menos uma tecnologia para ser avaliado"
        )
    desconhecidas = sorted(
        set(pacote) - set(COMPLEXIDADE_POR_TECNOLOGIA), key=repr
    )
    if desconhecidas:
        raise ErroRegraRecomendacao(
            f"tecnologias fora do catálogo das {len(TECNOLOGIAS_NVIDIA)} do TAPI: "
            f"{desconhecidas}"
        )
    degrau = max(
        ESCADA_COMPLEXIDADE.index(COMPLEXIDADE_POR_TECNOLOGIA[tecnologia])
        for tecnologia in pacote
    )
    if pontos_centralidade_ia <= TETO_CENTRALIDADE_SEM_EQUIPE_DE_IA:
        degrau += 1
    return ESCADA_COMPLEXIDADE[min(degrau, len(ESCADA_COMPLEXIDADE) - 1)]
