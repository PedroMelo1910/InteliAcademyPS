"""Leitura humana de um ``ItemRanking``: um rótulo por estado de análise.

A tela não recalcula nada — ela nomeia. Os quatro estados que a aplicação pode
devolver (concluída aderente, concluída non-AI, evidência insuficiente e
análise ausente) precisam de quatro leituras diferentes, porque confundir zero
validado com ausência de análise é exatamente o erro que o projeto não pode
cometer na frente de um avaliador.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - só para tipagem
    from collections.abc import Iterable

    from radar.aplicacao import ItemRanking


LOCALIZACAO_AUSENTE = "não informada"
DESCRICAO_AUSENTE = "sem descrição curta na base"

ROTULO_CONCLUIDA = "Análise concluída"
ROTULO_CONCLUIDA_NON_AI = "Análise concluída — empresa sem IA no produto"
ROTULO_EVIDENCIA_INSUFICIENTE = "Evidência insuficiente"
ROTULO_AUSENTE = "Análise ainda não disponível"

EXPLICACAO_CONCLUIDA = (
    "Classe e fit-score foram validados pela rubrica e ficaram gravados no cache."
)
EXPLICACAO_CONCLUIDA_NON_AI = (
    "O zero é o resultado validado da rubrica para uma empresa que não usa IA "
    "no produto."
)
EXPLICACAO_EVIDENCIA_INSUFICIENTE = (
    "A base pública não sustentou classe nem pontuação para esta empresa."
)
EXPLICACAO_AUSENTE = (
    "Esta candidata ainda não tem análise gravada; nenhuma classe e nenhuma "
    "pontuação são exibidas."
)


@dataclass(frozen=True)
class ResumoCandidata:
    """Tudo que o cartão de uma candidata precisa, já formatado."""

    posicao: int
    id_startup: int
    nome: str
    descricao: str
    setor: str
    estagio: str
    localizacao: str
    rotulo_status: str
    explicacao_status: str
    classe: str | None
    fit_score: str | None
    justificativa_fit_score: str | None
    motivo_evidencia_insuficiente: str | None
    bm25: str


def resumir_candidata(item: ItemRanking) -> ResumoCandidata:
    """Traduz um item do ranking sem reordenar, recalcular ou completar lacuna."""
    rotulo, explicacao = _estado_legivel(item)
    concluida = item.status_analise == "concluida"
    return ResumoCandidata(
        posicao=item.posicao,
        id_startup=item.empresa.id_startup,
        nome=item.empresa.nome,
        descricao=item.empresa.descricao_curta or DESCRICAO_AUSENTE,
        setor=item.empresa.setor,
        estagio=item.empresa.estagio,
        localizacao=item.empresa.localizacao or LOCALIZACAO_AUSENTE,
        rotulo_status=rotulo,
        explicacao_status=explicacao,
        classe=item.classe if concluida else None,
        fit_score=(
            f"{item.fit_score_total}/100"
            if concluida and item.fit_score_total is not None
            else None
        ),
        justificativa_fit_score=item.justificativa_fit_score if concluida else None,
        motivo_evidencia_insuficiente=(
            item.motivo_evidencia_insuficiente
            if item.status_analise == "evidencia_insuficiente"
            else None
        ),
        bm25=f"{item.melhor_score_bm25:.6f}",
    )


def resumir_ranking(itens: Iterable[ItemRanking]) -> tuple[ResumoCandidata, ...]:
    """Preserva a ordem entregue pela aplicação, item a item."""
    return tuple(resumir_candidata(item) for item in itens)


def _estado_legivel(item: ItemRanking) -> tuple[str, str]:
    if item.status_analise == "concluida":
        if item.classe == "non-AI":
            return ROTULO_CONCLUIDA_NON_AI, EXPLICACAO_CONCLUIDA_NON_AI
        return ROTULO_CONCLUIDA, EXPLICACAO_CONCLUIDA
    if item.status_analise == "evidencia_insuficiente":
        return ROTULO_EVIDENCIA_INSUFICIENTE, EXPLICACAO_EVIDENCIA_INSUFICIENTE
    return ROTULO_AUSENTE, EXPLICACAO_AUSENTE


def atualizar_item_com_briefing(item: ItemRanking, briefing) -> ItemRanking:
    """Reescreve só o veredito do cartão a partir de um ``Briefing`` validado.

    Nada é recalculado: classe e fit-score vêm do ``veredito`` que o nó
    Briefing já validou, e o motivo da insuficiência vem do aviso que a §11.3
    obriga a nomear a causa. Posição, empresa, BM25 e documentos ficam
    intactos — a atualização é do resultado da análise, não do ranking.

    ``justificativa_fit_score`` é zerada de propósito: ela pertencia à análise
    anterior do cache e o Briefing não carrega equivalente. Manter a antiga ao
    lado de um score novo seria justificar uma pontuação com o argumento de
    outra.
    """
    if briefing.variante == "evidencia_insuficiente":
        return replace(
            item,
            status_analise="evidencia_insuficiente",
            classe=None,
            fit_score_total=None,
            justificativa_fit_score=None,
            motivo_evidencia_insuficiente=(
                briefing.avisos[0] if briefing.avisos else None
            ),
        )
    return replace(
        item,
        status_analise="concluida",
        classe=briefing.veredito.classe,
        fit_score_total=briefing.veredito.fit_score_total,
        justificativa_fit_score=None,
        motivo_evidencia_insuficiente=None,
    )


# ----------------------------------------------------------------------
# Apoio visual: cor do selo, contagem por estado e escala da barra
# ----------------------------------------------------------------------

# O verde pertence ao eixo da aderência NVIDIA e só aparece quando a rubrica
# concluiu com lastro. Âmbar nomeia a ausência de lastro; cinza cobre tanto a
# conclusão non-AI quanto a análise que ainda não existe — nesses dois casos
# quem diferencia é o rótulo, não a cor.
TOM_DO_STATUS = {
    ROTULO_CONCLUIDA: "green",
    ROTULO_CONCLUIDA_NON_AI: "gray",
    ROTULO_EVIDENCIA_INSUFICIENTE: "orange",
    ROTULO_AUSENTE: "gray",
}

TOM_DA_CLASSE = "violet"

TOTAL_MAXIMO_FIT_SCORE = 100


def tom_do_status(resumo: ResumoCandidata) -> str:
    """Cor do selo de situação, derivada do rótulo já decidido acima."""
    return TOM_DO_STATUS[resumo.rotulo_status]


@dataclass(frozen=True)
class ContagemDoRanking:
    """Quantas candidatas há em cada estado de análise, para o painel."""

    total: int
    concluidas: int
    sem_lastro: int
    ausentes: int


def contar_estados(itens: Iterable[ItemRanking]) -> ContagemDoRanking:
    """Conta por estado sem reordenar, reclassificar nem completar lacuna."""
    estados = [item.status_analise for item in itens]
    return ContagemDoRanking(
        total=len(estados),
        concluidas=estados.count("concluida"),
        sem_lastro=estados.count("evidencia_insuficiente"),
        ausentes=estados.count("ausente"),
    )


def fracao_do_fit_score(total: int) -> float:
    """Converte o total já validado (0 a 100) na fração que a barra desenha.

    Não é recálculo e não é renormalização: a pontuação exibida continua sendo
    exatamente a que o Briefing trouxe. Só a unidade muda, porque a barra do
    Streamlit desenha de 0 a 1. A conversão mora aqui, e não na tela, para que
    ``app.py`` não faça aritmética nenhuma sobre pontuação.
    """
    return total / TOTAL_MAXIMO_FIT_SCORE


ROTULO_FUNDAMENTO = {
    "gap_confirmado": "Gap endereçado",
    "oportunidade_confirmada": "Oportunidade confirmada",
}


def rotular_fundamento(recomendacao) -> str:
    """Nomeia o fundamento sem transformar carga de trabalho em deficiência."""
    rotulo = ROTULO_FUNDAMENTO[recomendacao.tipo_fundamento]
    return f"{rotulo}: {recomendacao.identificador_fundamento}"
