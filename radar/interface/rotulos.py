"""Leitura humana de um ``ItemRanking``: um rótulo por estado de análise.

A tela não recalcula nada — ela nomeia. Os quatro estados que a aplicação pode
devolver (concluída aderente, concluída non-AI, evidência insuficiente e
análise ausente) precisam de quatro leituras diferentes, porque confundir zero
validado com ausência de análise é exatamente o erro que o projeto não pode
cometer na frente de um avaliador.
"""

from __future__ import annotations

import re
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
    "A classificação e a pontuação foram verificadas e salvas nesta análise."
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

# A rubrica usa uma escala pública comum, mas os tetos brutos são diferentes:
# momento chega a 9 e alinhamento setorial a 7. Exibir todos como "/10" faria
# a interface prometer pontos que a função determinística nunca pode conceder.
MAXIMO_POR_PILAR = {
    "centralidade_ia": 10,
    "gap_enderecavel": 10,
    "momento": 9,
    "alinhamento_setorial": 7,
}


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


def maximo_do_pilar(pilar: str) -> int:
    """Retorna o teto bruto congelado da rubrica para apresentação."""
    return MAXIMO_POR_PILAR[pilar]


ROTULO_FUNDAMENTO = {
    "gap_confirmado": "Necessidade confirmada",
    "oportunidade_confirmada": "Oportunidade confirmada",
}


# Os contratos mantêm seus identificadores estáveis. Só a apresentação os
# traduz: a mesma leitura deve aparecer na tela e no arquivo baixado.
ROTULOS = {
    "query_planner": "Entendimento da busca",
    "retriever": "Busca de documentos",
    "extractor": "Leitura das evidências",
    "classifier": "Classificação da startup",
    "evidence_validator": "Verificação das evidências",
    "nvidia_rag": "Consulta à base NVIDIA",
    "recommendation": "Elaboração de recomendações",
    "briefing": "Preparação do briefing",
    "R1": "Triagem da busca",
    "R2": "Revisão das evidências",
    "R3": "Decisão da análise",
    "analisar": "Análise selecionada",
    "candidatas_prontas": "Startups encontradas",
    "relaxar": "Ampliação dos critérios da busca",
    "sem_resultado": "Nenhuma startup encontrada",
    "reextrair": "Nova leitura das evidências",
    "evidencia_pronta": "Evidências verificadas",
    "evidencia_insuficiente": "Evidência insuficiente",
    "nao_aderente": "Sem aderência nesta análise",
    "prosseguir": "Detalhamento autorizado",
    "concluida": "Análise concluída",
    "ausente": "Análise ainda não disponível",
    "setor": "Setor",
    "estagio": "Estágio",
    "localizacao": "Localização",
    "tamanho_time": "Tamanho da equipe",
    "classe_analisada": "Classificação",
    "dados_proprietarios": "Dados próprios",
    "workflow_profundo": "Integração aos processos do cliente",
    "distribuicao": "Distribuição do produto",
    "otimizacao_tecnica": "Otimização técnica",
    "stack_propria": "Tecnologia própria",
    "dependencia_api_externa": "Dependência de serviços externos de IA",
    "escala_e_dor_operacional": "Escala e dificuldades operacionais",
    "momento_e_financiamento": "Momento da empresa e financiamento",
    "equipe_e_contratacao": "Equipe e contratação",
    "outro": "Outras informações",
    "presenca": "Capacidade observada",
    "ausencia_explicita": "Ausência declarada pela fonte",
    "neutro": "Informação contextual",
    "confirmada": "Referência confirmada",
    "derrubada": "Referência não confirmada",
    "capacidade_confirmada": "Capacidade confirmada",
    "gap_confirmado": "Necessidade confirmada",
    "desconhecido": "Informação ainda não confirmada",
    "oportunidade_confirmada": "Oportunidade confirmada",
    "inferencia_llm": "Uso de modelos de linguagem",
    "treinamento_ou_finetuning": "Treinamento ou adaptação de modelos",
    "voz_fala_ou_transcricao": "Voz, fala ou transcrição",
    "dados_em_escala": "Processamento de grandes volumes de dados",
    "machine_learning_classico": "Aprendizado de máquina tradicional",
    "visao_computacional": "Visão computacional",
    "robotica_ou_simulacao": "Robótica ou simulação",
    "imagem_medica": "Análise de imagens médicas",
    "agentes_com_acoes_ou_controles": "Agentes de IA com ações ou controles",
    "ciberseguranca_em_escala": "Cibersegurança em grande escala",
    "convite_inception": "Convite ao NVIDIA Inception",
    "call_tecnica_descoberta": "Conversa técnica inicial",
    "benchmark_custo_latencia": "Comparação de custo e tempo de resposta",
    "poc_nim": "Teste de viabilidade com NVIDIA NIM",
    "workshop_guardrails": "Oficina sobre controles para IA",
    "intro_comunidade_evento": "Apresentação à comunidade ou a um evento",
    "alta": "Alta",
    "media": "Média",
    "baixa": "Baixa",
    "normal": "Regular",
    "centralidade_ia": "Importância da IA no produto",
    "gap_enderecavel": "Necessidade que pode ser atendida",
    "momento": "Momento da empresa",
    "alinhamento_setorial": "Afinidade do setor",
    "gate_evidencia": "Pontuação limitada pela evidência disponível",
    "teto_corrobacao": "Pontuação limitada pela confirmação entre fontes",
    "gate_non_ai": "Pontuação zero pela classificação sem IA no produto",
    "tecnologia": "Tecnologia NVIDIA",
    "conceitual": "Base conceitual",
    "site institucional": "site institucional",
    "blog": "blog",
    "notícia": "notícia",
    "vaga": "vaga",
    "perfil de founder": "perfil da pessoa fundadora",
    "release": "comunicado à imprensa",
    "AI-native": "AI-native",
    "AI-enabled": "AI-enabled",
    "non-AI": "non-AI",
}


def rotulo(valor: str) -> str:
    """Lê um valor fechado sem expor identificadores desconhecidos na tela."""
    return ROTULOS.get(valor, "Informação adicional")


# Texto livre permanece texto livre: apenas identificadores inequivocamente
# técnicos são substituídos. Palavras comuns como "normal" não são alteradas.
_TOKENS_EM_TEXTO = {
    token.casefold(): label
    for token, label in ROTULOS.items()
    if "_" in token
    or token in {"extractor", "classifier", "retriever", "recommendation", "R1", "R2", "R3"}
}
_PADRAO_TOKENS = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(token) for token in sorted(_TOKENS_EM_TEXTO, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)


def texto_legivel(texto: str) -> str:
    """Traduz tokens de mensagens; nunca usar em citações ou metadados de fonte.

    Não interpreta nem escapa Markdown. Quem renderiza continua responsável
    pelo escape; assim uma tradução não transforma texto em HTML ou link.
    """
    return _PADRAO_TOKENS.sub(
        lambda trecho: _TOKENS_EM_TEXTO[trecho.group().casefold()], texto
    )


def trajeto_legivel(valores: Iterable[str]) -> str:
    """Mostra as etapas executadas, preservando sua ordem e suas repetições."""
    return " → ".join(rotulo(valor) for valor in valores)


def rotular_fundamento(recomendacao) -> str:
    """Nomeia o fundamento sem transformar carga de trabalho em deficiência."""
    fundamento = ROTULO_FUNDAMENTO[recomendacao.tipo_fundamento]
    return f"{fundamento}: {rotulo(recomendacao.identificador_fundamento)}"
