"""Integração do Briefing no StateGraph e a segunda invocação da aplicação.

As três rotas do R3 terminam num ``Briefing`` válido:
  R3 evidencia_insuficiente → Briefing → END
  R3 nao_aderente           → Briefing → END
  R3 prosseguir             → NVIDIA RAG → Recommendation → Briefing → END

A jornada completa (descoberta e depois aprofundamento da escolhida) roda
offline, com provedores injetados. Nenhum teste toca a rede.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from radar.agentes.briefing import ErroBriefing
from radar.agentes.extractor import Extractor
from radar.agentes.recommendation import ErroRecommendation
from radar.agentes.retriever import Retriever
from radar.aplicacao import ErroAplicacao, criar_aplicacao
from radar.base_startups import BaseStartups
from radar.contratos import (
    Briefing,
    FiltrosEstruturados,
    PlanoConsulta,
)
from radar.grafo import montar_grafo
from tests.conftest import (
    ConsultorNvidiaFalso,
    ProvedorSequencialFalso,
)


DATA_FIXA = date(2026, 9, 3)

TRAJETO_ATE_R3 = (
    "query_planner",
    "retriever",
    "extractor",
    "classifier",
    "evidence_validator",
)
TRAJETO_NORMAL = TRAJETO_ATE_R3 + ("nvidia_rag", "recommendation", "briefing")
TRAJETO_BYPASS = TRAJETO_ATE_R3 + ("briefing",)


class ProvedorFixo:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        if isinstance(self.resposta, Exception):
            raise self.resposta
        return self.resposta


def plano_caju():
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor="Fintech / RH"),
        termos_busca=["benefícios", "cartão"],
        sinais_ia=[],
        foco_analise="plataforma de benefícios corporativos",
    )


def linha_caju(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        return conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.conteudo_texto
            FROM startups s
            JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = 'Caju'
            ORDER BY d.id
            LIMIT 1
            """
        ).fetchone()


def perfil_caju(
    caminho_banco, *, categoria="distribuicao", polaridade="ausencia_explicita",
    trecho=None,
):
    linha = linha_caju(caminho_banco)
    return {
        "id_startup": linha["id_startup"],
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A fonte descreve o canal de distribuição do produto.",
                "categoria": categoria,
                "polaridade": polaridade,
                "id_documento": linha["id_documento"],
                "trecho_citado": trecho or linha["conteudo_texto"][:200],
            }
        ],
    }


def classificacao(classe="AI-enabled"):
    return {
        "classe": classe,
        "justificativa": (
            "A plataforma de benefícios corporativos é o produto contratado. "
            "O perfil descreve o canal pelo qual ela chega ao cliente."
        ),
        "ids_afirmacoes_suporte": [1],
    }


def rascunho_recomendacao(ids_afirmacoes=(1,)):
    return {
        "gap_enderecado": "distribuicao",
        "tecnologias": ["NVIDIA Inception"],
        "justificativa_tecnica": (
            "O programa abre acesso a suporte técnico e créditos de computação."
        ),
        "justificativa_negocio": (
            "A validação junto ao ecossistema encurta o ciclo de venda enterprise."
        ),
        "proxima_acao": {
            "tipo_acao": "convite_inception",
            "detalhe": "Enviar o convite de admissão ao programa nesta semana.",
        },
        "ids_afirmacoes": list(ids_afirmacoes),
        "ids_chunks": [101],
    }


def lote(*rascunhos):
    return {"rascunhos": list(rascunhos)}


def rascunho_briefing():
    return {
        "tese": {
            "texto": "A empresa é AI-enabled com lacuna de distribuição confirmada.",
            "ids_afirmacoes_suporte": [1],
        },
        "sintese_executiva": {
            "texto": "A plataforma atende empresas e depende de canal de terceiros.",
            "ids_afirmacoes_suporte": [1],
        },
        "pontos_de_conversa": [
            {"texto": "Perguntar como o canal atual é remunerado.", "ids_afirmacoes_suporte": [1]},
            {"texto": "Explorar o plano de expansão para 2027.", "ids_afirmacoes_suporte": [1]},
        ],
    }


def estado_selecionado(id_startup, **ajustes):
    base = {
        "consulta_usuario": "detalhar Caju",
        "startup_selecionada": id_startup,
        "plano_consulta": plano_caju(),
        "tentativas_relaxamento": 0,
        "tentativas_extracao": 0,
        "criterios_relaxados": [],
        "erros": [],
        "trajeto": [],
    }
    base.update(ajustes)
    return base


def montar(
    caminho_banco,
    tmp_path,
    *,
    perfil,
    classe="AI-enabled",
    consultor=None,
    recomendacao=None,
    briefing=None,
    nome="checkpoints.db",
):
    consultor = consultor if consultor is not None else ConsultorNvidiaFalso()
    recomendacao = (
        recomendacao
        if recomendacao is not None
        else ProvedorSequencialFalso(lote(rascunho_recomendacao()))
    )
    briefing = (
        briefing
        if briefing is not None
        else ProvedorSequencialFalso(rascunho_briefing())
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        ProvedorFixo(RuntimeError("o Query Planner não deve ser chamado")),
        ProvedorFixo(perfil),
        ProvedorFixo(classificacao(classe)),
        tmp_path / nome,
        consultor,
        recomendacao,
        briefing,
        relogio=lambda: DATA_FIXA,
    )
    return grafo, conexao, consultor, recomendacao, briefing


# ----------------------------------------------------------------------
# As três rotas terminam num Briefing válido
# ----------------------------------------------------------------------


def test_rota_prosseguir_termina_em_briefing_normal(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, consultor, recomendacao, briefing = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "normal"}},
        )
    finally:
        conexao.close()

    assert tuple(saida["trajeto"]) == TRAJETO_NORMAL
    final = Briefing.model_validate(saida["briefing"])
    assert final.variante == "normal"
    assert final.veredito.classe == "AI-enabled"
    assert final.recomendacoes
    assert final.rodape.rota_r3 == "prosseguir"
    assert consultor.chamadas == 1
    assert recomendacao.chamadas == 1
    assert briefing.chamadas == 1


def test_rota_nao_aderente_termina_em_briefing_sem_rag_nem_recomendacao(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco, polaridade="presenca")
    grafo, conexao, consultor, recomendacao, briefing = montar(
        caminho_banco, tmp_path, perfil=perfil, classe="non-AI"
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "non-ai"}},
        )
    finally:
        conexao.close()

    assert tuple(saida["trajeto"]) == TRAJETO_BYPASS
    assert consultor.chamadas == 0
    assert recomendacao.chamadas == 0
    assert briefing.chamadas == 0
    final = Briefing.model_validate(saida["briefing"])
    assert final.variante == "nao_aderente"
    assert final.veredito.fit_score_total == 0
    assert final.recomendacoes == []
    assert saida["fit_score"] is None


def test_rota_evidencia_insuficiente_termina_em_briefing_sem_llm(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(
        caminho_banco,
        trecho="Este trecho não ocorre literalmente em nenhum documento curado.",
    )
    grafo, conexao, consultor, recomendacao, briefing = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "insuficiente"}},
        )
    finally:
        conexao.close()

    assert saida["trajeto"][-1] == "briefing"
    assert "nvidia_rag" not in saida["trajeto"]
    assert consultor.chamadas == 0
    assert recomendacao.chamadas == 0
    assert briefing.chamadas == 0
    final = Briefing.model_validate(saida["briefing"])
    assert final.variante == "evidencia_insuficiente"
    assert final.veredito.classe is None
    assert final.veredito.fit_score_total is None
    assert final.avisos


# ----------------------------------------------------------------------
# §11.3: o Recommendation deixa de falhar alto onde não há o que recomendar
# ----------------------------------------------------------------------


def test_sem_gap_sustentado_chega_ao_briefing_sem_chamar_o_provedor_de_recomendacao(
    tmp_path, caminho_banco
):
    # Capacidade confirmada, não gap: não há o que recomendar, e chamar o LLM
    # só produziria rascunhos que seriam todos descartados.
    perfil = perfil_caju(caminho_banco, polaridade="presenca")
    grafo, conexao, consultor, recomendacao, briefing = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "sem-gap"}},
        )
    finally:
        conexao.close()

    assert recomendacao.chamadas == 0
    assert briefing.chamadas == 0
    assert consultor.chamadas == 1
    final = Briefing.model_validate(saida["briefing"])
    assert final.variante == "evidencia_insuficiente"
    assert final.recomendacoes == []
    assert saida["recomendacoes"] == []
    assert saida["fit_score"] is None
    assert any("gap" in erro for erro in saida["erros"])


def test_rascunhos_invalidos_nao_viram_recomendacao_fabricada(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    # As duas tentativas citam um id que não existe no perfil validado.
    provedor = ProvedorSequencialFalso(
        lote(rascunho_recomendacao(ids_afirmacoes=(999,))),
        lote(rascunho_recomendacao(ids_afirmacoes=(999,))),
    )
    grafo, conexao, _consultor, recomendacao, briefing = montar(
        caminho_banco, tmp_path, perfil=perfil, recomendacao=provedor
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "descartadas"}},
        )
    finally:
        conexao.close()

    assert recomendacao.chamadas == 2
    assert briefing.chamadas == 0
    final = Briefing.model_validate(saida["briefing"])
    assert final.variante == "evidencia_insuficiente"
    assert final.recomendacoes == []
    assert saida["recomendacoes"] == []
    assert any("999" in erro for erro in saida["erros"])


def test_queda_do_provedor_de_recomendacao_continua_sendo_falha(
    tmp_path, caminho_banco
):
    """Indisponibilidade operacional não vira 'evidência insuficiente'."""
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec, briefing = montar(
        caminho_banco,
        tmp_path,
        perfil=perfil,
        recomendacao=ProvedorFixo(RuntimeError("provedor fora do ar")),
    )
    config = {"configurable": {"thread_id": "queda-recomendacao"}}
    try:
        with pytest.raises(ErroRecommendation, match="não respondeu"):
            grafo.invoke(estado_selecionado(perfil["id_startup"]), config=config)
        valores = grafo.get_state(config).values
    finally:
        conexao.close()

    assert briefing.chamadas == 0
    assert valores.get("briefing") is None


def test_queda_do_provedor_do_briefing_nao_expoe_briefing_parcial(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec, _briefing = montar(
        caminho_banco,
        tmp_path,
        perfil=perfil,
        briefing=ProvedorFixo(RuntimeError("provedor fora do ar")),
    )
    config = {"configurable": {"thread_id": "queda-briefing"}}
    try:
        with pytest.raises(ErroBriefing, match="não respondeu"):
            grafo.invoke(estado_selecionado(perfil["id_startup"]), config=config)
        valores = grafo.get_state(config).values
    finally:
        conexao.close()

    assert valores.get("briefing") is None


# ----------------------------------------------------------------------
# Invalidação de briefing velho e checkpoint
# ----------------------------------------------------------------------


def test_briefing_velho_e_invalidado_por_reexecucao_a_montante(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec, _briefing = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(
                perfil["id_startup"],
                briefing={"variante": "de outra análise"},
            ),
            config={"configurable": {"thread_id": "briefing-velho"}},
        )
    finally:
        conexao.close()

    final = Briefing.model_validate(saida["briefing"])
    assert final.cabecalho.nome == "Caju"
    assert final.variante == "normal"


@pytest.mark.parametrize("no", ["retriever", "extractor"])
def test_nos_a_montante_zeram_o_briefing_do_estado(caminho_banco, no):
    """A invalidação é explícita no nó, não um efeito colateral do grafo."""
    base = BaseStartups(caminho_banco)
    perfil = perfil_caju(caminho_banco)
    estado = estado_selecionado(perfil["id_startup"], briefing={"variante": "velho"})
    if no == "retriever":
        saida = Retriever(base)(estado)
    else:
        estado["resultado_recuperacao"] = Retriever(base)(estado)[
            "resultado_recuperacao"
        ]
        saida = Extractor(base, ProvedorFixo(perfil))(estado)

    assert saida["briefing"] is None


def test_briefing_final_sobrevive_a_serializacao_do_checkpoint(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec, _briefing = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    config = {"configurable": {"thread_id": "checkpoint-briefing"}}
    try:
        saida = grafo.invoke(estado_selecionado(perfil["id_startup"]), config=config)
        recuperado = grafo.get_state(config).values
    finally:
        conexao.close()

    for estado in (saida, recuperado):
        assert isinstance(estado["briefing"], dict)
        final = Briefing.model_validate(estado["briefing"])
        assert final.variante == "normal"
        assert final.recomendacoes[0].evidencias_startup
        assert final.rodape.trajeto[-1] == "briefing"
    with sqlite3.connect(tmp_path / "checkpoints.db") as banco:
        assert banco.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


# ----------------------------------------------------------------------
# Segunda invocação: descoberta → aprofundamento da escolhida
# ----------------------------------------------------------------------


def aplicacao_completa(caminho_banco, tmp_path, *, perfil, classe="AI-enabled"):
    provedor_plano = ProvedorFixo(plano_caju())
    aplicacao = criar_aplicacao(
        provedor_plano,
        caminho_banco,
        tmp_path / "checkpoints_jornada.db",
        ProvedorFixo(perfil),
        ProvedorFixo(classificacao(classe)),
        ConsultorNvidiaFalso(),
        ProvedorSequencialFalso(lote(rascunho_recomendacao())),
        ProvedorSequencialFalso(rascunho_briefing()),
        relogio=lambda: DATA_FIXA,
    )
    return aplicacao, provedor_plano


def test_segunda_invocacao_reaproveita_o_plano_sem_novo_query_planner(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    aplicacao, provedor_plano = aplicacao_completa(
        caminho_banco, tmp_path, perfil=perfil
    )

    descoberta = aplicacao.executar_descoberta(
        "fintech brasileira de benefícios com cartão"
    )
    assert provedor_plano.chamadas == 1

    aprofundamento = aplicacao.executar_aprofundamento(
        descoberta, descoberta.ranking[0].empresa.id_startup
    )

    assert provedor_plano.chamadas == 1
    assert aprofundamento.briefing.variante == "normal"
    assert aprofundamento.plano == descoberta.plano


def test_segunda_invocacao_exige_startup_vinda_da_descoberta(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco)
    aplicacao, _provedor = aplicacao_completa(caminho_banco, tmp_path, perfil=perfil)
    descoberta = aplicacao.executar_descoberta(
        "fintech brasileira de benefícios com cartão"
    )
    intrusa = max(item.empresa.id_startup for item in descoberta.ranking) + 500

    with pytest.raises(ErroAplicacao, match="não está entre as candidatas"):
        aplicacao.executar_aprofundamento(descoberta, intrusa)


def test_jornada_offline_completa_descobre_e_depois_aprofunda(
    tmp_path, caminho_banco
):
    """A jornada real de backend: consulta, ranking, clique e briefing."""
    perfil = perfil_caju(caminho_banco)
    aplicacao, provedor_plano = aplicacao_completa(
        caminho_banco, tmp_path, perfil=perfil
    )

    descoberta = aplicacao.executar_descoberta(
        "fintech brasileira de benefícios com cartão"
    )

    assert descoberta.rota == "candidatas_prontas"
    assert [item.empresa.nome for item in descoberta.ranking] == ["Caju"]
    assert descoberta.trajeto == ("query_planner", "retriever")
    assert descoberta.consulta == "fintech brasileira de benefícios com cartão"

    escolhida = descoberta.ranking[0].empresa.id_startup
    aprofundamento = aplicacao.executar_aprofundamento(descoberta, escolhida)

    briefing = aprofundamento.briefing
    assert briefing.variante == "normal"
    assert briefing.cabecalho.nome == "Caju"
    assert briefing.cabecalho.consulta_original == descoberta.consulta
    assert briefing.cabecalho.data_geracao == DATA_FIXA
    assert briefing.veredito.ids_afirmacoes_suporte
    assert briefing.sintese_executiva.ids_afirmacoes_suporte
    assert all(p.ids_afirmacoes_suporte for p in briefing.pontos_de_conversa)
    assert briefing.recomendacoes[0].citacoes_nvidia
    assert briefing.fontes
    assert briefing.rodape.versao_rubrica == "rubrica-v1"
    assert aprofundamento.trajeto == TRAJETO_NORMAL
    assert provedor_plano.chamadas == 1


def test_jornada_non_ai_devolve_briefing_de_nao_aderencia(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco, polaridade="presenca")
    aplicacao, _provedor = aplicacao_completa(
        caminho_banco, tmp_path, perfil=perfil, classe="non-AI"
    )
    descoberta = aplicacao.executar_descoberta(
        "fintech brasileira de benefícios com cartão"
    )

    aprofundamento = aplicacao.executar_aprofundamento(
        descoberta, descoberta.ranking[0].empresa.id_startup
    )

    assert aprofundamento.briefing.variante == "nao_aderente"
    assert aprofundamento.briefing.veredito.fit_score_total == 0
    assert aprofundamento.trajeto == TRAJETO_BYPASS
