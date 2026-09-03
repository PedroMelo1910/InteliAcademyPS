"""Integração do NVIDIA RAG e do Recommendation no StateGraph.

Caminho exigido pelo marco:
Extractor → Classifier → Evidence Validator → R2 → R3 → NVIDIA RAG →
Recommendation → END. As duas saídas de bypass do R3 continuam terminando com
segurança, sem tocar a base de conhecimento NVIDIA. Tudo offline.
"""

from __future__ import annotations

import sqlite3

import pytest

from radar.agentes.rag_nvidia import ErroRagNvidia
from radar.agentes.recommendation import ErroRecommendation
from radar.contratos import (
    ContextoNvidia,
    FiltrosEstruturados,
    FitScore,
    PlanoConsulta,
    Recomendacao,
)
from radar.grafo import montar_grafo
from tests.conftest import (
    ConsultorNvidiaFalso,
    ProvedorSequencialFalso,
    contexto_nvidia_falso,
)


TRAJETO_ATE_R3 = (
    "query_planner",
    "retriever",
    "extractor",
    "classifier",
    "evidence_validator",
)
TRAJETO_COMPLETO = TRAJETO_ATE_R3 + ("nvidia_rag", "recommendation")


class ProvedorFixo:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
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
    caminho_banco,
    *,
    categoria="distribuicao",
    polaridade="ausencia_explicita",
    trecho=None,
):
    """Por padrão a dimensão ``distribuicao`` é um gap confirmado.

    O caminho aderente só produz recomendação para gap sustentado por evidência
    confirmada; uma afirmação de presença descreveria capacidade, não lacuna.
    A variante ``polaridade="presenca"`` existe para o caso non-AI, em que o
    Classifier exige evidência positiva sobre o produto.
    """
    linha = linha_caju(caminho_banco)
    textos = {
        "ausencia_explicita": (
            "A fonte registra ausência de canal de distribuição próprio."
        ),
        "presenca": "A fonte descreve o canal de distribuição do produto.",
    }
    return {
        "id_startup": linha["id_startup"],
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": textos.get(
                    polaridade, "A fonte traz informações sobre o produto."
                ),
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


def rascunho_valido(id_chunk=101):
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
        "ids_afirmacoes": [1],
        "ids_chunks": [id_chunk],
    }


def lote(*rascunhos):
    return {"rascunhos": list(rascunhos)}


def contexto_antigo(consulta: str):
    """Contexto de outra análise na mesma forma JSON que o grafo grava."""
    return contexto_nvidia_falso(consulta).model_dump(mode="json")


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


def montar(caminho_banco, tmp_path, *, perfil, classe="AI-enabled", consultor=None,
           recomendacao=None, nome="checkpoints.db"):
    from radar.base_startups import BaseStartups

    consultor = consultor if consultor is not None else ConsultorNvidiaFalso()
    recomendacao = (
        recomendacao
        if recomendacao is not None
        else ProvedorSequencialFalso(lote(rascunho_valido()))
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        ProvedorFixo(RuntimeError("o Query Planner não deve ser chamado")),
        ProvedorFixo(perfil),
        ProvedorFixo(classificacao(classe)),
        tmp_path / nome,
        consultor,
        recomendacao,
    )
    return grafo, conexao, consultor, recomendacao


# ----------------------------------------------------------------------
# Caminho normal
# ----------------------------------------------------------------------


def test_prosseguir_percorre_rag_e_recommendation_ate_o_fim(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, consultor, recomendacao = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "prosseguir"}},
        )
    finally:
        conexao.close()

    assert tuple(saida["trajeto"]) == TRAJETO_COMPLETO
    assert consultor.chamadas == 1
    assert recomendacao.chamadas == 1
    assert ContextoNvidia.model_validate(saida["contexto_nvidia"]).trechos
    assert isinstance(saida["fit_score"], FitScore)
    assert len(saida["recomendacoes"]) == 1
    assert Recomendacao.model_validate(saida["recomendacoes"][0])


def test_recomendacao_do_caminho_normal_tem_proveniencia_dos_dois_lados(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec = montar(caminho_banco, tmp_path, perfil=perfil)
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "proveniencia"}},
        )
    finally:
        conexao.close()

    recomendacao = Recomendacao.model_validate(saida["recomendacoes"][0])
    assert recomendacao.evidencias_startup[0].id_afirmacao == 1
    assert recomendacao.evidencias_startup[0].id_documento == perfil["afirmacoes"][0][
        "id_documento"
    ]
    assert str(recomendacao.evidencias_startup[0].url_fonte).startswith("http")
    assert any(
        citacao.origem == "tecnologia" for citacao in recomendacao.citacoes_nvidia
    )


def test_consulta_nvidia_nasce_do_perfil_validado(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, consultor, _rec = montar(caminho_banco, tmp_path, perfil=perfil)
    try:
        grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "consulta"}},
        )
    finally:
        conexao.close()

    assert len(consultor.consultas) == 1
    assert "Fintech / RH" in consultor.consultas[0]
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        assert classe not in consultor.consultas[0]


# ----------------------------------------------------------------------
# Os dois bypasses do R3 continuam terminando com segurança
# ----------------------------------------------------------------------


def test_nao_aderente_termina_sem_recuperacao_nvidia(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco, polaridade="presenca")
    grafo, conexao, consultor, recomendacao = montar(
        caminho_banco, tmp_path, perfil=perfil, classe="non-AI"
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "nao-aderente"}},
        )
    finally:
        conexao.close()

    assert tuple(saida["trajeto"]) == TRAJETO_ATE_R3
    assert consultor.chamadas == 0
    assert recomendacao.chamadas == 0
    assert saida["contexto_nvidia"] is None
    assert saida["recomendacoes"] is None
    assert saida["fit_score"] is None


def test_evidencia_insuficiente_termina_sem_recuperacao_nvidia(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(
        caminho_banco,
        trecho="Este trecho não ocorre literalmente em nenhum documento curado.",
    )
    grafo, conexao, consultor, recomendacao = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "insuficiente"}},
        )
    finally:
        conexao.close()

    assert saida["trajeto"][-1] == "evidence_validator"
    assert "nvidia_rag" not in saida["trajeto"]
    assert consultor.chamadas == 0
    assert recomendacao.chamadas == 0
    assert saida["contexto_nvidia"] is None


# ----------------------------------------------------------------------
# Invalidação de estado velho e isolamento entre execuções
# ----------------------------------------------------------------------


def test_estado_velho_de_recomendacao_e_substituido_na_reexecucao(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _rec = montar(caminho_banco, tmp_path, perfil=perfil)
    try:
        saida = grafo.invoke(
            estado_selecionado(
                perfil["id_startup"],
                contexto_nvidia=contexto_antigo("consulta de outra análise"),
                recomendacoes=[],
                fit_score=None,
            ),
            config={"configurable": {"thread_id": "estado-velho"}},
        )
    finally:
        conexao.close()

    contexto = ContextoNvidia.model_validate(saida["contexto_nvidia"])
    assert contexto.consulta_gerada != "consulta de outra análise"
    assert len(saida["recomendacoes"]) == 1


def test_falha_do_rag_nao_deixa_contexto_nvidia_velho_no_checkpoint(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    consultor = ConsultorNvidiaFalso(erro=RuntimeError("índice NVIDIA indisponível"))
    grafo, conexao, _consultor, recomendacao = montar(
        caminho_banco, tmp_path, perfil=perfil, consultor=consultor
    )
    config = {"configurable": {"thread_id": "rag-falho"}}
    try:
        with pytest.raises(ErroRagNvidia):
            grafo.invoke(
                estado_selecionado(
                    perfil["id_startup"],
                    contexto_nvidia=contexto_antigo("contexto antigo"),
                ),
                config=config,
            )
        estado = grafo.get_state(config).values
    finally:
        conexao.close()

    assert estado.get("contexto_nvidia") is None
    assert estado.get("recomendacoes") is None
    assert estado.get("fit_score") is None
    assert recomendacao.chamadas == 0


def test_falha_da_recomendacao_nao_deixa_recomendacao_velha_no_checkpoint(
    tmp_path, caminho_banco
):
    perfil = perfil_caju(caminho_banco)
    provedor = ProvedorSequencialFalso(
        lote(rascunho_valido(id_chunk=999)), lote(rascunho_valido(id_chunk=999))
    )
    grafo, conexao, _consultor, _rec = montar(
        caminho_banco, tmp_path, perfil=perfil, recomendacao=provedor
    )
    config = {"configurable": {"thread_id": "recomendacao-falha"}}
    try:
        with pytest.raises(ErroRecommendation):
            grafo.invoke(
                estado_selecionado(perfil["id_startup"], recomendacoes=[]),
                config=config,
            )
        estado = grafo.get_state(config).values
    finally:
        conexao.close()

    assert estado.get("recomendacoes") is None
    assert estado.get("fit_score") is None
    assert provedor.chamadas == 2


def test_threads_distintos_nao_compartilham_contexto_nem_recomendacao(
    tmp_path, caminho_banco
):
    """Isolamento de checkpoint: cada thread recupera e recomenda do zero."""
    perfil = perfil_caju(caminho_banco)
    consultor = ConsultorNvidiaFalso()
    provedor = ProvedorSequencialFalso(
        lote(rascunho_valido()), lote(rascunho_valido())
    )
    grafo, conexao, _consultor, _rec = montar(
        caminho_banco,
        tmp_path,
        perfil=perfil,
        consultor=consultor,
        recomendacao=provedor,
    )
    try:
        primeira = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "thread-a"}},
        )
        segunda = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "thread-b"}},
        )
    finally:
        conexao.close()

    assert consultor.chamadas == 2
    assert provedor.chamadas == 2
    assert tuple(primeira["trajeto"]) == TRAJETO_COMPLETO
    assert tuple(segunda["trajeto"]) == TRAJETO_COMPLETO


def test_descarte_parcial_registra_o_erro_exato_no_estado(tmp_path, caminho_banco):
    perfil = perfil_caju(caminho_banco)
    # Gap distinto do rascunho válido de propósito: assim o descarte que este
    # teste prova continua sendo o da tecnologia fora das candidatas do gap, e
    # não o da regra de gap duplicado (§6.1), que agiria antes.
    invalido = rascunho_valido()
    invalido["gap_enderecado"] = "otimizacao_tecnica"
    invalido["tecnologias"] = ["NVIDIA Riva"]
    provedor = ProvedorSequencialFalso(
        lote(rascunho_valido(), invalido), lote(rascunho_valido(), invalido)
    )
    grafo, conexao, _consultor, _rec = montar(
        caminho_banco, tmp_path, perfil=perfil, recomendacao=provedor
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(perfil["id_startup"]),
            config={"configurable": {"thread_id": "descarte"}},
        )
    finally:
        conexao.close()

    assert len(saida["recomendacoes"]) == 1
    assert any("NVIDIA Riva" in erro for erro in saida["erros"])


# ----------------------------------------------------------------------
# Forma serializada do estado: o que EstadoRadar anota é o que trafega
# ----------------------------------------------------------------------


def test_estado_do_grafo_e_do_checkpoint_tem_a_forma_declarada_em_estado_radar(
    tmp_path, caminho_banco
):
    """``contexto_nvidia`` e ``recomendacoes`` trafegam como dicionário JSON.

    ``AnyHttpUrl`` não atravessa o msgpack do checkpointer, então o grafo grava
    a forma JSON do mesmo contrato. ``FitScore`` não tem URL e segue instância.
    A anotação de ``EstadoRadar`` precisa dizer exatamente isso.
    """
    perfil = perfil_caju(caminho_banco)
    grafo, conexao, _consultor, _recomendacao = montar(
        caminho_banco, tmp_path, perfil=perfil
    )
    config = {"configurable": {"thread_id": "forma-do-estado"}}
    try:
        saida = grafo.invoke(estado_selecionado(perfil["id_startup"]), config=config)
        recuperado = grafo.get_state(config).values
    finally:
        conexao.close()

    for estado in (saida, recuperado):
        assert isinstance(estado["contexto_nvidia"], dict)
        assert isinstance(estado["recomendacoes"], list)
        assert estado["recomendacoes"]
        assert all(isinstance(item, dict) for item in estado["recomendacoes"])
        assert isinstance(estado["fit_score"], FitScore)

        # A forma serializada é reidratável pelos contratos originais.
        contexto = ContextoNvidia.model_validate(estado["contexto_nvidia"])
        assert len(contexto.trechos) >= 5
        recomendacoes = [
            Recomendacao.model_validate(item) for item in estado["recomendacoes"]
        ]
        assert all(item.evidencias_startup for item in recomendacoes)
        assert all(item.citacoes_nvidia for item in recomendacoes)


def test_checkpoint_nunca_traz_gap_repetido_no_relatorio_final(
    tmp_path, caminho_banco
):
    """§6.1 no estado persistido: nem a saída nem o checkpoint repetem gap."""
    perfil = perfil_caju(caminho_banco)
    # As duas tentativas insistem no mesmo gap; o retry não corrige o lote.
    provedor = ProvedorSequencialFalso(
        lote(rascunho_valido(), rascunho_valido(102), rascunho_valido(103)),
        lote(rascunho_valido(), rascunho_valido(102)),
    )
    grafo, conexao, _consultor, _rec = montar(
        caminho_banco, tmp_path, perfil=perfil, recomendacao=provedor
    )
    config = {"configurable": {"thread_id": "gap-repetido"}}
    try:
        saida = grafo.invoke(estado_selecionado(perfil["id_startup"]), config=config)
        recuperado = grafo.get_state(config).values
    finally:
        conexao.close()

    assert provedor.chamadas == 2
    for estado in (saida, recuperado):
        gaps = [
            Recomendacao.model_validate(item).gap_enderecado
            for item in estado["recomendacoes"]
        ]
        assert gaps == ["distribuicao"]
        assert len(gaps) == len(set(gaps))
    assert any("uma única vez" in erro for erro in recuperado["erros"])
