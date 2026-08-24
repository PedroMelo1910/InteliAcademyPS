import sqlite3

import pytest

from radar.agentes.roteadores import rotear_r1
from radar.aplicacao import criar_aplicacao
from radar.configuracao import ErroConfiguracao
from radar.contratos import FiltrosEstruturados, PlanoConsulta


class ProvedorFixo:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0

    def invocar(self, mensagens):
        self.chamadas += 1
        return self.resposta


def plano_caju():
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor="Fintech / RH"),
        termos_busca=["benefícios", "cartão"],
        sinais_ia=[],
        foco_analise="plataforma de benefícios corporativos",
    )


def test_ranking_da_aplicacao_recebe_resultado_real_do_retriever(
    tmp_path, caminho_banco
):
    provedor = ProvedorFixo(plano_caju())
    checkpoints = tmp_path / "checkpoints.db"
    aplicacao = criar_aplicacao(provedor, caminho_banco, checkpoints)

    saida = aplicacao.executar_descoberta("fintech brasileira de benefícios com cartão")

    assert provedor.chamadas == 1
    assert saida.rota == "candidatas_prontas"
    assert [item.empresa.nome for item in saida.ranking] == ["Caju"]
    assert saida.ranking[0].documentos
    assert saida.trajeto == ("query_planner", "retriever")
    with sqlite3.connect(checkpoints) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


def test_chave_gemini_ausente_gera_erro_de_configuracao(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("radar.aplicacao.load_dotenv", lambda *_args, **_kwargs: False)

    with pytest.raises(ErroConfiguracao, match="GOOGLE_API_KEY"):
        criar_aplicacao(
            caminho_banco=tmp_path / "radar_sem_chave.db",
            caminho_checkpoints=tmp_path / "checkpoints_sem_chave.db",
        )


def test_caminho_selecionado_e_alcancavel_no_grafo(tmp_path, caminho_banco):
    provedor = ProvedorFixo(RuntimeError("não deve ser chamado"))
    aplicacao = criar_aplicacao(
        provedor, caminho_banco, tmp_path / "checkpoints_selecionada.db"
    )
    estado = aplicacao.grafo.invoke(
        {
            "consulta_usuario": "detalhar Caju",
            "startup_selecionada": 3,
            "plano_consulta": plano_caju(),
            "tentativas_relaxamento": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        },
        config={"configurable": {"thread_id": "selecionada"}},
    )
    assert rotear_r1(estado) == "analisar"
    assert provedor.chamadas == 0
