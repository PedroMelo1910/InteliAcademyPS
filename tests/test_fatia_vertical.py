import json
import sqlite3

import pytest

from radar.agentes.extractor import ErroExtractor
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


def plano_sem_resultado():
    return PlanoConsulta(
        filtros=FiltrosEstruturados(),
        termos_busca=["termo-inexistente-para-provar-rota"],
        sinais_ia=[],
        foco_analise="consulta sem correspondência na base",
    )


def perfil_caju(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        linha = conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.conteudo_texto
            FROM startups s
            JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = 'Caju'
            ORDER BY d.id
            LIMIT 1
            """
        ).fetchone()
    return {
        "id_startup": linha["id_startup"],
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A fonte apresenta informações sobre o produto da Caju.",
                "categoria": "outro",
                "polaridade": "neutro",
                "id_documento": linha["id_documento"],
                "trecho_citado": linha["conteudo_texto"][:200],
            }
        ],
    }


def test_ranking_da_aplicacao_recebe_resultado_real_do_retriever(
    tmp_path, caminho_banco
):
    provedor = ProvedorFixo(plano_caju())
    provedor_extracao = ProvedorFixo(RuntimeError("não deve ser chamado"))
    checkpoints = tmp_path / "checkpoints.db"
    aplicacao = criar_aplicacao(
        provedor, caminho_banco, checkpoints, provedor_extracao
    )

    saida = aplicacao.executar_descoberta("fintech brasileira de benefícios com cartão")

    assert provedor.chamadas == 1
    assert provedor_extracao.chamadas == 0
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
    provedor = ProvedorFixo(RuntimeError("o Query Planner não deve ser chamado"))
    provedor_extracao = ProvedorFixo(perfil_caju(caminho_banco))
    aplicacao = criar_aplicacao(
        provedor,
        caminho_banco,
        tmp_path / "checkpoints_selecionada.db",
        provedor_extracao,
    )
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    estado = aplicacao.grafo.invoke(
        {
            "consulta_usuario": "detalhar Caju",
            "startup_selecionada": id_caju,
            "plano_consulta": plano_caju(),
            "tentativas_relaxamento": 0,
            "tentativas_extracao": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        },
        config={"configurable": {"thread_id": "selecionada"}},
    )
    assert rotear_r1(estado) == "analisar"
    assert provedor.chamadas == 0
    assert provedor_extracao.chamadas == 1
    assert estado["perfil_extraido"].id_startup == id_caju
    assert estado["tentativas_extracao"] == 1
    assert estado["trajeto"] == ["query_planner", "retriever", "extractor"]
    assert "conteudo_texto" not in json.dumps(
        estado["perfil_extraido"].model_dump(), ensure_ascii=False
    )


def test_falha_do_extractor_interrompe_o_caminho_sem_fabricar_perfil(
    tmp_path, caminho_banco
):
    provedor_plano = ProvedorFixo(RuntimeError("não deve ser chamado"))
    provedor_extracao = ProvedorFixo({"afirmacoes": []})
    aplicacao = criar_aplicacao(
        provedor_plano,
        caminho_banco,
        tmp_path / "checkpoints_extractor_invalido.db",
        provedor_extracao,
    )
    id_caju = perfil_caju(caminho_banco)["id_startup"]

    with pytest.raises(ErroExtractor, match="duas vezes fora do contrato"):
        aplicacao.grafo.invoke(
            {
                "consulta_usuario": "detalhar Caju",
                "startup_selecionada": id_caju,
                "plano_consulta": plano_caju(),
                "tentativas_relaxamento": 0,
                "tentativas_extracao": 0,
                "criterios_relaxados": [],
                "erros": [],
                "trajeto": [],
            },
            config={"configurable": {"thread_id": "extractor-invalido"}},
        )

    assert provedor_plano.chamadas == 0
    assert provedor_extracao.chamadas == 2


def test_injecao_offline_exige_os_dois_provedores(tmp_path, caminho_banco):
    with pytest.raises(ErroConfiguracao, match="informe juntos"):
        criar_aplicacao(
            ProvedorFixo(plano_caju()),
            caminho_banco,
            tmp_path / "checkpoints_incompletos.db",
        )


def test_grafo_preserva_relaxamento_e_termino_sem_resultado(
    tmp_path, caminho_banco
):
    provedor_plano = ProvedorFixo(plano_sem_resultado())
    provedor_extracao = ProvedorFixo(RuntimeError("não deve ser chamado"))
    aplicacao = criar_aplicacao(
        provedor_plano,
        caminho_banco,
        tmp_path / "checkpoints_sem_resultado.db",
        provedor_extracao,
    )

    estado = aplicacao.grafo.invoke(
        {
            "consulta_usuario": "consulta deliberadamente sem resultado",
            "startup_selecionada": None,
            "tentativas_relaxamento": 0,
            "tentativas_extracao": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        },
        config={"configurable": {"thread_id": "sem-resultado"}},
    )

    assert rotear_r1(estado) == "sem_resultado"
    assert estado["tentativas_relaxamento"] == 2
    assert estado["resultado_recuperacao"].empresas == []
    assert provedor_plano.chamadas == 1
    assert provedor_extracao.chamadas == 0
    assert estado["trajeto"] == [
        "query_planner",
        "retriever",
        "query_planner",
        "retriever",
        "query_planner",
        "retriever",
    ]
