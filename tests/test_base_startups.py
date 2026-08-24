import sqlite3

from radar.base_startups import (
    BaseStartups,
    inicializar_banco,
    montar_consulta_descoberta,
)
from radar.configuracao import CAMINHO_DADOS_CURADOS
from radar.contratos import FiltrosEstruturados, PlanoConsulta


def plano(termos: list[str], **filtros) -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(**filtros),
        termos_busca=termos,
        sinais_ia=[],
        foco_analise="teste offline",
    )


def test_inicializacao_e_repetivel_e_preserva_contagens(tmp_path):
    banco = tmp_path / "repetivel.db"
    inicializar_banco(banco, CAMINHO_DADOS_CURADOS)
    inicializar_banco(banco, CAMINHO_DADOS_CURADOS)
    with sqlite3.connect(banco) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM startups").fetchone()[0] == 3
        assert conexao.execute("SELECT COUNT(*) FROM documentos").fetchone()[0] == 9
        dominios = conexao.execute(
            """
            SELECT startup_id, COUNT(DISTINCT dominio_fonte)
            FROM documentos GROUP BY startup_id ORDER BY startup_id
            """
        ).fetchall()
    assert [quantidade for _, quantidade in dominios] == [3, 3, 3]


def test_filtro_estruturado_restringe_as_empresas(base: BaseStartups):
    resultado = base.recuperar(plano(["tecnologia"], setor="Saúde"))
    assert [empresa.nome for empresa in resultado.empresas] == ["Alice"]
    assert all(documento.id_startup == resultado.empresas[0].id_startup for documento in resultado.documentos)


def test_consulta_do_usuario_permanece_em_parametros(caminho_banco):
    entrada = "fintech' OR 1=1 --"
    consulta = montar_consulta_descoberta(plano([entrada]))
    assert entrada not in consulta.sql
    assert entrada in consulta.parametros[0]
    # Além da inspeção da fronteira, a entrada hostil executa sem alterar a base.
    resultado = BaseStartups(caminho_banco).recuperar(plano([entrada]))
    assert resultado.empresas == []
    with sqlite3.connect(caminho_banco) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM startups").fetchone()[0] == 3


def test_fts5_aceita_termo_com_hifen(base: BaseStartups):
    resultado = base.recuperar(plano(["Sabiá-4"]))
    assert resultado.empresas
    assert resultado.empresas[0].nome == "Maritaca AI"


def test_bm25_usa_ordem_crescente_do_sqlite(base: BaseStartups):
    resultado = base.recuperar(plano(["modelos", "linguagem", "português"]))
    scores = [documento.score_bm25 for documento in resultado.documentos]
    assert len(scores) >= 2
    assert scores == sorted(scores)


def test_retriever_funciona_offline_com_plano_fixo(base: BaseStartups):
    resultado = base.recuperar(
        plano(["benefícios", "cartão"], setor="Fintech / RH")
    )
    assert [empresa.nome for empresa in resultado.empresas] == ["Caju"]
    assert {documento.dominio_fonte for documento in resultado.documentos} <= {
        "caju.com.br",
        "bloomberglinea.com.br",
        "onevc.vc",
    }
