import sqlite3

import pytest

from radar.base_startups import (
    BaseStartups,
    ErroDocumentosStartup,
    carregar_curadoria,
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
    curadoria = carregar_curadoria(CAMINHO_DADOS_CURADOS)
    startups_esperadas = len(curadoria)
    documentos_esperados = sum(len(startup.documentos) for startup in curadoria)

    banco = tmp_path / "repetivel.db"
    inicializar_banco(banco, CAMINHO_DADOS_CURADOS)
    inicializar_banco(banco, CAMINHO_DADOS_CURADOS)
    with sqlite3.connect(banco) as conexao:
        assert (
            conexao.execute("SELECT COUNT(*) FROM startups").fetchone()[0]
            == startups_esperadas
        )
        assert (
            conexao.execute("SELECT COUNT(*) FROM documentos").fetchone()[0]
            == documentos_esperados
        )
        dominios = conexao.execute(
            """
            SELECT startup_id, COUNT(DISTINCT dominio_fonte)
            FROM documentos GROUP BY startup_id ORDER BY startup_id
            """
        ).fetchall()
    assert len(dominios) == startups_esperadas
    assert all(quantidade >= 3 for _, quantidade in dominios)


def test_filtro_estruturado_restringe_as_empresas(base: BaseStartups):
    resultado = base.recuperar(plano(["tecnologia"], setor="Saúde"))
    assert resultado.empresas
    assert all(empresa.setor.casefold() == "saúde" for empresa in resultado.empresas)
    ids_validos = {empresa.id_startup for empresa in resultado.empresas}
    assert all(documento.id_startup in ids_validos for documento in resultado.documentos)


def test_consulta_do_usuario_permanece_em_parametros(caminho_banco):
    entrada = "fintech' OR 1=1 --"
    consulta = montar_consulta_descoberta(plano([entrada]))
    assert entrada not in consulta.sql
    assert entrada in consulta.parametros[0]
    with sqlite3.connect(caminho_banco) as conexao:
        contagem_antes = conexao.execute("SELECT COUNT(*) FROM startups").fetchone()[0]
    # Além da inspeção da fronteira, a entrada hostil executa sem alterar a base.
    resultado = BaseStartups(caminho_banco).recuperar(plano([entrada]))
    assert resultado.empresas == []
    with sqlite3.connect(caminho_banco) as conexao:
        contagem_depois = conexao.execute("SELECT COUNT(*) FROM startups").fetchone()[0]
    assert contagem_depois == contagem_antes


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


# --------------------------------------------------------------------------
# Carga determinística dos documentos completos (fronteira do Extractor).
# --------------------------------------------------------------------------


def documentos_de_duas_startups(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        primeira = conexao.execute(
            "SELECT startup_id, id FROM documentos ORDER BY startup_id, id"
        ).fetchall()
    agrupado: dict[int, list[int]] = {}
    for linha in primeira:
        agrupado.setdefault(linha["startup_id"], []).append(linha["id"])
    ids_startups = sorted(agrupado)
    return ids_startups[0], agrupado[ids_startups[0]], ids_startups[1], agrupado[ids_startups[1]]


def test_carregar_documentos_preserva_ids_ordem_e_conteudo(base: BaseStartups, caminho_banco):
    id_startup, ids, _, _ = documentos_de_duas_startups(caminho_banco)
    pedidos = list(reversed(ids))
    documentos = base.carregar_documentos(id_startup, pedidos)
    assert [documento.id_documento for documento in documentos] == pedidos
    assert all(documento.id_startup == id_startup for documento in documentos)
    assert all(documento.conteudo_texto.strip() for documento in documentos)
    with sqlite3.connect(caminho_banco) as conexao:
        esperado = conexao.execute(
            "SELECT conteudo_texto FROM documentos WHERE id = ?", (pedidos[0],)
        ).fetchone()[0]
    assert documentos[0].conteudo_texto == esperado


def test_carregar_documentos_nao_expoe_classe_referencia(base: BaseStartups, caminho_banco):
    id_startup, ids, _, _ = documentos_de_duas_startups(caminho_banco)
    documento = base.carregar_documentos(id_startup, ids[:1])[0]
    assert "classe_referencia" not in documento.model_dump()


def test_carregar_documentos_recusa_id_de_outra_startup(base: BaseStartups, caminho_banco):
    id_startup, ids, outra, ids_outra = documentos_de_duas_startups(caminho_banco)
    with pytest.raises(ErroDocumentosStartup, match="outra startup"):
        base.carregar_documentos(id_startup, [ids[0], ids_outra[0]])
    assert base.carregar_documentos(outra, ids_outra[:1])[0].id_startup == outra


def test_carregar_documentos_recusa_id_inexistente(base: BaseStartups, caminho_banco):
    id_startup, ids, _, _ = documentos_de_duas_startups(caminho_banco)
    with pytest.raises(ErroDocumentosStartup, match="não existem"):
        base.carregar_documentos(id_startup, [ids[0], 10**9])


def test_carregar_documentos_recusa_ids_repetidos(base: BaseStartups, caminho_banco):
    id_startup, ids, _, _ = documentos_de_duas_startups(caminho_banco)
    with pytest.raises(ErroDocumentosStartup, match="repetidos"):
        base.carregar_documentos(id_startup, [ids[0], ids[0]])


def test_carregar_documentos_recusa_lista_vazia(base: BaseStartups, caminho_banco):
    id_startup, _, _, _ = documentos_de_duas_startups(caminho_banco)
    with pytest.raises(ErroDocumentosStartup, match="nenhum documento"):
        base.carregar_documentos(id_startup, [])


def test_carregar_documentos_mantem_a_entrada_em_parametros(base: BaseStartups, caminho_banco):
    id_startup, ids, _, _ = documentos_de_duas_startups(caminho_banco)
    hostil = "1); DROP TABLE documentos; --"
    with pytest.raises(ErroDocumentosStartup):
        base.carregar_documentos(id_startup, [hostil])
    with sqlite3.connect(caminho_banco) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM documentos").fetchone()[0] > 0
    assert base.carregar_documentos(id_startup, ids[:1])[0].id_documento == ids[0]


# ----------------------------------------------------------------------
# O cache de análises tem versão de schema, não busca de substring no DDL
# ----------------------------------------------------------------------
#
# A decisão era `"motivo_evidencia_insuficiente IS NOT NULL" in sqlite_master.sql`.
# Reformatar o CHECK — quebrar uma linha, dobrar um espaço — bastava para o boot
# considerar o cache obsoleto e DROPAR as 30 análises concluídas, com aviso só
# em log que o Streamlit não mostra. A versão explícita torna a decisão estável.

import logging

from radar.base_startups import (
    SCHEMA_ANALISES_SQL,
    VERSAO_SCHEMA_ANALISES,
    preparar_cache_analises,
)

_BASE_MINIMA = """
CREATE TABLE startups (id INTEGER PRIMARY KEY, nome TEXT NOT NULL);
CREATE TABLE documentos (
    id INTEGER PRIMARY KEY,
    startup_id INTEGER NOT NULL REFERENCES startups(id)
);
"""

_ANALISE_VALIDA = (
    1, "concluida", "AI-native", 42, '{"total": 42}', '{"afirmacoes": []}',
    None, "2026-09-07", "rubrica-v1",
)


def _banco_com_base(tmp_path, nome="radar_teste_cache.db"):
    caminho = tmp_path / nome
    with sqlite3.connect(caminho) as conexao:
        conexao.executescript(_BASE_MINIMA)
        conexao.execute("INSERT INTO startups (id, nome) VALUES (1, 'Exemplo')")
    return caminho


def _versao(caminho) -> int:
    with sqlite3.connect(caminho) as conexao:
        return conexao.execute("PRAGMA user_version").fetchone()[0]


def _analises_gravadas(caminho) -> int:
    with sqlite3.connect(caminho) as conexao:
        return conexao.execute("SELECT COUNT(*) FROM analises").fetchone()[0]


def _inserir_analise(caminho):
    with sqlite3.connect(caminho) as conexao:
        conexao.execute(
            "INSERT INTO analises VALUES (?,?,?,?,?,?,?,?,?)", _ANALISE_VALIDA
        )


def test_banco_novo_cria_o_cache_e_carimba_a_versao(tmp_path):
    caminho = _banco_com_base(tmp_path)

    preparar_cache_analises(caminho)

    assert _versao(caminho) == VERSAO_SCHEMA_ANALISES
    assert _analises_gravadas(caminho) == 0


def test_versao_atual_e_reconhecida_sem_tocar_nas_linhas(tmp_path, caplog):
    caminho = _banco_com_base(tmp_path)
    preparar_cache_analises(caminho)
    _inserir_analise(caminho)

    with caplog.at_level(logging.WARNING):
        preparar_cache_analises(caminho)

    assert _analises_gravadas(caminho) == 1
    assert caplog.text == ""


def test_preparacao_repetida_e_idempotente(tmp_path):
    caminho = _banco_com_base(tmp_path)
    preparar_cache_analises(caminho)
    _inserir_analise(caminho)

    for _ in range(3):
        preparar_cache_analises(caminho)

    assert _analises_gravadas(caminho) == 1
    assert _versao(caminho) == VERSAO_SCHEMA_ANALISES


def test_banco_sem_versao_mas_estruturalmente_atual_preserva_as_analises(tmp_path):
    """A transição de uma vez: carimba a versão sem descartar linha válida."""
    caminho = _banco_com_base(tmp_path)
    with sqlite3.connect(caminho) as conexao:
        conexao.executescript(SCHEMA_ANALISES_SQL)
        conexao.execute(
            "INSERT INTO analises VALUES (?,?,?,?,?,?,?,?,?)", _ANALISE_VALIDA
        )
        conexao.execute("PRAGMA user_version = 0")
    assert _versao(caminho) == 0

    preparar_cache_analises(caminho)

    assert _versao(caminho) == VERSAO_SCHEMA_ANALISES
    assert _analises_gravadas(caminho) == 1


def test_diferenca_inocente_de_formatacao_do_ddl_nao_destroi_o_cache(tmp_path):
    """O defeito exato: reformatar o CHECK apagava o cache inteiro."""
    caminho = _banco_com_base(tmp_path)
    ddl_reformatado = (
        SCHEMA_ANALISES_SQL
        .replace(
            "AND motivo_evidencia_insuficiente IS NOT NULL",
            "AND  motivo_evidencia_insuficiente\n            IS NOT NULL",
        )
    )
    with sqlite3.connect(caminho) as conexao:
        conexao.executescript(ddl_reformatado)
        conexao.execute(
            "INSERT INTO analises VALUES (?,?,?,?,?,?,?,?,?)", _ANALISE_VALIDA
        )
        conexao.execute(f"PRAGMA user_version = {VERSAO_SCHEMA_ANALISES}")

    preparar_cache_analises(caminho)

    assert _analises_gravadas(caminho) == 1


def test_cache_genuinamente_legado_e_recriado_com_aviso(tmp_path, caplog):
    caminho = _banco_com_base(tmp_path)
    with sqlite3.connect(caminho) as conexao:
        # schema antigo: sem `motivo_evidencia_insuficiente`
        conexao.execute(
            "CREATE TABLE analises ("
            " startup_id INTEGER PRIMARY KEY, status TEXT NOT NULL,"
            " perfil_validado_json TEXT NOT NULL, data_execucao TEXT NOT NULL,"
            " versao_rubrica TEXT NOT NULL)"
        )
        conexao.execute(
            "INSERT INTO analises VALUES (1,'concluida','{}','2026-01-01','rubrica-v0')"
        )

    with caplog.at_level(logging.WARNING):
        preparar_cache_analises(caminho)

    assert _versao(caminho) == VERSAO_SCHEMA_ANALISES
    assert _analises_gravadas(caminho) == 0
    assert "analisar_lote" in caplog.text
    # o aviso é sanitizado: nomeia a ação, não despeja DDL nem dado de linha
    assert "CREATE TABLE" not in caplog.text


def test_versao_futura_e_rejeitada_deterministicamente(tmp_path):
    caminho = _banco_com_base(tmp_path)
    preparar_cache_analises(caminho)
    with sqlite3.connect(caminho) as conexao:
        conexao.execute(f"PRAGMA user_version = {VERSAO_SCHEMA_ANALISES + 1}")

    with pytest.raises(ValueError, match="versão"):
        preparar_cache_analises(caminho)


def test_falha_no_meio_da_migracao_faz_rollback(tmp_path, monkeypatch):
    caminho = _banco_com_base(tmp_path)
    with sqlite3.connect(caminho) as conexao:
        conexao.execute(
            "CREATE TABLE analises ("
            " startup_id INTEGER PRIMARY KEY, status TEXT NOT NULL)"
        )
        conexao.execute("INSERT INTO analises VALUES (1,'concluida')")

    import radar.base_startups as modulo

    def explodir(*_args, **_kwargs):
        raise sqlite3.OperationalError("falha simulada no meio da migração")

    monkeypatch.setattr(modulo, "_executar_schema_analises", explodir)

    with pytest.raises(sqlite3.OperationalError):
        preparar_cache_analises(caminho)

    # a tabela legada continua lá: nada foi perdido pela migração interrompida
    with sqlite3.connect(caminho) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM analises").fetchone()[0] == 1


def test_a_decisao_de_schema_nao_usa_substring_do_ddl():
    from pathlib import Path

    from radar.configuracao import RAIZ_PROJETO

    fonte = (RAIZ_PROJETO / "radar" / "base_startups.py").read_text(encoding="utf-8")

    assert "motivo_evidencia_insuficiente IS NOT NULL\" in" not in fonte
    assert "PRAGMA user_version" in fonte
