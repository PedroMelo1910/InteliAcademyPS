"""Schema e ingestão da base de conhecimento NVIDIA: tabelas, FTS5, vec0 com
cosseno, ingestão transacional em duas fases, cache e remoção de obsoletos."""

import sqlite3
from pathlib import Path

import pytest

from tests.conftest import EmbeddingFalso
from radar.conhecimento_nvidia import ingestao as modulo_ingestao
from radar.conhecimento_nvidia.fontes import ErroFonteNvidia
from radar.conhecimento_nvidia.ingestao import (
    ErroIngestaoNvidia,
    conectar_conhecimento,
    criar_schema,
    ingerir,
    serializar_vetor,
)


CORPO_TRITON = """# Visão geral

O Triton Inference Server executa modelos de IA em produção com baixa latência.

# Recursos

## Batching dinâmico

O batching dinâmico agrupa requisições para aumentar o throughput na GPU.
"""

CORPO_RAPIDS = """# Visão geral

RAPIDS acelera pipelines de dados com dataframes na GPU.
"""

CORPO_CONCEITO = """# Serviços AI-native

Empresas AI-native vendem resultado operacional combinando software e agentes.
"""


def escrever_fonte(
    diretorio: Path,
    nome: str,
    *,
    topico: str,
    origem: str,
    tecnologia: str | None,
    url: str,
    titulo: str,
    corpo: str,
) -> Path:
    linhas = [
        "---",
        f"topico: {topico}",
        f"origem: {origem}",
    ]
    if tecnologia is not None:
        linhas.append(f"tecnologia: {tecnologia}")
    linhas += [
        f"fonte_url: {url}",
        f"titulo: {titulo}",
        "data_acesso: 2026-08-25",
        "---",
        "",
        corpo,
    ]
    caminho = diretorio / nome
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


@pytest.fixture
def diretorio_fontes(tmp_path):
    diretorio = tmp_path / "fontes"
    diretorio.mkdir()
    escrever_fonte(
        diretorio, "01_triton.md", topico="triton", origem="tecnologia",
        tecnologia="NVIDIA Triton Inference Server",
        url="https://exemplo.nvidia.com/triton",
        titulo="NVIDIA Triton Inference Server", corpo=CORPO_TRITON,
    )
    escrever_fonte(
        diretorio, "02_rapids.md", topico="rapids", origem="tecnologia",
        tecnologia="NVIDIA RAPIDS", url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS", corpo=CORPO_RAPIDS,
    )
    escrever_fonte(
        diretorio, "03_conceito.md", topico="ai_native_services", origem="conceitual",
        tecnologia=None, url="https://exemplo.com/ai-native",
        titulo="Serviços AI-native", corpo=CORPO_CONCEITO,
    )
    return diretorio


def contar(conexao, tabela: str) -> int:
    return conexao.execute(f"SELECT count(*) AS n FROM {tabela}").fetchone()["n"]


@pytest.fixture
def conexao(tmp_path):
    conexao = conectar_conhecimento(tmp_path / "kb.db")
    criar_schema(conexao, dimensao=4)
    yield conexao
    conexao.close()


def test_criar_schema_cria_tabelas_e_e_idempotente(tmp_path):
    caminho = tmp_path / "kb.db"
    conexao = conectar_conhecimento(caminho)
    criar_schema(conexao, dimensao=4)
    criar_schema(conexao, dimensao=4)  # segunda chamada não pode falhar
    tabelas = {
        linha["name"]
        for linha in conexao.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    for esperada in ("chunks_nvidia", "chunks_nvidia_fts", "vetores_nvidia",
                     "cache_embeddings_nvidia", "metadados_indice_nvidia"):
        assert esperada in tabelas
    conexao.close()


def test_vec0_usa_distancia_de_cosseno(conexao):
    # Com métrica L2 o vetor [0.9, 0.5, 0, 0] estaria mais perto de [1, 0, 0, 0]
    # do que [10, 0, 0, 0]; com cosseno o colinear vence com distância 0.
    conexao.execute(
        "INSERT INTO vetores_nvidia(rowid, embedding) VALUES (1, ?)",
        (serializar_vetor([10.0, 0.0, 0.0, 0.0]),),
    )
    conexao.execute(
        "INSERT INTO vetores_nvidia(rowid, embedding) VALUES (2, ?)",
        (serializar_vetor([0.9, 0.5, 0.0, 0.0]),),
    )
    linhas = conexao.execute(
        "SELECT rowid, distance FROM vetores_nvidia "
        "WHERE embedding MATCH ? AND k = 2 ORDER BY distance",
        (serializar_vetor([1.0, 0.0, 0.0, 0.0]),),
    ).fetchall()
    assert [linha["rowid"] for linha in linhas] == [1, 2]
    assert linhas[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_chunks_nvidia_valida_origem_e_tecnologia_no_sql(conexao):
    def inserir(origem, tecnologia):
        conexao.execute(
            "INSERT INTO chunks_nvidia (topico, origem, tecnologia, breadcrumb,"
            " texto_limpo, fonte_url, indice_parte, hash_texto)"
            " VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            (
                "nim", origem, tecnologia, "NVIDIA NIM",
                "texto", f"https://exemplo.com/{origem}-{tecnologia}", "a" * 64,
            ),
        )

    inserir("tecnologia", "NVIDIA NIM")
    inserir("conceitual", None)
    with pytest.raises(sqlite3.IntegrityError):
        inserir("tecnologia", None)
    with pytest.raises(sqlite3.IntegrityError):
        inserir("conceitual", "NVIDIA NIM")
    with pytest.raises(sqlite3.IntegrityError):
        inserir("tecnologia", "NVIDIA Inventada")


def test_vec0_rejeita_dimensao_errada(conexao):
    with pytest.raises(sqlite3.Error):
        conexao.execute(
            "INSERT INTO vetores_nvidia(rowid, embedding) VALUES (1, ?)",
            (serializar_vetor([1.0, 2.0]),),
        )


def test_ingestao_completa_grava_chunks_fts_vetores_e_cache(tmp_path, diretorio_fontes):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    resumo = ingerir(
        caminho, diretorio_fontes, provedor, exigir_cobertura=False, tamanho_lote=2
    )
    assert resumo.fontes == 3
    assert resumo.chunks_totais == 4  # a seção vazia "Recursos" é ignorada
    assert resumo.chunks_inseridos == 4
    assert resumo.embeddings_calculados == 4
    assert resumo.chamadas_embedding == 2  # 4 textos em lotes de 2
    conexao = conectar_conhecimento(caminho)
    assert contar(conexao, "chunks_nvidia") == 4
    assert contar(conexao, "vetores_nvidia") == 4
    assert contar(conexao, "cache_embeddings_nvidia") == 4
    achado = conexao.execute(
        "SELECT rowid FROM chunks_nvidia_fts WHERE chunks_nvidia_fts MATCH 'triton'"
    ).fetchall()
    assert achado
    conexao.close()


def test_reingestao_identica_e_idempotente(tmp_path, diretorio_fontes):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    chamadas_apos_primeira = provedor.chamadas_passagens
    conexao = conectar_conhecimento(caminho)
    ids_primeira = {
        linha["id"] for linha in conexao.execute("SELECT id FROM chunks_nvidia")
    }
    conexao.close()

    resumo = ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    assert provedor.chamadas_passagens == chamadas_apos_primeira
    assert resumo.embeddings_calculados == 0
    assert resumo.chunks_inseridos == 0
    assert resumo.chunks_atualizados == 0
    assert resumo.chunks_removidos == 0
    assert resumo.chunks_inalterados == resumo.chunks_totais
    conexao = conectar_conhecimento(caminho)
    ids_segunda = {
        linha["id"] for linha in conexao.execute("SELECT id FROM chunks_nvidia")
    }
    assert ids_segunda == ids_primeira
    assert contar(conexao, "vetores_nvidia") == len(ids_primeira)
    conexao.close()


def test_troca_de_modelo_regrava_todos_os_vetores_e_metadados(
    tmp_path, diretorio_fontes
):
    caminho = tmp_path / "kb.db"
    primeiro = EmbeddingFalso(modelo="modelo-a")
    ingerir(caminho, diretorio_fontes, primeiro, exigir_cobertura=False)

    segundo = EmbeddingFalso(modelo="modelo-b")
    resumo = ingerir(caminho, diretorio_fontes, segundo, exigir_cobertura=False)

    assert resumo.embeddings_calculados == resumo.chunks_totais
    assert resumo.chunks_atualizados == resumo.chunks_totais
    assert resumo.chunks_inalterados == 0
    conexao = conectar_conhecimento(caminho)
    metadados = dict(
        conexao.execute("SELECT chave, valor FROM metadados_indice_nvidia").fetchall()
    )
    assert metadados == {"dimensao_embedding": "32", "modelo_embedding": "modelo-b"}
    assert contar(conexao, "vetores_nvidia") == resumo.chunks_totais
    conexao.close()


def test_troca_de_dimensao_recria_tabela_vetorial(
    tmp_path, diretorio_fontes
):
    caminho = tmp_path / "kb.db"
    ingerir(
        caminho,
        diretorio_fontes,
        EmbeddingFalso(dimensao=32),
        exigir_cobertura=False,
    )

    resumo = ingerir(
        caminho,
        diretorio_fontes,
        EmbeddingFalso(dimensao=16),
        exigir_cobertura=False,
    )

    assert resumo.embeddings_calculados == resumo.chunks_totais
    conexao = conectar_conhecimento(caminho)
    metadados = dict(
        conexao.execute("SELECT chave, valor FROM metadados_indice_nvidia").fetchall()
    )
    assert metadados["dimensao_embedding"] == "16"
    assert contar(conexao, "vetores_nvidia") == resumo.chunks_totais
    conexao.close()


def test_cache_corrompido_e_recalculado(tmp_path, diretorio_fontes):
    caminho = tmp_path / "kb.db"
    primeiro = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, primeiro, exigir_cobertura=False)
    conexao = conectar_conhecimento(caminho)
    hash_texto = conexao.execute(
        "SELECT hash_texto FROM chunks_nvidia ORDER BY id LIMIT 1"
    ).fetchone()["hash_texto"]
    conexao.execute(
        "UPDATE cache_embeddings_nvidia SET vetor = ? WHERE hash_texto = ?",
        (b"\x00\x00\x80?", hash_texto),
    )
    conexao.close()

    segundo = EmbeddingFalso()
    resumo = ingerir(caminho, diretorio_fontes, segundo, exigir_cobertura=False)

    assert resumo.embeddings_calculados == 1
    assert segundo.chamadas_passagens == 1
    conexao = conectar_conhecimento(caminho)
    tamanho = conexao.execute(
        "SELECT length(vetor) AS n FROM cache_embeddings_nvidia WHERE hash_texto = ?",
        (hash_texto,),
    ).fetchone()["n"]
    assert tamanho == segundo.dimensao * 4
    conexao.close()


def test_mudanca_apenas_de_metadado_atualiza_chunk_sem_reembutir(
    tmp_path, diretorio_fontes
):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    escrever_fonte(
        diretorio_fontes,
        "02_rapids.md",
        topico="rapids_atualizado",
        origem="tecnologia",
        tecnologia="NVIDIA RAPIDS",
        url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS",
        corpo=CORPO_RAPIDS,
    )

    resumo = ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    assert resumo.embeddings_calculados == 0
    assert resumo.chunks_atualizados == 1
    conexao = conectar_conhecimento(caminho)
    assert conexao.execute(
        "SELECT topico FROM chunks_nvidia WHERE fonte_url = ?",
        ("https://exemplo.nvidia.com/rapids",),
    ).fetchone()["topico"] == "rapids_atualizado"
    conexao.close()

def test_alteracao_parcial_reembeda_somente_o_chunk_mudado(tmp_path, diretorio_fontes):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    conexao = conectar_conhecimento(caminho)
    id_rapids = conexao.execute(
        "SELECT id FROM chunks_nvidia WHERE topico = 'rapids'"
    ).fetchone()["id"]
    conexao.close()

    escrever_fonte(
        diretorio_fontes, "02_rapids.md", topico="rapids", origem="tecnologia",
        tecnologia="NVIDIA RAPIDS", url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS",
        corpo=CORPO_RAPIDS.replace("acelera", "acelera muito"),
    )
    resumo = ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    assert resumo.chunks_atualizados == 1
    assert resumo.embeddings_calculados == 1
    assert resumo.chunks_inseridos == 0
    assert resumo.chunks_removidos == 0
    conexao = conectar_conhecimento(caminho)
    linha = conexao.execute(
        "SELECT id, texto_limpo FROM chunks_nvidia WHERE topico = 'rapids'"
    ).fetchone()
    assert linha["id"] == id_rapids  # identidade estável pela chave natural
    assert "acelera muito" in linha["texto_limpo"]
    conexao.close()


def test_chunks_obsoletos_somem_do_banco_do_fts_e_do_vec0(tmp_path, diretorio_fontes):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)

    (diretorio_fontes / "02_rapids.md").unlink()
    resumo = ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    assert resumo.chunks_removidos == 1
    conexao = conectar_conhecimento(caminho)
    assert contar(conexao, "chunks_nvidia") == 3
    assert contar(conexao, "vetores_nvidia") == 3
    assert not conexao.execute(
        "SELECT rowid FROM chunks_nvidia_fts WHERE chunks_nvidia_fts MATCH 'rapids'"
    ).fetchall()
    conexao.close()


def test_falha_na_fase_de_gravacao_nao_deixa_banco_parcial(
    tmp_path, diretorio_fontes, monkeypatch
):
    caminho = tmp_path / "kb.db"
    provedor = EmbeddingFalso()
    ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)
    conexao = conectar_conhecimento(caminho)
    estado_antes = conexao.execute(
        "SELECT fonte_url, breadcrumb, indice_parte, hash_texto FROM chunks_nvidia "
        "ORDER BY id"
    ).fetchall()
    cache_antes = contar(conexao, "cache_embeddings_nvidia")
    conexao.close()

    escrever_fonte(
        diretorio_fontes, "02_rapids.md", topico="rapids", origem="tecnologia",
        tecnologia="NVIDIA RAPIDS", url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS",
        corpo=CORPO_RAPIDS.replace("acelera", "transforma"),
    )

    def explodir(_conexao):
        raise RuntimeError("falha simulada na sincronização do FTS")

    monkeypatch.setattr(modulo_ingestao, "_reconstruir_fts", explodir)
    with pytest.raises(RuntimeError, match="falha simulada"):
        ingerir(caminho, diretorio_fontes, provedor, exigir_cobertura=False)

    conexao = conectar_conhecimento(caminho)
    estado_depois = conexao.execute(
        "SELECT fonte_url, breadcrumb, indice_parte, hash_texto FROM chunks_nvidia "
        "ORDER BY id"
    ).fetchall()
    assert [tuple(linha) for linha in estado_depois] == [
        tuple(linha) for linha in estado_antes
    ]
    assert contar(conexao, "cache_embeddings_nvidia") == cache_antes
    assert contar(conexao, "vetores_nvidia") == len(estado_antes)
    conexao.close()


def test_cobertura_e_exigida_por_padrao(tmp_path, diretorio_fontes):
    with pytest.raises(ErroFonteNvidia, match="cobertura incompleta"):
        ingerir(tmp_path / "kb.db", diretorio_fontes, EmbeddingFalso())


def test_vetor_com_dimensao_errada_falha_antes_de_gravar(tmp_path, diretorio_fontes):
    class EmbeddingTorto:
        modelo = "torto"
        dimensao = 8

        def embutir_passagens(self, textos):
            return [[1.0, 2.0] for _ in textos]  # 2 dimensões, não 8

        def embutir_consulta(self, texto):
            return [1.0, 2.0]

    caminho = tmp_path / "kb.db"
    with pytest.raises(ErroIngestaoNvidia, match="dimens"):
        ingerir(caminho, diretorio_fontes, EmbeddingTorto(), exigir_cobertura=False)
    conexao = conectar_conhecimento(caminho)
    assert contar(conexao, "chunks_nvidia") == 0
    conexao.close()
