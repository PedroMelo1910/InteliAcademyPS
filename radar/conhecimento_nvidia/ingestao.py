"""Schema e ingestão da base de conhecimento NVIDIA no ``radar.db``.

O SQLite é o system of record: as linhas de ``chunks_nvidia`` são a fonte do
texto de citação, da URL e dos metadados. ``vetores_nvidia`` (vec0, métrica
de cosseno declarada na coluna) guarda somente vetores derivados associados
aos ids dos chunks, e ``cache_embeddings_nvidia`` é dado derivado que evita
chamadas repetidas ao provedor — nunca uma segunda fonte de verdade.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from radar.configuracao import (
    DIMENSAO_EMBEDDING_NVIDIA,
    TAMANHO_LOTE_EMBEDDING,
    TETO_CARACTERES_CHUNK,
)
from radar.contratos import TECNOLOGIAS_NVIDIA
from radar.conhecimento_nvidia.chunking import gerar_chunks, texto_para_embedding
from radar.conhecimento_nvidia.fontes import carregar_fontes, validar_cobertura
from radar.provedores import EmbeddingProvider


_LISTA_TECNOLOGIAS_SQL = ", ".join(f"'{nome}'" for nome in TECNOLOGIAS_NVIDIA)

SCHEMA_CHUNKS_SQL = f"""
CREATE TABLE IF NOT EXISTS chunks_nvidia (
    id INTEGER PRIMARY KEY,
    topico TEXT NOT NULL,
    origem TEXT NOT NULL CHECK (origem IN ('tecnologia', 'conceitual')),
    tecnologia TEXT CHECK (tecnologia IS NULL OR tecnologia IN ({_LISTA_TECNOLOGIAS_SQL})),
    breadcrumb TEXT NOT NULL,
    texto_limpo TEXT NOT NULL,
    fonte_url TEXT NOT NULL,
    indice_parte INTEGER NOT NULL CHECK (indice_parte >= 1),
    hash_texto TEXT NOT NULL,
    UNIQUE (fonte_url, breadcrumb, indice_parte),
    CHECK ((origem = 'tecnologia') = (tecnologia IS NOT NULL))
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_nvidia_fts USING fts5(
    breadcrumb,
    texto_limpo,
    content='chunks_nvidia',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS cache_embeddings_nvidia (
    hash_texto TEXT NOT NULL,
    modelo TEXT NOT NULL,
    vetor BLOB NOT NULL,
    PRIMARY KEY (hash_texto, modelo)
);

CREATE TABLE IF NOT EXISTS metadados_indice_nvidia (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""

CHAVE_MODELO_EMBEDDING = "modelo_embedding"
CHAVE_DIMENSAO_EMBEDDING = "dimensao_embedding"


def conectar_conhecimento(caminho_banco: Path) -> sqlite3.Connection:
    """Conexão com a extensão sqlite-vec carregada; determinística, zero rede.

    ``isolation_level=None`` deixa a conexão em autocommit para que a fase de
    gravação da ingestão controle a transação explicitamente (BEGIN/COMMIT).
    """
    conexao = sqlite3.connect(caminho_banco, isolation_level=None)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA foreign_keys = ON")
    conexao.enable_load_extension(True)
    sqlite_vec.load(conexao)
    conexao.enable_load_extension(False)
    return conexao


def criar_schema(
    conexao: sqlite3.Connection, dimensao: int = DIMENSAO_EMBEDDING_NVIDIA
) -> None:
    conexao.executescript(SCHEMA_CHUNKS_SQL)
    conexao.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vetores_nvidia USING vec0("
        f"embedding float[{int(dimensao)}] distance_metric=cosine)"
    )


def ler_metadados_indice(
    conexao: sqlite3.Connection,
) -> tuple[str, int] | None:
    """Retorna o modelo e a dimensão que produziram o índice vetorial atual."""
    linhas = conexao.execute(
        "SELECT chave, valor FROM metadados_indice_nvidia WHERE chave IN (?, ?)",
        (CHAVE_MODELO_EMBEDDING, CHAVE_DIMENSAO_EMBEDDING),
    ).fetchall()
    if not linhas:
        return None
    valores = {linha["chave"]: linha["valor"] for linha in linhas}
    if set(valores) != {CHAVE_MODELO_EMBEDDING, CHAVE_DIMENSAO_EMBEDDING}:
        raise ErroIngestaoNvidia(
            "metadados do índice NVIDIA estão incompletos; reingestão obrigatória"
        )
    try:
        dimensao = int(valores[CHAVE_DIMENSAO_EMBEDDING])
    except ValueError as erro:
        raise ErroIngestaoNvidia(
            "dimensão registrada no índice NVIDIA não é um inteiro válido"
        ) from erro
    return valores[CHAVE_MODELO_EMBEDDING], dimensao


def _gravar_metadados_indice(
    conexao: sqlite3.Connection, modelo: str, dimensao: int
) -> None:
    conexao.executemany(
        "INSERT INTO metadados_indice_nvidia (chave, valor) VALUES (?, ?) "
        "ON CONFLICT(chave) DO UPDATE SET valor = excluded.valor",
        (
            (CHAVE_MODELO_EMBEDDING, modelo),
            (CHAVE_DIMENSAO_EMBEDDING, str(dimensao)),
        ),
    )


def serializar_vetor(valores: list[float]) -> bytes:
    return struct.pack(f"<{len(valores)}f", *valores)


def desserializar_vetor(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class ErroIngestaoNvidia(RuntimeError):
    """Invariante da ingestão violada; nada foi gravado de forma parcial."""


@dataclass(frozen=True)
class ResumoIngestao:
    fontes: int
    chunks_totais: int
    chunks_inseridos: int
    chunks_atualizados: int
    chunks_removidos: int
    chunks_inalterados: int
    embeddings_calculados: int
    chamadas_embedding: int


def _reconstruir_fts(conexao: sqlite3.Connection) -> None:
    """Sincroniza o índice FTS5 com a tabela de conteúdo (base estática)."""
    conexao.execute("INSERT INTO chunks_nvidia_fts(chunks_nvidia_fts) VALUES('rebuild')")


def _ler_cache(
    conexao: sqlite3.Connection, modelo: str, hashes: list[str]
) -> dict[str, list[float]]:
    vetores: dict[str, list[float]] = {}
    for inicio in range(0, len(hashes), 500):
        parte = hashes[inicio : inicio + 500]
        marcadores = ", ".join("?" for _ in parte)
        linhas = conexao.execute(
            "SELECT hash_texto, vetor FROM cache_embeddings_nvidia "
            f"WHERE modelo = ? AND hash_texto IN ({marcadores})",
            (modelo, *parte),
        ).fetchall()
        for linha in linhas:
            vetores[linha["hash_texto"]] = desserializar_vetor(linha["vetor"])
    return vetores


def ingerir(
    caminho_banco: Path,
    diretorio_fontes: Path,
    provedor: EmbeddingProvider,
    *,
    exigir_cobertura: bool = True,
    teto: int = TETO_CARACTERES_CHUNK,
    tamanho_lote: int = TAMANHO_LOTE_EMBEDDING,
) -> ResumoIngestao:
    """Ingestão em duas fases.

    Fase A (fora da transação): valida o manifesto, gera os chunks, calcula
    hashes, identifica embeddings ausentes no cache, chama o provedor em
    lotes controlados e valida dimensão/finitude de cada vetor.

    Fase B (uma transação SQLite): grava cache, chunks, FTS e vetores e
    remove tudo que ficou obsoleto. Qualquer falha reverte a transação
    inteira; o banco nunca fica parcialmente atualizado.
    """
    # ---------- Fase A ----------
    fontes = carregar_fontes(diretorio_fontes)
    if exigir_cobertura:
        validar_cobertura(fontes)
    chunks = [chunk for fonte in fontes for chunk in gerar_chunks(fonte, teto)]

    caminho_banco.parent.mkdir(parents=True, exist_ok=True)
    conexao = conectar_conhecimento(caminho_banco)
    try:
        criar_schema(conexao, dimensao=provedor.dimensao)
        modelo = provedor.modelo
        metadados_anteriores = ler_metadados_indice(conexao)
        total_existente = conexao.execute(
            "SELECT count(*) AS n FROM chunks_nvidia"
        ).fetchone()["n"]
        reindexar_todos = (
            total_existente > 0
            and metadados_anteriores != (modelo, provedor.dimensao)
        )
        recriar_tabela_vetorial = (
            metadados_anteriores is None
            or metadados_anteriores[1] != provedor.dimensao
        )

        textos_por_hash: dict[str, str] = {}
        for chunk in chunks:
            textos_por_hash.setdefault(chunk.hash_texto, texto_para_embedding(chunk))

        vetores = _ler_cache(conexao, modelo, list(textos_por_hash))
        # Cache é dado derivado: entradas truncadas, não finitas ou de uma
        # dimensão antiga são descartadas e recalculadas pelo provedor.
        vetores = {
            hash_texto: vetor
            for hash_texto, vetor in vetores.items()
            if len(vetor) == provedor.dimensao
            and all(math.isfinite(valor) for valor in vetor)
        }
        faltantes = [h for h in textos_por_hash if h not in vetores]

        chamadas_embedding = 0
        for inicio in range(0, len(faltantes), tamanho_lote):
            lote = faltantes[inicio : inicio + tamanho_lote]
            novos = provedor.embutir_passagens([textos_por_hash[h] for h in lote])
            chamadas_embedding += 1
            if len(novos) != len(lote):
                raise ErroIngestaoNvidia(
                    f"o provedor devolveu {len(novos)} vetores para {len(lote)} textos"
                )
            for hash_texto, vetor in zip(lote, novos):
                if len(vetor) != provedor.dimensao:
                    raise ErroIngestaoNvidia(
                        f"vetor com {len(vetor)} dimensões; o provedor declara "
                        f"{provedor.dimensao}"
                    )
                if not all(math.isfinite(valor) for valor in vetor):
                    raise ErroIngestaoNvidia("vetor com valor não finito do provedor")
                vetores[hash_texto] = list(vetor)

        # ---------- Fase B ----------
        conexao.execute("BEGIN IMMEDIATE")
        try:
            for hash_texto in faltantes:
                conexao.execute(
                    "INSERT OR REPLACE INTO cache_embeddings_nvidia "
                    "(hash_texto, modelo, vetor) VALUES (?, ?, ?)",
                    (hash_texto, modelo, serializar_vetor(vetores[hash_texto])),
                )

            existentes = {
                (linha["fonte_url"], linha["breadcrumb"], linha["indice_parte"]): linha
                for linha in conexao.execute(
                    "SELECT id, fonte_url, breadcrumb, indice_parte, hash_texto, "
                    "topico, origem, tecnologia, texto_limpo "
                    "FROM chunks_nvidia"
                )
            }

            if recriar_tabela_vetorial:
                conexao.execute("DROP TABLE vetores_nvidia")
                conexao.execute(
                    "CREATE VIRTUAL TABLE vetores_nvidia USING vec0("
                    f"embedding float[{int(provedor.dimensao)}] distance_metric=cosine)"
                )

            inseridos = atualizados = inalterados = 0
            chaves_atuais: set[tuple[str, str, int]] = set()
            for chunk in chunks:
                chave = (str(chunk.fonte_url), chunk.breadcrumb, chunk.indice_parte)
                chaves_atuais.add(chave)
                vetor_serializado = serializar_vetor(vetores[chunk.hash_texto])
                if chave not in existentes:
                    cursor = conexao.execute(
                        "INSERT INTO chunks_nvidia (topico, origem, tecnologia,"
                        " breadcrumb, texto_limpo, fonte_url, indice_parte, hash_texto)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            chunk.topico,
                            chunk.origem,
                            chunk.tecnologia,
                            chunk.breadcrumb,
                            chunk.texto_limpo,
                            str(chunk.fonte_url),
                            chunk.indice_parte,
                            chunk.hash_texto,
                        ),
                    )
                    conexao.execute(
                        "INSERT INTO vetores_nvidia(rowid, embedding) VALUES (?, ?)",
                        (cursor.lastrowid, vetor_serializado),
                    )
                    inseridos += 1
                else:
                    existente = existentes[chave]
                    id_atual = existente["id"]
                    dados_mudaram = (
                        existente["hash_texto"] != chunk.hash_texto
                        or existente["topico"] != chunk.topico
                        or existente["origem"] != chunk.origem
                        or existente["tecnologia"] != chunk.tecnologia
                        or existente["texto_limpo"] != chunk.texto_limpo
                    )
                    vetor_mudou = (
                        existente["hash_texto"] != chunk.hash_texto
                        or reindexar_todos
                    )
                    if dados_mudaram:
                        conexao.execute(
                            "UPDATE chunks_nvidia SET topico = ?, origem = ?,"
                            " tecnologia = ?, texto_limpo = ?, hash_texto = ?"
                            " WHERE id = ?",
                            (
                                chunk.topico,
                                chunk.origem,
                                chunk.tecnologia,
                                chunk.texto_limpo,
                                chunk.hash_texto,
                                id_atual,
                            ),
                        )
                    if vetor_mudou:
                        # A tabela virtual vec0 não participa de cascatas; a linha
                        # obsoleta é removida e regravada explicitamente.
                        conexao.execute(
                            "DELETE FROM vetores_nvidia WHERE rowid = ?", (id_atual,)
                        )
                        conexao.execute(
                            "INSERT INTO vetores_nvidia(rowid, embedding) VALUES (?, ?)",
                            (id_atual, vetor_serializado),
                        )
                    if dados_mudaram or vetor_mudou:
                        atualizados += 1
                    else:
                        inalterados += 1

            removidos = 0
            for chave, existente in existentes.items():
                if chave not in chaves_atuais:
                    id_antigo = existente["id"]
                    conexao.execute(
                        "DELETE FROM chunks_nvidia WHERE id = ?", (id_antigo,)
                    )
                    conexao.execute(
                        "DELETE FROM vetores_nvidia WHERE rowid = ?", (id_antigo,)
                    )
                    removidos += 1

            _gravar_metadados_indice(conexao, modelo, provedor.dimensao)
            _reconstruir_fts(conexao)
            conexao.execute("COMMIT")
        except BaseException:
            conexao.execute("ROLLBACK")
            raise

        return ResumoIngestao(
            fontes=len(fontes),
            chunks_totais=len(chunks),
            chunks_inseridos=inseridos,
            chunks_atualizados=atualizados,
            chunks_removidos=removidos,
            chunks_inalterados=inalterados,
            embeddings_calculados=len(faltantes),
            chamadas_embedding=chamadas_embedding,
        )
    finally:
        conexao.close()
