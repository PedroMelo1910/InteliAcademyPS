"""Consulta híbrida da base de conhecimento NVIDIA.

Pipeline: lexical (FTS5/BM25, top 20) + vetorial (vec0 cosseno, top 20) →
Reciprocal Rank Fusion (k=60) → reranking dos 20 melhores fundidos → top 6
dentro da faixa 5–8 do ``ContextoNvidia``. Constantes iniciais avaliáveis,
não afirmação de otimalidade. Ordenação determinística: empates na fusão
resolvem por ``id_chunk``; empates no reranking preservam a ordem da fusão.
A rede só aparece atrás dos provedores injetados.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

from radar.configuracao import (
    K_LEXICAL_NVIDIA,
    K_RRF,
    K_VETORIAL_NVIDIA,
    N_CANDIDATOS_RERANK,
    N_TRECHOS_FINAL,
)
from radar.contratos import ContextoNvidia, TrechoNvidia
from radar.conhecimento_nvidia.ingestao import (
    conectar_conhecimento,
    ler_metadados_indice,
    serializar_vetor,
)
from radar.provedores import EmbeddingProvider, RerankProvider


class ErroConsultaNvidia(RuntimeError):
    """Consulta recusada porque o índice ou um provedor violou seu contrato."""


def _montar_match(consulta: str) -> str | None:
    tokens: list[str] = []
    vistos: set[str] = set()
    for token in re.findall(r"\w+", consulta):
        chave = token.casefold()
        if chave not in vistos:
            vistos.add(chave)
            tokens.append(token)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def buscar_lexical(
    conexao: sqlite3.Connection, consulta: str, k: int = K_LEXICAL_NVIDIA
) -> list[int]:
    match = _montar_match(consulta)
    if match is None:
        return []
    linhas = conexao.execute(
        "SELECT rowid FROM chunks_nvidia_fts WHERE chunks_nvidia_fts MATCH ? "
        "ORDER BY bm25(chunks_nvidia_fts) ASC, rowid ASC LIMIT ?",
        (match, k),
    ).fetchall()
    return [linha["rowid"] for linha in linhas]


def buscar_vetorial(
    conexao: sqlite3.Connection, vetor: list[float], k: int = K_VETORIAL_NVIDIA
) -> list[int]:
    # O vec0 só admite "ORDER BY distance" puro na consulta KNN; o desempate
    # determinístico por rowid é aplicado aqui, sobre as distâncias retornadas.
    linhas = conexao.execute(
        "SELECT rowid, distance FROM vetores_nvidia WHERE embedding MATCH ? "
        "AND k = ? ORDER BY distance",
        (serializar_vetor(vetor), k),
    ).fetchall()
    ordenadas = sorted(linhas, key=lambda linha: (linha["distance"], linha["rowid"]))
    return [linha["rowid"] for linha in ordenadas]


def fusao_rrf(listas: list[list[int]], k_rrf: int = K_RRF) -> list[int]:
    """Fusão por rank (RRF): score = Σ 1/(k + posição), posições 1-based.

    Funde posições, nunca médias de scores brutos incomensuráveis (BM25
    negativo × distância de cosseno). Empate resolve por id ascendente.
    """
    scores: dict[int, float] = {}
    for lista in listas:
        for posicao, id_chunk in enumerate(lista, start=1):
            scores[id_chunk] = scores.get(id_chunk, 0.0) + 1.0 / (k_rrf + posicao)
    return sorted(scores, key=lambda id_chunk: (-scores[id_chunk], id_chunk))


class ConhecimentoNvidia:
    """Boundary direto do RAG NVIDIA: ``consultar(consulta) -> ContextoNvidia``.

    Independente do grafo multi-agente: recebe a consulta pronta e devolve
    trechos citáveis com breadcrumb, texto limpo e URL vindos do SQLite.
    """

    def __init__(
        self,
        caminho_banco: Path,
        embedding: EmbeddingProvider,
        rerank: RerankProvider,
    ):
        self.caminho_banco = caminho_banco
        self._embedding = embedding
        self._rerank = rerank

    def consultar(self, consulta: str) -> ContextoNvidia:
        consulta = consulta.strip()
        if not consulta:
            raise ErroConsultaNvidia("a consulta NVIDIA não pode ser vazia")
        conexao = conectar_conhecimento(self.caminho_banco)
        try:
            try:
                metadados = ler_metadados_indice(conexao)
            except (sqlite3.Error, RuntimeError) as erro:
                raise ErroConsultaNvidia(
                    "índice NVIDIA ausente ou com metadados inválidos; execute a ingestão"
                ) from erro
            if metadados is None:
                raise ErroConsultaNvidia(
                    "índice NVIDIA ainda não foi ingerido; execute a ingestão"
                )
            modelo_indice, dimensao_indice = metadados
            if (
                modelo_indice != self._embedding.modelo
                or dimensao_indice != self._embedding.dimensao
            ):
                raise ErroConsultaNvidia(
                    "modelo de embedding da consulta não corresponde ao índice: "
                    f"índice={modelo_indice}/{dimensao_indice}, "
                    f"consulta={self._embedding.modelo}/{self._embedding.dimensao}"
                )
            ids_lexicais = buscar_lexical(conexao, consulta)
            vetor_consulta = self._embedding.embutir_consulta(consulta)
            if len(vetor_consulta) != dimensao_indice or not all(
                math.isfinite(valor) for valor in vetor_consulta
            ):
                raise ErroConsultaNvidia(
                    "embedding da consulta possui dimensão ou valores inválidos"
                )
            ids_vetoriais = buscar_vetorial(conexao, vetor_consulta)
            fundidos = fusao_rrf([ids_lexicais, ids_vetoriais])
            candidatos = fundidos[:N_CANDIDATOS_RERANK]

            linhas = self._carregar_chunks(conexao, candidatos)
        finally:
            conexao.close()

        if len(linhas) < 5:
            raise ErroConsultaNvidia(
                "o corpus recuperou menos de 5 chunks; reingira uma base com ao menos 5"
            )

        textos = [
            f"{linha['breadcrumb']}\n\n{linha['texto_limpo']}" for linha in linhas
        ]
        scores = self._rerank.reordenar(consulta, textos)
        if len(scores) != len(textos) or not all(
            math.isfinite(score) for score in scores
        ):
            raise ErroConsultaNvidia(
                "o reranker devolveu quantidade de scores ou valores inválidos"
            )

        # Empate de score preserva a ordem da fusão (posição) e, por fim, o id.
        ordem = sorted(
            range(len(linhas)),
            key=lambda indice: (-scores[indice], indice, linhas[indice]["id"]),
        )[:N_TRECHOS_FINAL]

        trechos = [
            TrechoNvidia(
                id_chunk=linhas[indice]["id"],
                topico=linhas[indice]["topico"],
                origem=linhas[indice]["origem"],
                tecnologia=linhas[indice]["tecnologia"],
                breadcrumb=linhas[indice]["breadcrumb"],
                texto=linhas[indice]["texto_limpo"],
                fonte_url=linhas[indice]["fonte_url"],
                score_rerank=scores[indice],
            )
            for indice in ordem
        ]
        return ContextoNvidia(consulta_gerada=consulta, trechos=trechos)

    @staticmethod
    def _carregar_chunks(
        conexao: sqlite3.Connection, ids: list[int]
    ) -> list[sqlite3.Row]:
        if not ids:
            return []
        marcadores = ", ".join("?" for _ in ids)
        linhas = conexao.execute(
            "SELECT id, topico, origem, tecnologia, breadcrumb, texto_limpo,"
            f" fonte_url FROM chunks_nvidia WHERE id IN ({marcadores})",
            tuple(ids),
        ).fetchall()
        por_id = {linha["id"]: linha for linha in linhas}
        return [por_id[id_chunk] for id_chunk in ids if id_chunk in por_id]
