"""Avaliação leve da recuperação da KB NVIDIA, antes e depois do reranking.

Uso: ``python -m scripts.avaliar_recuperacao``

Para cada caso rotulado à mão em ``dados/avaliacao_rag.json``, compara:
- baseline: top-6 da fusão RRF (lexical + vetorial), sem reranking;
- final:    top-6 devolvido por ``ConhecimentoNvidia.consultar`` (com reranking).

Métricas: hit@6 e posição da primeira ocorrência da tecnologia esperada.
Amostra pequena e original: orienta iteração, sem alegar significância
estatística. Requer NVIDIA_API_KEY válida (embedding e reranker); o fallback
listwise usa GOOGLE_API_KEY somente se o reranker NVIDIA falhar
operacionalmente.
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from radar.configuracao import (
    CAMINHO_BANCO,
    N_TRECHOS_FINAL,
    RAIZ_PROJETO,
)
from radar.conhecimento_nvidia.consulta import (
    ConhecimentoNvidia,
    buscar_lexical,
    buscar_vetorial,
    fusao_rrf,
)
from radar.conhecimento_nvidia.ingestao import conectar_conhecimento
from radar.provedores import (
    ProvedorEmbeddingNvidia,
    ProvedorRerankListwiseGemini,
    ProvedorRerankNvidia,
    RerankComFallback,
)


def _tecnologias_por_id(conexao, ids: list[int]) -> list[str | None]:
    if not ids:
        return []
    marcadores = ", ".join("?" for _ in ids)
    linhas = conexao.execute(
        f"SELECT id, tecnologia FROM chunks_nvidia WHERE id IN ({marcadores})",
        tuple(ids),
    ).fetchall()
    por_id = {linha["id"]: linha["tecnologia"] for linha in linhas}
    return [por_id.get(id_chunk) for id_chunk in ids]


def _posicao(tecnologias: list[str | None], esperada: str) -> int | None:
    for indice, tecnologia in enumerate(tecnologias, start=1):
        if tecnologia == esperada:
            return indice
    return None


def principal() -> int:
    load_dotenv(RAIZ_PROJETO / ".env")
    api_key_nvidia = os.getenv("NVIDIA_API_KEY", "").strip()
    api_key_google = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key_nvidia:
        print("BLOQUEIO OPERACIONAL: NVIDIA_API_KEY ausente no ambiente/.env.")
        return 2

    embedding = ProvedorEmbeddingNvidia(api_key_nvidia)
    rerank_primario = ProvedorRerankNvidia(api_key_nvidia)
    if api_key_google:
        rerank = RerankComFallback(
            rerank_primario, ProvedorRerankListwiseGemini(api_key_google)
        )
    else:
        rerank = rerank_primario
    kb = ConhecimentoNvidia(CAMINHO_BANCO, embedding, rerank)

    casos = json.loads(
        (RAIZ_PROJETO / "dados" / "avaliacao_rag.json").read_text(encoding="utf-8")
    )["casos"]

    hits_antes = hits_depois = 0
    soma_rr_antes = soma_rr_depois = 0.0
    for caso in casos:
        consulta = caso["consulta"]
        esperada = caso["tecnologia_esperada"]

        conexao = conectar_conhecimento(CAMINHO_BANCO)
        try:
            fundidos = fusao_rrf(
                [
                    buscar_lexical(conexao, consulta),
                    buscar_vetorial(conexao, embedding.embutir_consulta(consulta)),
                ]
            )[:N_TRECHOS_FINAL]
            tecnologias_antes = _tecnologias_por_id(conexao, fundidos)
        finally:
            conexao.close()

        contexto = kb.consultar(consulta)
        tecnologias_depois = [trecho.tecnologia for trecho in contexto.trechos]

        pos_antes = _posicao(tecnologias_antes, esperada)
        pos_depois = _posicao(tecnologias_depois, esperada)
        hits_antes += pos_antes is not None
        hits_depois += pos_depois is not None
        soma_rr_antes += 1.0 / pos_antes if pos_antes else 0.0
        soma_rr_depois += 1.0 / pos_depois if pos_depois else 0.0
        print(
            f"[{esperada}] antes: {pos_antes or '-'} | depois: {pos_depois or '-'} | "
            f"{consulta[:60]}"
        )

    total = len(casos)
    print(f"\nhit@{N_TRECHOS_FINAL} antes do reranking:  {hits_antes}/{total}")
    print(f"hit@{N_TRECHOS_FINAL} depois do reranking: {hits_depois}/{total}")
    print(f"MRR antes:  {soma_rr_antes / total:.3f}")
    print(f"MRR depois: {soma_rr_depois / total:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
