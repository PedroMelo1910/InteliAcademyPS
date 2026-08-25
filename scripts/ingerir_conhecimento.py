"""Ingestão da base de conhecimento NVIDIA no ``radar.db``.

Uso:
    python -m scripts.ingerir_conhecimento --validar   # só valida manifesto e chunking (offline)
    python -m scripts.ingerir_conhecimento             # ingestão completa (exige NVIDIA_API_KEY)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from radar.configuracao import (
    CAMINHO_BANCO,
    CAMINHO_FONTES_NVIDIA,
    RAIZ_PROJETO,
    TETO_CARACTERES_CHUNK,
)
from radar.conhecimento_nvidia.chunking import gerar_chunks
from radar.conhecimento_nvidia.fontes import carregar_fontes, validar_cobertura
from radar.conhecimento_nvidia.ingestao import ErroIngestaoNvidia, ingerir
from radar.provedores import ErroProvedorEmbedding, ProvedorEmbeddingNvidia


def validar_offline() -> int:
    fontes = carregar_fontes(CAMINHO_FONTES_NVIDIA)
    validar_cobertura(fontes)
    total_chunks = 0
    for fonte in fontes:
        chunks = gerar_chunks(fonte, TETO_CARACTERES_CHUNK)
        total_chunks += len(chunks)
        origem = fonte.fonte.origem
        rotulo = fonte.fonte.tecnologia or fonte.fonte.topico
        print(f"{fonte.caminho.name}: {len(chunks)} chunks ({origem}: {rotulo})")
    print(f"\nmanifesto válido: {len(fontes)} fontes, {total_chunks} chunks, "
          "cobertura das 16 tecnologias + material conceitual confirmada")
    return 0


def ingerir_completo() -> int:
    load_dotenv(RAIZ_PROJETO / ".env")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("BLOQUEIO OPERACIONAL: NVIDIA_API_KEY ausente no ambiente/.env.")
        return 2
    try:
        provedor = ProvedorEmbeddingNvidia(api_key)
        resumo = ingerir(CAMINHO_BANCO, CAMINHO_FONTES_NVIDIA, provedor)
    except (ErroProvedorEmbedding, ErroIngestaoNvidia) as erro:
        print(f"FALHA NA INGESTÃO NVIDIA: {erro}")
        return 2
    print(f"fontes: {resumo.fontes} | chunks: {resumo.chunks_totais}")
    print(
        f"inseridos: {resumo.chunks_inseridos} | atualizados: {resumo.chunks_atualizados}"
        f" | removidos: {resumo.chunks_removidos} | inalterados: {resumo.chunks_inalterados}"
    )
    print(
        f"embeddings calculados: {resumo.embeddings_calculados} em "
        f"{resumo.chamadas_embedding} chamadas"
    )
    print(f"base de conhecimento gravada em: {CAMINHO_BANCO}")
    return 0


if __name__ == "__main__":
    if "--validar" in sys.argv[1:]:
        sys.exit(validar_offline())
    sys.exit(ingerir_completo())
