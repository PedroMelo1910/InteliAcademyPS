"""Smoke operacional do reranker — roda fora do pytest.

Uso:
    python -m scripts.smoke_rerank_nvidia             # reranker NVIDIA (primário)
    python -m scripts.smoke_rerank_nvidia --fallback  # prova o fallback listwise Gemini

Entrada mínima: 1 consulta e 4 passagens; verifica que a passagem correta
sobe ao topo. Nunca imprime chaves nem grava respostas em arquivo.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from radar.configuracao import MODELO_GEMINI, MODELO_RERANK_NVIDIA, RAIZ_PROJETO
from radar.provedores import (
    ErroProvedorRerank,
    ProvedorRerankListwiseGemini,
    ProvedorRerankNvidia,
)

CONSULTA = "reduzir a latência de inferência ao servir modelos em produção"
PASSAGENS = [
    "O Triton Inference Server serve modelos em produção com batching dinâmico e baixa latência.",
    "Receita de bolo de cenoura com cobertura de chocolate.",
    "O programa Inception oferece créditos de nuvem e comunidade para startups.",
    "RAPIDS acelera dataframes e pipelines de dados na GPU.",
]


def principal() -> int:
    load_dotenv(RAIZ_PROJETO / ".env")
    usar_fallback = "--fallback" in sys.argv[1:]

    if usar_fallback:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            print("BLOQUEIO OPERACIONAL: GOOGLE_API_KEY ausente no ambiente/.env.")
            return 2
        tipo_provedor = ProvedorRerankListwiseGemini
        rotulo = f"fallback listwise ({MODELO_GEMINI})"
    else:
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not api_key:
            print("BLOQUEIO OPERACIONAL: NVIDIA_API_KEY ausente no ambiente/.env.")
            return 2
        tipo_provedor = ProvedorRerankNvidia
        rotulo = f"reranker NVIDIA ({MODELO_RERANK_NVIDIA})"

    inicio = time.perf_counter()
    try:
        provedor = tipo_provedor(api_key)
        scores = provedor.reordenar(CONSULTA, PASSAGENS)
    except ErroProvedorRerank as erro:
        tipo = "operacional" if erro.operacional else "de contrato"
        print(f"FALHA {tipo} no smoke do {rotulo}: {erro}")
        return 2
    duracao = time.perf_counter() - inicio

    melhor = max(range(len(PASSAGENS)), key=lambda indice: scores[indice])
    print(f"provedor: {rotulo}")
    print(f"scores (indice: valor): {[f'{i}: {s:.4f}' for i, s in enumerate(scores)]}")
    print(f"passagem no topo: [{melhor}] {PASSAGENS[melhor][:60]}")
    print(f"latencia: {duracao:.2f}s")
    aprovado = melhor == 0  # a passagem do Triton é a relevante para a consulta
    print("veredito:", "APROVADO" if aprovado else "REPROVADO")
    return 0 if aprovado else 1


if __name__ == "__main__":
    sys.exit(principal())
