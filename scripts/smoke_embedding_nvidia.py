"""Smoke operacional do embedding NVIDIA — roda fora do pytest.

Uso: ``python scripts/smoke_embedding_nvidia.py``

Faz duas chamadas mínimas (passagens + consulta), valida dimensão e valores
finitos e verifica que a passagem semanticamente próxima vence no cosseno.
Nunca imprime a chave nem grava resposta do provedor em arquivo.
"""

from __future__ import annotations

import math
import os
import sys
import time

from dotenv import load_dotenv

from radar.configuracao import (
    DIMENSAO_EMBEDDING_NVIDIA,
    MODELO_EMBEDDING_NVIDIA,
    RAIZ_PROJETO,
)
from radar.provedores import ErroProvedorEmbedding, ProvedorEmbeddingNvidia


def _cosseno(a: list[float], b: list[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    return produto / (norma_a * norma_b)


def principal() -> int:
    load_dotenv(RAIZ_PROJETO / ".env")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("BLOQUEIO OPERACIONAL: NVIDIA_API_KEY ausente no ambiente/.env.")
        return 2

    passagens = [
        "NVIDIA Triton Inference Server serve modelos de IA em produção.",
        "Receita de bolo de cenoura com cobertura de chocolate.",
    ]
    consulta = "como servir modelos de inteligência artificial em produção"

    inicio = time.perf_counter()
    try:
        provedor = ProvedorEmbeddingNvidia(api_key)
        vetores = provedor.embutir_passagens(passagens)
        vetor_consulta = provedor.embutir_consulta(consulta)
    except ErroProvedorEmbedding as erro:
        tipo = "operacional" if erro.operacional else "de contrato"
        print(f"FALHA {tipo} no smoke de embedding: {erro}")
        return 2
    duracao = time.perf_counter() - inicio

    tamanhos = {len(vetor) for vetor in [*vetores, vetor_consulta]}
    finitos = all(
        math.isfinite(valor) for vetor in [*vetores, vetor_consulta] for valor in vetor
    )
    relevante = _cosseno(vetor_consulta, vetores[0])
    irrelevante = _cosseno(vetor_consulta, vetores[1])

    print(f"modelo: {MODELO_EMBEDDING_NVIDIA}")
    print(f"dimensoes retornadas: {sorted(tamanhos)} (esperado {DIMENSAO_EMBEDDING_NVIDIA})")
    print(f"valores finitos: {'sim' if finitos else 'NAO'}")
    print(f"cosseno consulta x passagem relevante:   {relevante:.4f}")
    print(f"cosseno consulta x passagem irrelevante: {irrelevante:.4f}")
    print(f"latencia total (2 chamadas, 3 vetores): {duracao:.2f}s")
    aprovado = tamanhos == {DIMENSAO_EMBEDDING_NVIDIA} and finitos and relevante > irrelevante
    print("veredito:", "APROVADO" if aprovado else "REPROVADO")
    return 0 if aprovado else 1


if __name__ == "__main__":
    sys.exit(principal())
