"""Contratos do RAG NVIDIA: enums, fontes, chunks, trechos e contexto.

Todos os testes são offline e não dependem de provedor algum.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from radar.configuracao import (
    DIMENSAO_EMBEDDING_NVIDIA,
    K_LEXICAL_NVIDIA,
    K_RRF,
    K_VETORIAL_NVIDIA,
    N_CANDIDATOS_RERANK,
    N_TRECHOS_FINAL,
    TETO_CARACTERES_CHUNK,
)
from radar.contratos import (
    ChunkNvidia,
    ContextoNvidia,
    FonteNvidia,
    TECNOLOGIAS_NVIDIA,
    TrechoNvidia,
)


def fonte_valida(**alteracoes) -> dict:
    dados = {
        "topico": "nim",
        "origem": "tecnologia",
        "tecnologia": "NVIDIA NIM",
        "fonte_url": "https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/",
        "titulo": "NVIDIA NIM",
        "data_acesso": date(2026, 8, 25),
    }
    dados.update(alteracoes)
    return {chave: valor for chave, valor in dados.items() if valor is not ...}


def chunk_valido(**alteracoes) -> dict:
    dados = {
        "topico": "nim",
        "origem": "tecnologia",
        "tecnologia": "NVIDIA NIM",
        "fonte_url": "https://exemplo.nvidia.com/nim",
        "breadcrumb": "NVIDIA NIM > Deploy",
        "texto_limpo": "NIM empacota modelos como microservices otimizados.",
        "indice_parte": 1,
        "hash_texto": "a" * 64,
    }
    dados.update(alteracoes)
    return dados


def trecho_valido(**alteracoes) -> dict:
    dados = {
        "id_chunk": 7,
        "topico": "nim",
        "origem": "tecnologia",
        "tecnologia": "NVIDIA NIM",
        "breadcrumb": "NVIDIA NIM > Deploy",
        "texto": "NIM empacota modelos como microservices otimizados.",
        "fonte_url": "https://exemplo.nvidia.com/nim",
        "score_rerank": 0.87,
    }
    dados.update(alteracoes)
    return dados


def test_enum_contem_exatamente_as_dezesseis_tecnologias_do_tapi():
    esperadas = {
        "NVIDIA Inception",
        "NVIDIA NIM",
        "NVIDIA NeMo",
        "NeMo Guardrails",
        "NVIDIA Triton Inference Server",
        "TensorRT-LLM",
        "NVIDIA RAPIDS",
        "cuDF",
        "cuML",
        "CUDA",
        "NVIDIA Riva",
        "NVIDIA Omniverse",
        "NVIDIA Isaac",
        "NVIDIA Clara",
        "NVIDIA Morpheus",
        "NVIDIA AI Enterprise",
    }
    assert set(TECNOLOGIAS_NVIDIA) == esperadas
    assert len(TECNOLOGIAS_NVIDIA) == 16


def test_fonte_de_tecnologia_exige_tecnologia_do_enum():
    fonte = FonteNvidia(**fonte_valida())
    assert fonte.tecnologia == "NVIDIA NIM"
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(tecnologia=None))
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(tecnologia="NVIDIA Inexistente"))


def test_fonte_conceitual_proibe_tecnologia():
    fonte = FonteNvidia(
        **fonte_valida(origem="conceitual", tecnologia=None, topico="ai_native_services")
    )
    assert fonte.tecnologia is None
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(origem="conceitual"))


def test_fonte_rejeita_url_invalida_e_origem_desconhecida():
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(fonte_url="nao-e-uma-url"))
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(origem="marketing"))


def test_chunk_aplica_a_mesma_regra_de_origem_e_tecnologia():
    chunk = ChunkNvidia(**chunk_valido())
    assert chunk.indice_parte == 1
    with pytest.raises(ValidationError):
        ChunkNvidia(**chunk_valido(origem="conceitual"))
    with pytest.raises(ValidationError):
        ChunkNvidia(**chunk_valido(tecnologia=None))


def test_chunk_rejeita_texto_vazio_e_parte_invalida():
    with pytest.raises(ValidationError):
        ChunkNvidia(**chunk_valido(texto_limpo=""))
    with pytest.raises(ValidationError):
        ChunkNvidia(**chunk_valido(indice_parte=0))


def test_trecho_preserva_rastreabilidade_e_regra_de_origem():
    trecho = TrechoNvidia(**trecho_valido())
    assert trecho.id_chunk == 7
    assert str(trecho.fonte_url) == "https://exemplo.nvidia.com/nim"
    conceitual = TrechoNvidia(
        **trecho_valido(origem="conceitual", tecnologia=None, topico="ai_native_services")
    )
    assert conceitual.tecnologia is None
    with pytest.raises(ValidationError):
        TrechoNvidia(**trecho_valido(origem="conceitual"))


def test_contexto_exige_entre_cinco_e_oito_trechos():
    def contexto_com(n: int) -> ContextoNvidia:
        return ContextoNvidia(
            consulta_gerada="latência de inferência em voz",
            trechos=[TrechoNvidia(**trecho_valido(id_chunk=i)) for i in range(n)],
        )

    assert len(contexto_com(5).trechos) == 5
    assert len(contexto_com(8).trechos) == 8
    with pytest.raises(ValidationError):
        contexto_com(4)
    with pytest.raises(ValidationError):
        contexto_com(9)
    with pytest.raises(ValidationError):
        ContextoNvidia(consulta_gerada="", trechos=contexto_com(5).trechos)


def test_contratos_rejeitam_campos_extras():
    with pytest.raises(ValidationError):
        FonteNvidia(**fonte_valida(), inesperado=1)
    with pytest.raises(ValidationError):
        ChunkNvidia(**chunk_valido(), inesperado=1)
    with pytest.raises(ValidationError):
        TrechoNvidia(**trecho_valido(), inesperado=1)


def test_constantes_iniciais_do_pipeline():
    assert K_LEXICAL_NVIDIA == 20
    assert K_VETORIAL_NVIDIA == 20
    assert K_RRF == 60
    assert N_CANDIDATOS_RERANK == 20
    assert N_TRECHOS_FINAL == 6
    assert 5 <= N_TRECHOS_FINAL <= 8
    assert TETO_CARACTERES_CHUNK == 1800
    assert DIMENSAO_EMBEDDING_NVIDIA == 2048
