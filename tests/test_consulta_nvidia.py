"""Recuperação híbrida da KB NVIDIA: lexical, vetorial, RRF, reranking e o
boundary direto ``consultar`` com rastreabilidade de citação. Tudo offline."""

import pytest

from tests.conftest import EmbeddingFalso, RerankFalso
from radar.configuracao import (
    K_LEXICAL_NVIDIA,
    K_VETORIAL_NVIDIA,
    N_CANDIDATOS_RERANK,
    N_TRECHOS_FINAL,
)
from radar.contratos import ContextoNvidia
from radar.conhecimento_nvidia.consulta import (
    ConhecimentoNvidia,
    ErroConsultaNvidia,
    buscar_lexical,
    buscar_vetorial,
    fusao_rrf,
)
from radar.conhecimento_nvidia.ingestao import conectar_conhecimento, ingerir
from radar.provedores import ErroProvedorRerank
from tests.test_ingestao_nvidia import escrever_fonte


CORPO_TRITON = """# Visão geral

O Triton Inference Server serve modelos de IA em produção com baixa latência.

# Batching dinâmico

O batching dinâmico agrupa requisições para elevar o throughput na GPU.

# Model Analyzer

O Model Analyzer explora configurações para otimizar o serving de modelos.
"""

CORPO_RIVA = """# Visão geral

Riva oferece reconhecimento de fala e síntese de voz com modelos otimizados.

# Casos de uso

Call centers usam Riva para transcrição de chamadas e atendimento por voz.
"""

CORPO_RAPIDS = """# Visão geral

RAPIDS acelera pipelines de dados com dataframes processados na GPU.
"""

CORPO_CONCEITO = """# Serviços AI-native

Empresas AI-native vendem resultado operacional combinando software e agentes.

# Wrappers de LLM

Depender somente de wrappers de API externa cria risco de substituição.
"""


class RerankConstante:
    def __init__(self, valor: float = 0.5):
        self.valor = valor

    def reordenar(self, consulta, textos):
        return [self.valor] * len(textos)


class RerankExplosivo:
    def reordenar(self, consulta, textos):
        raise ErroProvedorRerank("indisponível nos dois provedores", operacional=True)


class RerankComQuantidadeErrada:
    def reordenar(self, consulta, textos):
        return [1.0]


@pytest.fixture
def caminho_kb(tmp_path):
    diretorio = tmp_path / "fontes"
    diretorio.mkdir()
    escrever_fonte(
        diretorio, "01_triton.md", topico="triton", origem="tecnologia",
        tecnologia="NVIDIA Triton Inference Server",
        url="https://exemplo.nvidia.com/triton",
        titulo="NVIDIA Triton Inference Server", corpo=CORPO_TRITON,
    )
    escrever_fonte(
        diretorio, "02_riva.md", topico="riva", origem="tecnologia",
        tecnologia="NVIDIA Riva", url="https://exemplo.nvidia.com/riva",
        titulo="NVIDIA Riva", corpo=CORPO_RIVA,
    )
    escrever_fonte(
        diretorio, "03_rapids.md", topico="rapids", origem="tecnologia",
        tecnologia="NVIDIA RAPIDS", url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS", corpo=CORPO_RAPIDS,
    )
    escrever_fonte(
        diretorio, "04_conceito.md", topico="ai_native_services", origem="conceitual",
        tecnologia=None, url="https://exemplo.com/ai-native",
        titulo="Serviços AI-native", corpo=CORPO_CONCEITO,
    )
    caminho = tmp_path / "kb.db"
    ingerir(caminho, diretorio, EmbeddingFalso(), exigir_cobertura=False)
    return caminho


def id_por_breadcrumb(conexao, trecho_breadcrumb: str) -> int:
    return conexao.execute(
        "SELECT id FROM chunks_nvidia WHERE breadcrumb LIKE ?",
        (f"%{trecho_breadcrumb}%",),
    ).fetchone()["id"]


def test_busca_lexical_ordena_por_bm25_e_ignora_consulta_sem_acerto(caminho_kb):
    conexao = conectar_conhecimento(caminho_kb)
    ids = buscar_lexical(conexao, "batching dinâmico de requisições", K_LEXICAL_NVIDIA)
    assert ids
    assert ids[0] == id_por_breadcrumb(conexao, "Batching dinâmico")
    assert buscar_lexical(conexao, "xyzabc inexistente", K_LEXICAL_NVIDIA) == []
    assert buscar_lexical(conexao, "!!! ???", K_LEXICAL_NVIDIA) == []
    conexao.close()


def test_busca_vetorial_recupera_o_chunk_mais_proximo(caminho_kb):
    conexao = conectar_conhecimento(caminho_kb)
    embedding = EmbeddingFalso()
    vetor = embedding.embutir_consulta(
        "batching dinâmico agrupa requisições throughput GPU"
    )
    ids = buscar_vetorial(conexao, vetor, K_VETORIAL_NVIDIA)
    assert len(ids) == 8  # todos os chunks do corpus, ordenados por distância
    assert ids[0] == id_por_breadcrumb(conexao, "Batching dinâmico")
    conexao.close()


def test_fusao_rrf_prioriza_presenca_nas_duas_listas():
    assert fusao_rrf([[1, 2, 3], [2, 4]], k_rrf=60) == [2, 1, 4, 3]


def test_fusao_rrf_desempata_por_id():
    # 5 e 7 têm o mesmo score (mesma posição em listas distintas).
    assert fusao_rrf([[7], [5]], k_rrf=60) == [5, 7]
    assert fusao_rrf([[5], [7]], k_rrf=60) == [5, 7]


def test_consultar_devolve_contexto_valido_com_rastreabilidade(caminho_kb):
    kb = ConhecimentoNvidia(caminho_kb, EmbeddingFalso(), RerankFalso())
    consulta = "latência de inferência para servir modelos em produção"
    contexto = kb.consultar(consulta)
    assert isinstance(contexto, ContextoNvidia)
    assert contexto.consulta_gerada == consulta
    assert len(contexto.trechos) == N_TRECHOS_FINAL
    scores = [trecho.score_rerank for trecho in contexto.trechos]
    assert scores == sorted(scores, reverse=True)

    conexao = conectar_conhecimento(caminho_kb)
    for trecho in contexto.trechos:
        linha = conexao.execute(
            "SELECT breadcrumb, texto_limpo, fonte_url, topico, origem, tecnologia "
            "FROM chunks_nvidia WHERE id = ?",
            (trecho.id_chunk,),
        ).fetchone()
        assert linha is not None
        assert trecho.breadcrumb == linha["breadcrumb"]
        assert trecho.texto == linha["texto_limpo"]
        assert str(trecho.fonte_url) == linha["fonte_url"]
        assert trecho.origem == linha["origem"]
        assert trecho.tecnologia == linha["tecnologia"]
    conexao.close()


def test_consultar_e_deterministico(caminho_kb):
    kb = ConhecimentoNvidia(caminho_kb, EmbeddingFalso(), RerankFalso())
    primeira = kb.consultar("dados em GPU com dataframes")
    segunda = kb.consultar("dados em GPU com dataframes")
    assert primeira == segunda


def test_rerank_recebe_no_maximo_o_teto_de_candidatos_com_breadcrumb(caminho_kb):
    rerank = RerankFalso()
    kb = ConhecimentoNvidia(caminho_kb, EmbeddingFalso(), rerank)
    kb.consultar("voz e transcrição de chamadas")
    assert rerank.ultimo_lote is not None
    assert len(rerank.ultimo_lote) <= N_CANDIDATOS_RERANK
    # O reranker pontua breadcrumb + corpo, o mesmo composto do embedding.
    assert all("\n\n" in texto for texto in rerank.ultimo_lote)


def test_scores_iguais_no_rerank_preservam_a_ordem_da_fusao(caminho_kb):
    consulta = "modelos de IA em produção"
    conexao = conectar_conhecimento(caminho_kb)
    embedding = EmbeddingFalso()
    fundidos = fusao_rrf(
        [
            buscar_lexical(conexao, consulta, K_LEXICAL_NVIDIA),
            buscar_vetorial(
                conexao, embedding.embutir_consulta(consulta), K_VETORIAL_NVIDIA
            ),
        ],
        k_rrf=60,
    )
    conexao.close()
    esperados = fundidos[:N_CANDIDATOS_RERANK][:N_TRECHOS_FINAL]

    kb = ConhecimentoNvidia(caminho_kb, EmbeddingFalso(), RerankConstante())
    contexto = kb.consultar(consulta)
    assert [trecho.id_chunk for trecho in contexto.trechos] == esperados


def test_falha_do_rerank_propaga_sem_devolver_ordem_da_fusao(caminho_kb):
    kb = ConhecimentoNvidia(caminho_kb, EmbeddingFalso(), RerankExplosivo())
    with pytest.raises(ErroProvedorRerank):
        kb.consultar("qualquer consulta")


def test_consulta_rejeita_modelo_diferente_do_indice_antes_de_embutir(caminho_kb):
    embedding = EmbeddingFalso(modelo="outro-modelo")
    kb = ConhecimentoNvidia(caminho_kb, embedding, RerankFalso())
    with pytest.raises(ErroConsultaNvidia, match="modelo de embedding"):
        kb.consultar("modelos em produção")
    assert embedding.chamadas_consulta == 0


def test_consulta_rejeita_quantidade_invalida_de_scores(caminho_kb):
    kb = ConhecimentoNvidia(
        caminho_kb, EmbeddingFalso(), RerankComQuantidadeErrada()
    )
    with pytest.raises(ErroConsultaNvidia, match="scores"):
        kb.consultar("modelos em produção")


def test_consulta_falha_com_erro_claro_quando_ha_menos_de_cinco_chunks(tmp_path):
    diretorio = tmp_path / "fontes"
    diretorio.mkdir()
    escrever_fonte(
        diretorio,
        "01_rapids.md",
        topico="rapids",
        origem="tecnologia",
        tecnologia="NVIDIA RAPIDS",
        url="https://exemplo.nvidia.com/rapids",
        titulo="NVIDIA RAPIDS",
        corpo="# Visão geral\n\nRAPIDS acelera dados na GPU.",
    )
    caminho = tmp_path / "kb_pequena.db"
    ingerir(caminho, diretorio, EmbeddingFalso(), exigir_cobertura=False)
    kb = ConhecimentoNvidia(caminho, EmbeddingFalso(), RerankFalso())
    with pytest.raises(ErroConsultaNvidia, match="ao menos 5"):
        kb.consultar("dados na GPU")
