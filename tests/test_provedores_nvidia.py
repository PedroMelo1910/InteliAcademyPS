"""Adaptadores de embedding e reranking NVIDIA: contratos, classificação de
falhas e fallback listwise controlado. Tudo offline, com clientes stub."""

import socket
import subprocess
from types import SimpleNamespace

import pytest

from tests.conftest import EmbeddingFalso, RerankFalso
from radar.provedores import (
    ErroProvedorEmbedding,
    ErroProvedorRerank,
    ErroRerankIndisponivel,
    OrdenacaoListwise,
    ProvedorEmbeddingNvidia,
    ProvedorRerankListwiseGemini,
    ProvedorRerankNvidia,
    RerankComFallback,
)


class ClienteEmbeddingStub:
    def __init__(self, documentos=None, consulta=None, erro=None):
        self.documentos = documentos
        self.consulta = consulta
        self.erro = erro
        self.metodos = []

    def embed_documents(self, textos):
        self.metodos.append("embed_documents")
        if self.erro:
            raise self.erro
        return self.documentos

    def embed_query(self, texto):
        self.metodos.append("embed_query")
        if self.erro:
            raise self.erro
        return self.consulta


class ClienteRerankStub:
    def __init__(self, resposta=None, erro=None):
        self.resposta = resposta
        self.erro = erro
        self.chamadas = 0

    def compress_documents(self, documentos, consulta):
        self.chamadas += 1
        if self.erro:
            raise self.erro
        return self.resposta


class ClienteListwiseStub:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0

    def invoke(self, mensagens):
        resposta = self.respostas[self.chamadas]
        self.chamadas += 1
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


class RerankExplosivo:
    def __init__(self, operacional: bool):
        self.operacional = operacional
        self.chamadas = 0

    def reordenar(self, consulta, textos):
        self.chamadas += 1
        raise ErroProvedorRerank("falha simulada", operacional=self.operacional)


def test_a_rede_esta_bloqueada_na_suite():
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("exemplo.com", 443))
    with pytest.raises(RuntimeError, match="offline"):
        socket.socket().connect(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="offline"):
        socket.socket().connect_ex(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="offline"):
        socket.getaddrinfo("exemplo.com", 443)
    with pytest.raises(RuntimeError, match="offline"):
        subprocess.run(["python", "-V"], check=False)


def test_embedding_usa_modos_distintos_para_passagem_e_consulta():
    stub = ClienteEmbeddingStub(
        documentos=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        consulta=[0.5, 0.5, 0.0, 0.0],
    )
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)
    assert provedor.dimensao == 4
    assert provedor.embutir_passagens(["a", "b"]) == stub.documentos
    assert provedor.embutir_consulta("q") == stub.consulta
    assert stub.metodos == ["embed_documents", "embed_query"]


def test_construcao_do_cliente_embedding_normaliza_falha_operacional(monkeypatch):
    def falhar(**_kwargs):
        raise TimeoutError()

    monkeypatch.setattr("radar.provedores.NVIDIAEmbeddings", falhar)
    with pytest.raises(ErroProvedorEmbedding) as erro:
        ProvedorEmbeddingNvidia(api_key="segredo-falso")
    assert erro.value.operacional is True


def test_embedding_classifica_falha_de_rede_como_operacional():
    stub = ClienteEmbeddingStub(erro=Exception("HTTP 429: too many requests"))
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)
    with pytest.raises(ErroProvedorEmbedding) as erro:
        provedor.embutir_passagens(["a"])
    assert erro.value.operacional is True


def test_embedding_classifica_timeout_sem_mensagem_como_operacional():
    stub = ClienteEmbeddingStub(erro=TimeoutError())
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)
    with pytest.raises(ErroProvedorEmbedding) as erro:
        provedor.embutir_passagens(["a"])
    assert erro.value.operacional is True


def test_numero_em_erro_de_contrato_nao_vira_falha_operacional():
    stub = ClienteEmbeddingStub(erro=ValueError("esperava 500 dimensões"))
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)
    with pytest.raises(ErroProvedorEmbedding) as erro:
        provedor.embutir_passagens(["a"])
    assert erro.value.operacional is False


def test_embedding_resposta_invalida_falha_alto_sem_ser_operacional():
    dimensao_errada = ClienteEmbeddingStub(documentos=[[1.0, 2.0]])
    with pytest.raises(ErroProvedorEmbedding) as erro:
        ProvedorEmbeddingNvidia(dimensao=4, cliente=dimensao_errada).embutir_passagens(["a"])
    assert erro.value.operacional is False

    nao_finito = ClienteEmbeddingStub(consulta=[1.0, float("nan"), 0.0, 0.0])
    with pytest.raises(ErroProvedorEmbedding) as erro:
        ProvedorEmbeddingNvidia(dimensao=4, cliente=nao_finito).embutir_consulta("q")
    assert erro.value.operacional is False

    contagem_errada = ClienteEmbeddingStub(documentos=[[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ErroProvedorEmbedding) as erro:
        ProvedorEmbeddingNvidia(dimensao=4, cliente=contagem_errada).embutir_passagens(
            ["a", "b"]
        )
    assert erro.value.operacional is False


def test_rerank_nvidia_alinha_scores_pelos_indices_originais():
    resposta = [
        SimpleNamespace(metadata={"indice": 1, "relevance_score": 0.9}),
        SimpleNamespace(metadata={"indice": 0, "relevance_score": 0.2}),
    ]
    provedor = ProvedorRerankNvidia(cliente=ClienteRerankStub(resposta=resposta))
    assert provedor.reordenar("consulta", ["texto a", "texto b"]) == [0.2, 0.9]


def test_rerank_nvidia_classifica_falhas():
    pagamento = ClienteRerankStub(erro=Exception("402 Payment Required"))
    with pytest.raises(ErroProvedorRerank) as erro:
        ProvedorRerankNvidia(cliente=pagamento).reordenar("q", ["a"])
    assert erro.value.operacional is True

    resposta_incompleta = [SimpleNamespace(metadata={"indice": 0, "relevance_score": 0.5})]
    with pytest.raises(ErroProvedorRerank) as erro:
        ProvedorRerankNvidia(
            cliente=ClienteRerankStub(resposta=resposta_incompleta)
        ).reordenar("q", ["a", "b"])
    assert erro.value.operacional is False


def test_listwise_converte_ordem_em_score_ordinal_deterministico():
    stub = ClienteListwiseStub([OrdenacaoListwise(ordem=[2, 0, 1])])
    provedor = ProvedorRerankListwiseGemini(cliente=stub)
    scores = provedor.reordenar("consulta", ["p0", "p1", "p2"])
    # Score ordinal apenas para satisfazer a ordenação do contrato; não é
    # probabilidade nem confiança do modelo.
    assert scores == pytest.approx([2 / 3, 1 / 3, 1.0])


def test_listwise_retenta_uma_vez_apos_contrato_invalido():
    stub = ClienteListwiseStub(
        [OrdenacaoListwise(ordem=[0, 0, 1]), OrdenacaoListwise(ordem=[1, 2, 0])]
    )
    scores = ProvedorRerankListwiseGemini(cliente=stub).reordenar("q", ["a", "b", "c"])
    assert stub.chamadas == 2
    assert scores == pytest.approx([1 / 3, 1.0, 2 / 3])


def test_listwise_retenta_apos_saida_estruturada_malformada():
    stub = ClienteListwiseStub(
        [{"ordem": "não é lista"}, OrdenacaoListwise(ordem=[1, 2, 0])]
    )
    scores = ProvedorRerankListwiseGemini(cliente=stub).reordenar(
        "q", ["a", "b", "c"]
    )
    assert stub.chamadas == 2
    assert scores == pytest.approx([1 / 3, 1.0, 2 / 3])


def test_listwise_contrato_invalido_persistente_nao_e_operacional():
    stub = ClienteListwiseStub(
        [OrdenacaoListwise(ordem=[0, 0, 1]), OrdenacaoListwise(ordem=[9, 1, 2])]
    )
    with pytest.raises(ErroProvedorRerank) as erro:
        ProvedorRerankListwiseGemini(cliente=stub).reordenar("q", ["a", "b", "c"])
    assert erro.value.operacional is False


def test_listwise_erro_de_rede_e_operacional():
    stub = ClienteListwiseStub([Exception("504 Deadline Exceeded")])
    with pytest.raises(ErroProvedorRerank) as erro:
        ProvedorRerankListwiseGemini(cliente=stub).reordenar("q", ["a"])
    assert erro.value.operacional is True


def test_fallback_aciona_reserva_somente_para_falha_operacional():
    reserva = RerankFalso()
    com_fallback = RerankComFallback(RerankExplosivo(operacional=True), reserva)
    scores = com_fallback.reordenar("triton serving", ["triton serve modelos", "outro"])
    assert reserva.chamadas == 1
    assert scores[0] > scores[1]

    reserva_intocada = RerankFalso()
    com_bug = RerankComFallback(RerankExplosivo(operacional=False), reserva_intocada)
    with pytest.raises(ErroProvedorRerank) as erro:
        com_bug.reordenar("q", ["a"])
    assert not isinstance(erro.value, ErroRerankIndisponivel)
    assert reserva_intocada.chamadas == 0


def test_falha_dupla_propaga_erro_tipado_sem_devolver_ordem_rrf():
    com_fallback = RerankComFallback(
        RerankExplosivo(operacional=True), RerankExplosivo(operacional=True)
    )
    with pytest.raises(ErroRerankIndisponivel):
        com_fallback.reordenar("q", ["a", "b"])


def test_falha_de_contrato_da_reserva_nao_e_disfarcada_como_indisponibilidade():
    com_fallback = RerankComFallback(
        RerankExplosivo(operacional=True), RerankExplosivo(operacional=False)
    )
    with pytest.raises(ErroProvedorRerank) as erro:
        com_fallback.reordenar("q", ["a", "b"])
    assert not isinstance(erro.value, ErroRerankIndisponivel)
    assert erro.value.operacional is False


def test_fakes_sao_deterministicos():
    embedding = EmbeddingFalso(dimensao=16)
    assert embedding.embutir_consulta("triton") == embedding.embutir_consulta("triton")
    rerank = RerankFalso()
    primeira = rerank.reordenar("triton", ["triton serve", "nada"])
    segunda = rerank.reordenar("triton", ["triton serve", "nada"])
    assert primeira == segunda
