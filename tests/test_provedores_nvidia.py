"""Adaptadores de embedding e reranking NVIDIA: contratos, classificação de
falhas e fallback listwise controlado. Tudo offline, com clientes stub."""

import socket
import subprocess
from types import SimpleNamespace

import pytest

from tests.conftest import EmbeddingFalso, RerankFalso
from radar.provedores import (
    ErroProvedorEmbedding,
    _descrever_falha_de_provedor,
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


def test_cliente_embedding_e_tardio_e_normaliza_falha_no_primeiro_uso(monkeypatch):
    chamadas = 0

    def falhar(**_kwargs):
        nonlocal chamadas
        chamadas += 1
        raise TimeoutError()

    monkeypatch.setattr("radar.provedores.NVIDIAEmbeddings", falhar)
    provedor = ProvedorEmbeddingNvidia(api_key="segredo-falso")
    assert chamadas == 0

    with pytest.raises(ErroProvedorEmbedding) as erro:
        provedor.embutir_consulta("consulta")
    assert chamadas == 1
    assert erro.value.operacional is True


def test_cliente_rerank_e_tardio_e_normaliza_falha_no_primeiro_uso(monkeypatch):
    chamadas = 0

    def falhar(**_kwargs):
        nonlocal chamadas
        chamadas += 1
        raise TimeoutError()

    monkeypatch.setattr("radar.provedores.NVIDIARerank", falhar)
    provedor = ProvedorRerankNvidia(api_key="segredo-falso")
    assert chamadas == 0

    with pytest.raises(ErroProvedorRerank) as erro:
        provedor.reordenar("consulta", ["texto"])
    assert chamadas == 1
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


# ----------------------------------------------------------------------
# Falha de terceiro nunca vira texto do projeto
# ----------------------------------------------------------------------
#
# A fronteira LLM (``ProvedorComFallback``) já extraía só classe e código
# seguro. Os adaptadores NVIDIA faziam o oposto: interpolavam ``str(excecao)``
# inteira e encadeavam a original com ``from excecao``, deixando corpo de
# resposta, cabeçalho e credencial alcançáveis por mensagem e por traceback.

import logging
import traceback

CHAVE_FALSA = "nvapi-0000FALSA1111CHAVE2222NAOPODEVAZAR"
CABECALHO_FALSO = "Authorization: Bearer sk-0000FALSO1111TOKEN2222"
CORPO_FALSO = '{"error":{"message":"quota exceeded","request_id":"req_FALSO_9999"}}'
FRAGMENTO_PROMPT = "trecho do prompt: a startup usa modelos de linguagem"
URL_FALSA = "https://integrate.api.nvidia.com/v1/embeddings?api_key=" + CHAVE_FALSA

VALORES_PROIBIDOS = (
    CHAVE_FALSA,
    CABECALHO_FALSO,
    CORPO_FALSO,
    FRAGMENTO_PROMPT,
    URL_FALSA,
    "quota exceeded",
    "req_FALSO_9999",
    "Bearer",
)


class ErroDeTerceiroHostil(Exception):
    """Exceção de biblioteca externa carregando tudo o que não pode vazar."""

    def __init__(self):
        super().__init__(
            f"HTTP 429 em {URL_FALSA} | {CABECALHO_FALSO} | corpo={CORPO_FALSO} "
            f"| {FRAGMENTO_PROMPT}"
        )
        self.status_code = 429
        self.response = SimpleNamespace(status_code=429, text=CORPO_FALSO)


def _texto_exposto(excecao: BaseException) -> str:
    """Mensagem do projeto mais o traceback encadeado que o operador veria."""
    return str(excecao) + "".join(
        traceback.format_exception(type(excecao), excecao, excecao.__traceback__)
    )


def _assertar_sem_vazamento(texto: str) -> None:
    for proibido in VALORES_PROIBIDOS:
        assert proibido not in texto, f"vazou {proibido!r}"


def test_embedding_de_passagem_nao_vaza_corpo_de_terceiro():
    stub = ClienteEmbeddingStub(erro=ErroDeTerceiroHostil())
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)

    with pytest.raises(ErroProvedorEmbedding) as capturado:
        provedor.embutir_passagens(["texto"])

    _assertar_sem_vazamento(_texto_exposto(capturado.value))
    assert capturado.value.operacional is True
    assert "429" in str(capturado.value)


def test_embedding_de_consulta_nao_vaza_corpo_de_terceiro():
    stub = ClienteEmbeddingStub(erro=ErroDeTerceiroHostil())
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)

    with pytest.raises(ErroProvedorEmbedding) as capturado:
        provedor.embutir_consulta("texto")

    _assertar_sem_vazamento(_texto_exposto(capturado.value))


def test_reranker_nvidia_nao_vaza_corpo_de_terceiro():
    class ClienteRerankHostil:
        top_n = 0

        def compress_documents(self, documentos, consulta):
            raise ErroDeTerceiroHostil()

    provedor = ProvedorRerankNvidia(cliente=ClienteRerankHostil())

    with pytest.raises(ErroProvedorRerank) as capturado:
        provedor.reordenar("consulta", ["a", "b"])

    _assertar_sem_vazamento(_texto_exposto(capturado.value))
    assert capturado.value.operacional is True


def test_contrato_violado_do_reranker_nao_vaza_metadado_de_terceiro():
    class ClienteRerankSemScore:
        top_n = 0

        def compress_documents(self, documentos, consulta):
            return [
                SimpleNamespace(metadata={"indice": 0, "segredo": CHAVE_FALSA}),
            ]

    provedor = ProvedorRerankNvidia(cliente=ClienteRerankSemScore())

    with pytest.raises(ErroProvedorRerank) as capturado:
        provedor.reordenar("consulta", ["a"])

    _assertar_sem_vazamento(_texto_exposto(capturado.value))
    # falha de contrato não é indisponibilidade: a distinção precisa sobreviver
    assert capturado.value.operacional is False


def test_a_causa_encadeada_nao_reaparece_no_traceback_comum():
    stub = ClienteEmbeddingStub(erro=ErroDeTerceiroHostil())
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)

    with pytest.raises(ErroProvedorEmbedding) as capturado:
        provedor.embutir_consulta("texto")

    excecao = capturado.value
    assert excecao.__cause__ is None
    assert excecao.__suppress_context__ is True


def test_o_boundary_nao_registra_corpo_de_terceiro_em_log(caplog):
    stub = ClienteEmbeddingStub(erro=ErroDeTerceiroHostil())
    provedor = ProvedorEmbeddingNvidia(dimensao=4, cliente=stub)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ErroProvedorEmbedding):
            provedor.embutir_consulta("texto")

    _assertar_sem_vazamento(caplog.text)


def test_a_saida_do_comando_de_ingestao_nao_vaza_corpo_de_terceiro(
    capsys, monkeypatch
):
    """O caminho real do script, com o provedor falhando de forma hostil.

    Exercita ``ingerir_completo`` de verdade: o ``print`` da linha de falha é o
    ponto que levava a mensagem do adaptador ao console do operador.
    """
    from scripts import ingerir_conhecimento

    def provedor_hostil(*_args, **_kwargs):
        raise ErroProvedorEmbedding(
            _descrever_falha_de_provedor(
                "falha ao inicializar o provedor de embedding", ErroDeTerceiroHostil()
            ),
            operacional=True,
        ) from None

    # sem ler o .env real e sem tocar a rede
    monkeypatch.setattr(ingerir_conhecimento, "load_dotenv", lambda *_a, **_k: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste-nao-usada")
    monkeypatch.setattr(
        ingerir_conhecimento, "ProvedorEmbeddingNvidia", provedor_hostil
    )

    codigo = ingerir_conhecimento.ingerir_completo()

    assert codigo == 2
    saida = capsys.readouterr().out
    assert "FALHA NA INGEST" in saida
    _assertar_sem_vazamento(saida)
