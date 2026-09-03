import hashlib
import asyncio
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

from radar.base_startups import BaseStartups, inicializar_banco
from radar.configuracao import CAMINHO_DADOS_CURADOS


@pytest.fixture(autouse=True)
def bloquear_rede(monkeypatch):
    """Nenhum teste automatizado pode acessar a rede; falha alta se tentar."""

    def negar(*_args, **_kwargs):
        raise RuntimeError(
            "teste tentou abrir uma conexão de rede; a suíte é estritamente offline"
        )

    monkeypatch.setattr(socket.socket, "connect", negar)
    monkeypatch.setattr(socket.socket, "connect_ex", negar)
    monkeypatch.setattr(socket, "create_connection", negar)
    monkeypatch.setattr(socket, "getaddrinfo", negar)
    monkeypatch.setattr(socket, "gethostbyname", negar)
    monkeypatch.setattr(socket, "gethostbyname_ex", negar)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", negar)
    monkeypatch.setattr(subprocess, "Popen", negar)
    monkeypatch.setattr(os, "system", negar)


@pytest.fixture
def caminho_banco(tmp_path: Path) -> Path:
    caminho = tmp_path / "radar_teste.db"
    inicializar_banco(caminho, CAMINHO_DADOS_CURADOS)
    return caminho


@pytest.fixture
def base(caminho_banco: Path) -> BaseStartups:
    return BaseStartups(caminho_banco)


def _tokens(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.casefold())


class EmbeddingFalso:
    """EmbeddingProvider determinístico: saco de tokens por hashing, sem rede.

    Textos que compartilham tokens compartilham componentes do vetor, o que
    dá similaridade de cosseno controlável nos testes.
    """

    def __init__(self, dimensao: int = 32, modelo: str | None = None):
        self._dimensao = dimensao
        self._modelo = modelo or f"embedding-falso-{dimensao}"
        self.chamadas_passagens = 0
        self.chamadas_consulta = 0
        self.textos_embedados: list[str] = []

    @property
    def dimensao(self) -> int:
        return self._dimensao

    @property
    def modelo(self) -> str:
        return self._modelo

    def _vetor(self, texto: str) -> list[float]:
        vetor = [0.0] * self._dimensao
        for token in _tokens(texto):
            digerido = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vetor[int(digerido, 16) % self._dimensao] += 1.0
        if not any(vetor):
            vetor[0] = 1.0  # evita vetor nulo, cujo cosseno é indefinido
        return vetor

    def embutir_passagens(self, textos: list[str]) -> list[list[float]]:
        self.chamadas_passagens += 1
        self.textos_embedados.extend(textos)
        return [self._vetor(texto) for texto in textos]

    def embutir_consulta(self, texto: str) -> list[float]:
        self.chamadas_consulta += 1
        return self._vetor(texto)


class RerankFalso:
    """RerankProvider determinístico: sobreposição de tokens com a consulta."""

    def __init__(self):
        self.chamadas = 0
        self.ultimo_lote: list[str] | None = None

    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        self.chamadas += 1
        self.ultimo_lote = list(textos)
        termos = set(_tokens(consulta))
        return [
            len(termos & set(_tokens(texto))) / (len(termos) or 1)
            for texto in textos
        ]


class ConsultorNvidiaFalso:
    """ProvedorContextoNvidia determinístico: nenhuma rede, consulta registrada."""

    def __init__(self, contexto=None, erro: Exception | None = None):
        self._contexto = contexto if contexto is not None else contexto_nvidia_falso()
        self._erro = erro
        self.chamadas = 0
        self.consultas: list[str] = []

    def consultar(self, consulta: str):
        self.chamadas += 1
        self.consultas.append(consulta)
        if self._erro is not None:
            raise self._erro
        return self._contexto


def trecho_nvidia_falso(
    id_chunk: int,
    *,
    tecnologia: str | None = "NVIDIA NIM",
    topico: str | None = None,
    score_rerank: float = 0.9,
):
    """Um ``TrechoNvidia`` coerente: origem e tecnologia sempre combinam."""
    from radar.contratos import TrechoNvidia

    origem = "tecnologia" if tecnologia is not None else "conceitual"
    rotulo = topico or (tecnologia or "ai-native-services")
    return TrechoNvidia(
        id_chunk=id_chunk,
        topico=rotulo,
        origem=origem,
        tecnologia=tecnologia,
        breadcrumb=f"{rotulo} > seção {id_chunk}",
        texto=(
            f"Trecho {id_chunk} da base NVIDIA sobre {rotulo}, com inferência, "
            "latência e custo em produção."
        ),
        fonte_url=f"https://nvidia.example/{id_chunk}",
        score_rerank=score_rerank,
    )


def contexto_nvidia_falso(consulta: str = "consulta NVIDIA de teste"):
    """Contexto com 5 chunks de tecnologia e 1 conceitual, dentro da faixa 5–8."""
    from radar.contratos import ContextoNvidia

    trechos = [
        trecho_nvidia_falso(101, tecnologia="NVIDIA NIM", score_rerank=0.95),
        trecho_nvidia_falso(102, tecnologia="NVIDIA Triton Inference Server", score_rerank=0.9),
        trecho_nvidia_falso(103, tecnologia="TensorRT-LLM", score_rerank=0.85),
        trecho_nvidia_falso(104, tecnologia="NeMo Guardrails", score_rerank=0.8),
        trecho_nvidia_falso(105, tecnologia="NVIDIA RAPIDS", score_rerank=0.75),
        trecho_nvidia_falso(
            106, tecnologia=None, topico="ai-native-services", score_rerank=0.7
        ),
    ]
    return ContextoNvidia(consulta_gerada=consulta, trechos=trechos)


DIMENSOES_ESTRUTURAIS = (
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
)


def afirmacao_validada_falsa(
    id_afirmacao: int,
    categoria: str,
    *,
    polaridade: str | None = None,
    situacao: str = "confirmada",
    id_documento: int | None = None,
    texto: str | None = None,
):
    """``AfirmacaoValidada`` mínima e coerente com as regras de polaridade."""
    from radar.contratos import AfirmacaoValidada

    if polaridade is None:
        polaridade = "neutro" if categoria not in DIMENSOES_ESTRUTURAIS else "presenca"
    return AfirmacaoValidada(
        id_afirmacao=id_afirmacao,
        texto=texto or f"A evidência número {id_afirmacao} está documentada.",
        categoria=categoria,
        polaridade=polaridade,
        id_documento=id_documento if id_documento is not None else id_afirmacao,
        trecho_citado=(
            f"Trecho público verificável para a evidência {id_afirmacao} citada."
        ),
        situacao=situacao,
        motivo=None if situacao == "confirmada" else "Trecho não ocorre na fonte.",
    )


def perfil_validado_falso(itens, hosts: list[str] | None = None):
    """Monta o ``PerfilValidado`` derivando dimensões e taxa das afirmações."""
    from radar.contratos import EstadoDimensaoGap, PerfilValidado

    estados = []
    for dimensao in DIMENSOES_ESTRUTURAIS:
        presencas = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "presenca"
        )
        ausencias = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "ausencia_explicita"
        )
        if presencas and ausencias:
            estado, ids = "desconhecido", sorted(presencas + ausencias)
        elif presencas:
            estado, ids = "capacidade_confirmada", presencas
        elif ausencias:
            estado, ids = "gap_confirmado", ausencias
        else:
            estado, ids = "desconhecido", []
        estados.append(
            EstadoDimensaoGap(dimensao=dimensao, estado=estado, ids_evidencias=ids)
        )
    derrubadas = sum(1 for item in itens if item.situacao == "derrubada")
    return PerfilValidado(
        afirmacoes_validadas=itens,
        taxa_derrubada=derrubadas / len(itens),
        hosts_distintos=sorted(hosts if hosts is not None else ["fonte-a.example"]),
        estado_dimensoes_gap=estados,
    )


class ProvedorSequencialFalso:
    """Provedor de structured output com uma resposta programada por chamada.

    Um item ``Exception`` é levantado em vez de devolvido, o que permite testar
    falha de provedor e falha de contrato com o mesmo fake.
    """

    def __init__(self, *respostas):
        self._respostas = list(respostas)
        self.chamadas = 0
        self.mensagens: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        if not self._respostas:
            raise AssertionError(
                f"o provedor foi chamado {self.chamadas} vezes, além das respostas "
                "programadas"
            )
        resposta = self._respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta
