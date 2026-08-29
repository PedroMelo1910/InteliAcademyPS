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

