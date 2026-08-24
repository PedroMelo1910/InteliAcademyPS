from pathlib import Path

import pytest

from radar.base_startups import BaseStartups, inicializar_banco
from radar.configuracao import CAMINHO_DADOS_CURADOS


@pytest.fixture
def caminho_banco(tmp_path: Path) -> Path:
    caminho = tmp_path / "radar_teste.db"
    inicializar_banco(caminho, CAMINHO_DADOS_CURADOS)
    return caminho


@pytest.fixture
def base(caminho_banco: Path) -> BaseStartups:
    return BaseStartups(caminho_banco)

