"""O apoio da tela tem um nome só, um lugar só e um ponto de entrada só.

Depois da renomeação de ``apresentacao`` para ``interface``, três coisas
precisam continuar verdadeiras ao mesmo tempo: o pacote novo existe inteiro, o
antigo não sobrevive como cópia de compatibilidade, e nenhum segundo frontend
apareceu ao lado do ``app.py``. Um teste que só olhasse o pacote novo deixaria
passar exatamente o erro mais provável de uma renomeação — a duplicata
esquecida.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from radar.configuracao import RAIZ_PROJETO


# Escrito por concatenação de propósito: o nome antigo não pode aparecer como
# literal aqui, senão a varredura do próprio repositório encontraria a si mesma.
PACOTE_ANTIGO = "radar." + "apresentacao"
DIRETORIO_ANTIGO = "apresentacao"

# Mesmo motivo: a chamada que marca o ponto de entrada não pode existir
# inteira neste arquivo, ou a varredura contaria o próprio teste.
MARCADOR_ENTRADA = "st.set_page" + "_config"

MODULOS_DE_APOIO = ("estado", "exportacao", "mensagens", "rotulos", "tema", "texto")

SUPERFICIE_PUBLICA = (
    "ResumoCandidata",
    "exportar_briefing_markdown",
    "nome_arquivo_briefing",
    "resumir_candidata",
    "resumir_ranking",
)

IGNORADOS = (".venv", "__pycache__", ".git", ".ruff_cache", ".pytest_cache")


def fontes_do_projeto() -> list[Path]:
    """Todo ``.py`` versionável do repositório, sem ambiente nem cache."""
    return [
        caminho
        for caminho in RAIZ_PROJETO.rglob("*.py")
        if not any(parte in IGNORADOS for parte in caminho.parts)
    ]


# ----------------------------------------------------------------------
# O pacote novo existe inteiro
# ----------------------------------------------------------------------


def test_o_pacote_de_interface_expoe_a_superficie_publica():
    pacote = importlib.import_module("radar.interface")

    for nome in SUPERFICIE_PUBLICA:
        assert hasattr(pacote, nome), f"faltou {nome} em radar.interface"


@pytest.mark.parametrize("modulo", MODULOS_DE_APOIO)
def test_cada_modulo_de_apoio_vive_sob_radar_interface(modulo):
    importlib.import_module(f"radar.interface.{modulo}")


# ----------------------------------------------------------------------
# O pacote antigo não sobrevive
# ----------------------------------------------------------------------


def test_o_pacote_antigo_nao_e_mais_importavel():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(PACOTE_ANTIGO)


def test_o_diretorio_antigo_nao_ficou_para_tras_nem_vazio():
    assert not (RAIZ_PROJETO / "radar" / DIRETORIO_ANTIGO).exists()


def test_nenhuma_fonte_do_projeto_ainda_cita_o_pacote_antigo():
    citam = [
        caminho.relative_to(RAIZ_PROJETO).as_posix()
        for caminho in fontes_do_projeto()
        if PACOTE_ANTIGO in caminho.read_text(encoding="utf-8")
    ]

    assert citam == [], f"ainda importam o pacote antigo: {citam}"


# ----------------------------------------------------------------------
# Um frontend só, um ponto de entrada só
# ----------------------------------------------------------------------


def test_existe_exatamente_um_ponto_de_entrada_streamlit():
    entradas = sorted(
        caminho.relative_to(RAIZ_PROJETO).as_posix()
        for caminho in fontes_do_projeto()
        if MARCADOR_ENTRADA in caminho.read_text(encoding="utf-8")
    )

    assert entradas == ["app.py"]


@pytest.mark.parametrize(
    "concorrente",
    [
        "frontend",
        "pages",
        "streamlit_app.py",
        "main.py",
        "package.json",
        "vite.config.ts",
        "next.config.js",
    ],
)
def test_nao_nasceu_um_segundo_frontend_ao_lado_do_app(concorrente):
    assert not (RAIZ_PROJETO / concorrente).exists()


# ----------------------------------------------------------------------
# O pacote é apoio de frontend, não camada de agente
# ----------------------------------------------------------------------


def modulos_importados(caminho: Path) -> set[str]:
    arvore = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes.update(alias.name for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module is not None:
            nomes.add(no.module)
    return nomes


def arquivos_da_interface() -> list[Path]:
    return sorted((RAIZ_PROJETO / "radar" / "interface").glob("*.py"))


def test_o_apoio_da_interface_nunca_importa_streamlit():
    """A jornada continua testável sem widget: a tela orquestra, o pacote não."""
    for caminho in arquivos_da_interface():
        importados = modulos_importados(caminho)
        assert not any(nome.startswith("streamlit") for nome in importados), (
            f"{caminho.name} importa streamlit"
        )


def test_o_apoio_da_interface_nao_alcanca_agente_grafo_nem_banco():
    proibidos = ("radar.agentes", "radar.grafo", "radar.provedores", "sqlite3")

    for caminho in arquivos_da_interface():
        importados = modulos_importados(caminho)
        for proibido in proibidos:
            assert not any(
                nome == proibido or nome.startswith(proibido + ".")
                for nome in importados
            ), f"{caminho.name} importa {proibido}"
