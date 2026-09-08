"""Fronteira da curadoria real: o que dá para provar sem fabricar semântica.

Este arquivo NÃO avalia qualidade de análise, extração ou classificação —
nenhum provedor é chamado e nenhum perfil é inventado. Ele prova quatro coisas
verificáveis sobre os 30 registros curados: que carregam, que são recuperáveis
pela fronteira SQLite, que uma startup nunca recebe documento de outra, e que o
rótulo de curadoria não vaza para o texto que iria a um provedor.
"""

from __future__ import annotations

import sqlite3

import pytest

from radar.agentes.extractor import Extractor
from radar.base_startups import BaseStartups, inicializar_banco
from radar.configuracao import CAMINHO_DADOS_CURADOS
from radar.contratos import FiltrosEstruturados, PlanoConsulta

TOTAL_STARTUPS = 30


@pytest.fixture(scope="module")
def curadoria(tmp_path_factory):
    """Banco temporário semeado com a curadoria real, fora do repositório."""
    banco = tmp_path_factory.mktemp("curadoria") / "radar_curadoria.db"
    inicializar_banco(banco, CAMINHO_DADOS_CURADOS)
    with sqlite3.connect(banco) as conexao:
        ids = [linha[0] for linha in conexao.execute("SELECT id FROM startups ORDER BY id")]
    return BaseStartups(banco), ids


def plano() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(),
        termos_busca=["startup"],
        sinais_ia=[],
        foco_analise="conferência de fronteira da base curada",
    )


def test_a_curadoria_carrega_exatamente_trinta_startups(curadoria):
    _, ids = curadoria

    assert len(ids) == TOTAL_STARTUPS


def test_cada_startup_tem_pelo_menos_tres_documentos(curadoria):
    base, ids = curadoria

    for id_startup in ids:
        resultado = base.recuperar(plano(), id_startup)
        assert len(resultado.documentos) >= 3, f"startup {id_startup}"


@pytest.mark.parametrize("indice", range(TOTAL_STARTUPS))
def test_toda_startup_e_recuperavel_e_so_traz_os_proprios_documentos(
    curadoria, indice
):
    base, ids = curadoria
    id_startup = ids[indice]

    resultado = base.recuperar(plano(), id_startup)

    assert [empresa.id_startup for empresa in resultado.empresas] == [id_startup]
    assert {documento.id_startup for documento in resultado.documentos} == {id_startup}


def test_nenhum_id_de_documento_se_repete_entre_startups(curadoria):
    base, ids = curadoria

    vistos: dict[int, int] = {}
    for id_startup in ids:
        for documento in base.recuperar(plano(), id_startup).documentos:
            anterior = vistos.get(documento.id_documento)
            assert anterior is None, (
                f"documento {documento.id_documento} aparece nas startups "
                f"{anterior} e {id_startup}"
            )
            vistos[documento.id_documento] = id_startup


@pytest.mark.parametrize("indice", range(TOTAL_STARTUPS))
def test_o_texto_que_iria_ao_provedor_so_contem_documentos_da_startup(
    curadoria, indice
):
    base, ids = curadoria
    id_startup = ids[indice]
    resultado = base.recuperar(plano(), id_startup)
    permitidos = [documento.id_documento for documento in resultado.documentos]
    documentos = base.carregar_documentos(id_startup, permitidos)

    dados = Extractor._dados(id_startup, resultado.empresas[0], documentos)

    assert {documento.id_documento for documento in documentos} == set(permitidos)
    for id_documento in permitidos:
        assert f"[documento {id_documento}" in dados


@pytest.mark.parametrize("indice", range(TOTAL_STARTUPS))
def test_o_rotulo_de_curadoria_nunca_chega_ao_texto_do_provedor(curadoria, indice):
    base, ids = curadoria
    id_startup = ids[indice]
    resultado = base.recuperar(plano(), id_startup)
    permitidos = [documento.id_documento for documento in resultado.documentos]
    documentos = base.carregar_documentos(id_startup, permitidos)

    dados = Extractor._dados(id_startup, resultado.empresas[0], documentos)

    assert "classe_referencia" not in dados


def test_a_coluna_de_curadoria_existe_no_banco_mas_fica_fora_do_contrato(curadoria):
    """Prova que o teste acima não é vazio: o rótulo existe, e é omitido."""
    base, _ = curadoria

    with sqlite3.connect(base.caminho_banco) as conexao:
        colunas = {
            linha[1] for linha in conexao.execute("PRAGMA table_info(startups)")
        }

    assert "classe_referencia" in colunas
    assert "classe_referencia" not in set(
        base.recuperar(plano(), 1).empresas[0].model_dump()
    )
