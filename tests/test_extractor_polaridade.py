"""Polaridade redundante em categoria não estrutural não pode derrubar o perfil.

O contrato é claro: só as quatro dimensões estruturais distinguem capacidade de
gap; nas outras seis categorias a polaridade não carrega informação e o único
valor válido é ``neutro``. Quando o Gemini devolve ``presenca`` em
``escala_e_dor_operacional``, ele não está afirmando nada de errado sobre a
empresa — está preenchendo um campo que, ali, não tem semântica. Corrigir isso
de forma determinística antes da validação não inventa fato nenhum; insistir na
recusa gasta o único retry e derruba a análise inteira por uma questão de forma.
"""

from __future__ import annotations

import copy

import pytest

from radar.agentes.extractor import (
    CATEGORIAS_NAO_ESTRUTURAIS,
    ErroExtractor,
    Extractor,
    normalizar_polaridade_nao_estrutural,
)
from radar.contratos import CATEGORIAS_ESTRUTURAIS, PerfilExtraido
from tests.test_extractor import (  # noqa: F401 - a fixture é registrada por import
    BaseControlada,
    ProvedorSequencial,
    TRECHO_ESCALA,
    controlada,
    estado,
    perfil_valido,
)


NAO_ESTRUTURAIS_ESPERADAS = (
    "stack_propria",
    "dependencia_api_externa",
    "escala_e_dor_operacional",
    "momento_e_financiamento",
    "equipe_e_contratacao",
    "outro",
)

REDUNDANTES = ("presenca", "ausencia_explicita")


def afirmacao_bruta(**ajustes) -> dict:
    campos = {
        "id_afirmacao": 7,
        "texto": "A plataforma processa quatro milhões de imagens por mês.",
        "categoria": "escala_e_dor_operacional",
        "polaridade": "neutro",
        "id_documento": 42,
        "trecho_citado": TRECHO_ESCALA,
    }
    campos.update(ajustes)
    return campos


def bruto_com(*afirmacoes) -> dict:
    return {
        "id_startup": 4,
        "resumo_produto": "A empresa vende inspeção visual. Atende doze fábricas.",
        "afirmacoes": list(afirmacoes),
    }


def polaridades(normalizado: dict) -> list[str]:
    return [item["polaridade"] for item in normalizado["afirmacoes"]]


# ----------------------------------------------------------------------
# O conjunto normalizável
# ----------------------------------------------------------------------


def test_as_seis_categorias_nao_estruturais_sao_exatamente_as_esperadas():
    assert set(CATEGORIAS_NAO_ESTRUTURAIS) == set(NAO_ESTRUTURAIS_ESPERADAS)
    assert not (set(CATEGORIAS_NAO_ESTRUTURAIS) & set(CATEGORIAS_ESTRUTURAIS))


# ----------------------------------------------------------------------
# O que é normalizado
# ----------------------------------------------------------------------


@pytest.mark.parametrize("categoria", NAO_ESTRUTURAIS_ESPERADAS)
@pytest.mark.parametrize("redundante", REDUNDANTES)
def test_polaridade_redundante_vira_neutro_em_categoria_nao_estrutural(
    categoria, redundante
):
    bruto = bruto_com(afirmacao_bruta(categoria=categoria, polaridade=redundante))

    normalizado = normalizar_polaridade_nao_estrutural(bruto)

    assert polaridades(normalizado) == ["neutro"]


@pytest.mark.parametrize("categoria", NAO_ESTRUTURAIS_ESPERADAS)
def test_neutro_valido_permanece_intocado(categoria):
    bruto = bruto_com(afirmacao_bruta(categoria=categoria, polaridade="neutro"))

    assert normalizar_polaridade_nao_estrutural(bruto) == bruto


def test_a_normalizacao_preserva_todos_os_outros_campos_da_afirmacao():
    original = afirmacao_bruta(polaridade="presenca")
    bruto = bruto_com(original)

    normalizado = normalizar_polaridade_nao_estrutural(bruto)
    corrigida = normalizado["afirmacoes"][0]

    assert corrigida["polaridade"] == "neutro"
    for campo in (
        "id_afirmacao",
        "texto",
        "categoria",
        "id_documento",
        "trecho_citado",
    ):
        assert corrigida[campo] == original[campo]
    assert normalizado["id_startup"] == bruto["id_startup"]
    assert normalizado["resumo_produto"] == bruto["resumo_produto"]
    assert len(normalizado["afirmacoes"]) == 1


def test_a_normalizacao_nao_muta_a_resposta_original_do_provedor():
    bruto = bruto_com(afirmacao_bruta(polaridade="presenca"))
    fotografia = copy.deepcopy(bruto)

    normalizar_polaridade_nao_estrutural(bruto)

    assert bruto == fotografia


def test_a_normalizacao_e_deterministica():
    bruto = bruto_com(afirmacao_bruta(polaridade="ausencia_explicita"))

    assert normalizar_polaridade_nao_estrutural(
        bruto
    ) == normalizar_polaridade_nao_estrutural(bruto)


# ----------------------------------------------------------------------
# O que NÃO é normalizado
# ----------------------------------------------------------------------


@pytest.mark.parametrize("categoria", sorted(CATEGORIAS_ESTRUTURAIS))
@pytest.mark.parametrize("polaridade", ("presenca", "ausencia_explicita", "neutro"))
def test_categoria_estrutural_nunca_e_tocada(categoria, polaridade):
    bruto = bruto_com(afirmacao_bruta(categoria=categoria, polaridade=polaridade))

    assert normalizar_polaridade_nao_estrutural(bruto) == bruto


def test_categoria_desconhecida_nao_e_reparada_em_silencio():
    bruto = bruto_com(
        afirmacao_bruta(categoria="categoria_inventada", polaridade="presenca")
    )

    assert normalizar_polaridade_nao_estrutural(bruto) == bruto


def test_polaridade_ausente_nao_e_reparada_em_silencio():
    afirmacao = afirmacao_bruta()
    del afirmacao["polaridade"]
    bruto = bruto_com(afirmacao)

    assert normalizar_polaridade_nao_estrutural(bruto) == bruto


@pytest.mark.parametrize("polaridade", ("PRESENCA", "positiva", "", None, 3))
def test_polaridade_irreconhecivel_nao_e_reparada_em_silencio(polaridade):
    bruto = bruto_com(afirmacao_bruta(polaridade=polaridade))

    assert normalizar_polaridade_nao_estrutural(bruto) == bruto


@pytest.mark.parametrize(
    "malformado",
    [
        "não é dicionário",
        None,
        {"id_startup": 4},
        {"id_startup": 4, "afirmacoes": "não é lista"},
        {"id_startup": 4, "afirmacoes": ["afirmação como texto"]},
        {"id_startup": 4, "afirmacoes": [None]},
    ],
)
def test_objeto_malformado_segue_para_a_validacao_sem_reparo(malformado):
    assert normalizar_polaridade_nao_estrutural(malformado) == malformado


def test_a_normalizacao_nao_cria_nem_remove_afirmacoes():
    bruto = bruto_com(
        afirmacao_bruta(id_afirmacao=1, polaridade="presenca"),
        afirmacao_bruta(id_afirmacao=2, categoria="dados_proprietarios"),
        afirmacao_bruta(id_afirmacao=3, categoria="outro", polaridade="presenca"),
    )

    normalizado = normalizar_polaridade_nao_estrutural(bruto)

    assert [item["id_afirmacao"] for item in normalizado["afirmacoes"]] == [1, 2, 3]
    assert polaridades(normalizado) == ["neutro", "neutro", "neutro"]
    assert [item["categoria"] for item in normalizado["afirmacoes"]] == [
        "escala_e_dor_operacional",
        "dados_proprietarios",
        "outro",
    ]


# ----------------------------------------------------------------------
# Comportamento do nó: o retry continua reservado ao erro real
# ----------------------------------------------------------------------


def perfil_com_polaridade_redundante(controlada) -> dict:
    bruto = perfil_valido(controlada)
    bruto["afirmacoes"][1]["polaridade"] = "presenca"  # escala_e_dor_operacional
    return bruto


def test_a_polaridade_redundante_nao_consome_o_retry_do_extractor(controlada):
    provedor = ProvedorSequencial(perfil_com_polaridade_redundante(controlada))

    resultado = Extractor(controlada.base, provedor)(estado(controlada))

    perfil = resultado["perfil_extraido"]
    assert isinstance(perfil, PerfilExtraido)
    assert len(provedor.chamadas) == 1  # o retry não foi usado
    assert resultado["tentativas_extracao"] == 1
    assert perfil.afirmacoes[1].polaridade == "neutro"


def test_a_normalizacao_no_no_preserva_texto_trecho_e_documento(controlada):
    original = perfil_com_polaridade_redundante(controlada)
    bruta = original["afirmacoes"][1]
    provedor = ProvedorSequencial(original)

    perfil = Extractor(controlada.base, provedor)(estado(controlada))["perfil_extraido"]
    corrigida = perfil.afirmacoes[1]

    assert corrigida.texto == bruta["texto"]
    assert corrigida.categoria == bruta["categoria"]
    assert corrigida.id_afirmacao == bruta["id_afirmacao"]
    assert corrigida.id_documento == bruta["id_documento"]
    assert corrigida.trecho_citado == bruta["trecho_citado"]
    assert perfil.id_startup == controlada.id_startup


def test_uma_afirmacao_estrutural_invalida_ainda_gasta_o_retry_e_falha_seguro(
    controlada,
):
    quebrado = perfil_valido(controlada)
    quebrado["afirmacoes"][0]["trecho_citado"] = "curto"  # viola o contrato de trecho
    provedor = ProvedorSequencial(quebrado, quebrado)

    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))

    assert len(provedor.chamadas) == 2


def test_uma_categoria_desconhecida_ainda_gasta_o_retry_e_falha_seguro(controlada):
    quebrado = perfil_valido(controlada)
    quebrado["afirmacoes"][1]["categoria"] = "categoria_inventada"
    quebrado["afirmacoes"][1]["polaridade"] = "presenca"
    provedor = ProvedorSequencial(quebrado, quebrado)

    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))

    assert len(provedor.chamadas) == 2


def test_a_normalizacao_nao_afrouxa_o_isolamento_de_documentos(controlada):
    invasor = perfil_com_polaridade_redundante(controlada)
    invasor["afirmacoes"][1]["id_documento"] = 9999
    provedor = ProvedorSequencial(invasor, invasor)

    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))

    assert len(provedor.chamadas) == 2
