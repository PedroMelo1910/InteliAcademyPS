"""Sinal técnico: oportunidade NVIDIA com lastro, sem virar inferência.

O sistema já recomendava a partir de gap confirmado ou dor documentada. O TAPI
§5.5 também recomenda a partir de carga de trabalho positivamente observada —
dados em escala, voz, visão, robótica, agentes governados. O que estes testes
travam é a fronteira: um sinal só existe quando a própria afirmação confirmada
o carrega, nunca por setor, nome da empresa, capacidade estrutural ampla ou
silêncio da fonte. Afirmação derrubada perde o sinal junto.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radar.contratos import (
    Afirmacao,
    AfirmacaoValidada,
    SINAIS_TECNICOS,
    TECNOLOGIAS_NVIDIA,
)
from radar.regras_recomendacao import (
    GAPS_ENDERECAVEIS,
    TECNOLOGIAS_POR_SINAL,
    ErroRegraRecomendacao,
    conferir_fundamento_sustentado,
    fundamentos_disponiveis,
    gaps_sustentados,
    sinais_sustentados,
    tecnologias_candidatas,
)
from tests.conftest import (
    afirmacao_validada_falsa,
    perfil_validado_falso,
    trecho_citado_falso,
)


SINAIS_ESPERADOS = (
    "inferencia_llm",
    "treinamento_ou_finetuning",
    "voz_fala_ou_transcricao",
    "dados_em_escala",
    "machine_learning_classico",
    "visao_computacional",
    "robotica_ou_simulacao",
    "imagem_medica",
    "agentes_com_acoes_ou_controles",
    "ciberseguranca_em_escala",
)


def afirmacao_bruta(**ajustes) -> dict:
    campos = {
        "id_afirmacao": 1,
        "texto": "A plataforma executa um modelo de linguagem próprio em produção.",
        "categoria": "stack_propria",
        "polaridade": "neutro",
        "id_documento": 1,
        "trecho_citado": trecho_citado_falso(1),
    }
    campos.update(ajustes)
    return campos


# ----------------------------------------------------------------------
# Vocabulário fechado
# ----------------------------------------------------------------------


def test_o_vocabulario_de_sinais_e_exatamente_o_aprovado():
    assert SINAIS_TECNICOS == SINAIS_ESPERADOS


def test_os_dominios_de_gap_e_de_sinal_sao_disjuntos():
    assert not (set(SINAIS_TECNICOS) & set(GAPS_ENDERECAVEIS))


# ----------------------------------------------------------------------
# Contrato da afirmação
# ----------------------------------------------------------------------


def test_uma_afirmacao_sem_sinal_continua_valida_e_o_default_e_vazio():
    afirmacao = Afirmacao(**afirmacao_bruta())

    assert afirmacao.sinais_tecnicos == ()


@pytest.mark.parametrize("sinal", SINAIS_ESPERADOS)
def test_cada_sinal_do_vocabulario_e_aceito(sinal):
    afirmacao = Afirmacao(**afirmacao_bruta(sinais_tecnicos=[sinal]))

    assert afirmacao.sinais_tecnicos == (sinal,)


def test_sinal_fora_do_vocabulario_e_recusado():
    with pytest.raises(ValidationError):
        Afirmacao(**afirmacao_bruta(sinais_tecnicos=["computacao_quantica"]))


def test_sinal_repetido_e_recusado():
    with pytest.raises(ValidationError, match="repet"):
        Afirmacao(
            **afirmacao_bruta(sinais_tecnicos=["inferencia_llm", "inferencia_llm"])
        )


def test_os_sinais_saem_na_ordem_canonica_do_enum():
    afirmacao = Afirmacao(
        **afirmacao_bruta(
            sinais_tecnicos=["visao_computacional", "inferencia_llm"]
        )
    )

    assert afirmacao.sinais_tecnicos == ("inferencia_llm", "visao_computacional")


def test_uma_ausencia_declarada_nunca_carrega_sinal():
    with pytest.raises(ValidationError, match="ausencia_explicita"):
        Afirmacao(
            **afirmacao_bruta(
                categoria="distribuicao",
                polaridade="ausencia_explicita",
                sinais_tecnicos=["inferencia_llm"],
            )
        )


def test_o_sinal_nao_altera_categoria_nem_polaridade():
    bruta = afirmacao_bruta(sinais_tecnicos=["dados_em_escala"])

    afirmacao = Afirmacao(**bruta)

    assert afirmacao.categoria == bruta["categoria"]
    assert afirmacao.polaridade == bruta["polaridade"]


def test_campo_extra_continua_proibido():
    with pytest.raises(ValidationError):
        Afirmacao(**afirmacao_bruta(sinal_inventado="x"))


def test_a_afirmacao_validada_herda_o_campo_sem_mudanca():
    validada = AfirmacaoValidada(
        **afirmacao_bruta(sinais_tecnicos=["visao_computacional"]),
        situacao="confirmada",
        motivo=None,
    )

    assert validada.sinais_tecnicos == ("visao_computacional",)


# ----------------------------------------------------------------------
# Agregação determinística
# ----------------------------------------------------------------------


def perfil_com(*pares):
    """Cada par é (categoria, sinais, situacao)."""
    itens = []
    for indice, (categoria, sinais, situacao) in enumerate(pares, start=1):
        itens.append(
            afirmacao_validada_falsa(
                indice,
                categoria,
                situacao=situacao,
                sinais_tecnicos=sinais,
            )
        )
    return perfil_validado_falso(itens)


def test_sinais_sustentados_agrupa_ids_por_sinal():
    perfil = perfil_com(
        ("stack_propria", ["inferencia_llm"], "confirmada"),
        ("escala_e_dor_operacional", ["inferencia_llm"], "confirmada"),
        ("outro", ["visao_computacional"], "confirmada"),
    )

    sinais = sinais_sustentados(perfil)

    assert sinais["inferencia_llm"] == frozenset({1, 2})
    assert sinais["visao_computacional"] == frozenset({3})


def test_uma_afirmacao_derrubada_nao_sustenta_o_proprio_sinal():
    perfil = perfil_com(
        ("stack_propria", ["inferencia_llm"], "derrubada"),
        ("outro", ["visao_computacional"], "confirmada"),
    )

    sinais = sinais_sustentados(perfil)

    assert "inferencia_llm" not in sinais
    assert sinais["visao_computacional"] == frozenset({2})


def test_um_perfil_sem_sinal_nenhum_devolve_dicionario_vazio():
    perfil = perfil_com(("outro", [], "confirmada"))

    assert sinais_sustentados(perfil) == {}


def test_os_sinais_saem_na_ordem_canonica_para_prompt_reprodutivel():
    perfil = perfil_com(
        ("outro", ["visao_computacional"], "confirmada"),
        ("stack_propria", ["inferencia_llm"], "confirmada"),
    )

    assert list(sinais_sustentados(perfil)) == [
        "inferencia_llm",
        "visao_computacional",
    ]


# ----------------------------------------------------------------------
# Fundamentos combinados
# ----------------------------------------------------------------------


def test_fundamentos_reune_gap_e_sinal_com_o_tipo_correto():
    perfil = perfil_com(
        ("dependencia_api_externa", [], "confirmada"),
        ("stack_propria", ["inferencia_llm"], "confirmada"),
    )

    fundamentos = fundamentos_disponiveis(perfil)

    assert ("gap_confirmado", "dependencia_api_externa") in fundamentos
    assert ("oportunidade_confirmada", "inferencia_llm") in fundamentos


def test_o_caminho_de_gap_existente_nao_e_enfraquecido():
    perfil = perfil_com(("dependencia_api_externa", [], "confirmada"))

    fundamentos = fundamentos_disponiveis(perfil)
    gaps = gaps_sustentados(perfil)

    for gap, ids in gaps.items():
        assert fundamentos[("gap_confirmado", gap)] == ids


def test_um_perfil_so_com_sinal_ja_e_elegivel():
    perfil = perfil_com(("outro", ["robotica_ou_simulacao"], "confirmada"))

    assert gaps_sustentados(perfil) == {}
    assert fundamentos_disponiveis(perfil) == {
        ("oportunidade_confirmada", "robotica_ou_simulacao"): frozenset({1})
    }


def test_conferir_fundamento_exige_id_que_sustente_aquele_fundamento():
    perfil = perfil_com(
        ("outro", ["visao_computacional"], "confirmada"),
        ("stack_propria", ["inferencia_llm"], "confirmada"),
    )
    fundamentos = fundamentos_disponiveis(perfil)

    conferir_fundamento_sustentado(
        "oportunidade_confirmada", "visao_computacional", [1], fundamentos
    )
    with pytest.raises(ErroRegraRecomendacao, match="não sustentam"):
        conferir_fundamento_sustentado(
            "oportunidade_confirmada", "visao_computacional", [2], fundamentos
        )


def test_conferir_fundamento_vale_igualmente_para_o_gap_confirmado():
    """A função ativa é uma só: gap e oportunidade passam pelo mesmo elo."""
    perfil = perfil_com(("dependencia_api_externa", [], "confirmada"))
    fundamentos = fundamentos_disponiveis(perfil)

    conferir_fundamento_sustentado(
        "gap_confirmado", "dependencia_api_externa", [1], fundamentos
    )
    with pytest.raises(ErroRegraRecomendacao, match="não sustentam"):
        conferir_fundamento_sustentado(
            "gap_confirmado", "dependencia_api_externa", [99], fundamentos
        )


def test_conferir_fundamento_recusa_fundamento_inexistente():
    perfil = perfil_com(("outro", ["visao_computacional"], "confirmada"))
    fundamentos = fundamentos_disponiveis(perfil)

    with pytest.raises(ErroRegraRecomendacao, match="não está sustentad"):
        conferir_fundamento_sustentado(
            "oportunidade_confirmada", "imagem_medica", [1], fundamentos
        )


# ----------------------------------------------------------------------
# Mapa sinal → tecnologia
# ----------------------------------------------------------------------


MAPA_APROVADO = {
    "inferencia_llm": (
        "NVIDIA NIM",
        "NVIDIA Triton Inference Server",
        "TensorRT-LLM",
        "NVIDIA AI Enterprise",
    ),
    "treinamento_ou_finetuning": ("NVIDIA NeMo", "CUDA", "NVIDIA AI Enterprise"),
    "voz_fala_ou_transcricao": ("NVIDIA Riva", "NVIDIA NIM"),
    "dados_em_escala": ("NVIDIA RAPIDS", "cuDF", "NVIDIA AI Enterprise"),
    "machine_learning_classico": ("cuML", "NVIDIA RAPIDS", "cuDF"),
    "visao_computacional": (
        "CUDA",
        "NVIDIA Triton Inference Server",
        "NVIDIA NIM",
        "NVIDIA AI Enterprise",
    ),
    "robotica_ou_simulacao": ("NVIDIA Isaac", "NVIDIA Omniverse", "CUDA"),
    "imagem_medica": ("NVIDIA Clara", "NVIDIA NIM", "NVIDIA AI Enterprise"),
    "agentes_com_acoes_ou_controles": (
        "NeMo Guardrails",
        "NVIDIA NeMo",
        "NVIDIA NIM",
    ),
    "ciberseguranca_em_escala": ("NVIDIA Morpheus", "NVIDIA AI Enterprise"),
}


def test_o_mapa_cobre_exatamente_os_dez_sinais():
    assert set(TECNOLOGIAS_POR_SINAL) == set(SINAIS_ESPERADOS)


@pytest.mark.parametrize("sinal", SINAIS_ESPERADOS)
def test_cada_sinal_mapeia_para_o_conjunto_aprovado(sinal):
    assert TECNOLOGIAS_POR_SINAL[sinal] == MAPA_APROVADO[sinal]


def test_nenhuma_tecnologia_do_mapa_esta_fora_do_catalogo_aceito():
    for tecnologias in TECNOLOGIAS_POR_SINAL.values():
        for tecnologia in tecnologias:
            assert tecnologia in TECNOLOGIAS_NVIDIA


def test_tecnologias_candidatas_despacha_por_tipo_de_fundamento():
    assert tecnologias_candidatas(
        "oportunidade_confirmada", "robotica_ou_simulacao"
    ) == MAPA_APROVADO["robotica_ou_simulacao"]
    assert "NVIDIA Inception" in tecnologias_candidatas(
        "gap_confirmado", "distribuicao"
    )


def test_um_identificador_do_dominio_errado_e_recusado():
    with pytest.raises(ErroRegraRecomendacao):
        tecnologias_candidatas("gap_confirmado", "visao_computacional")
    with pytest.raises(ErroRegraRecomendacao):
        tecnologias_candidatas("oportunidade_confirmada", "distribuicao")
