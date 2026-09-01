from copy import deepcopy

import pytest

from radar.agentes.roteadores import rotear_r2, rotear_r3
from radar.configuracao import LIMIAR_DERRUBADA, MAX_EXTRACOES
from radar.contratos import (
    DIMENSOES_GAP,
    Classificacao,
    EstadoRadar,
    PerfilValidado,
)


def afirmacao_validada(
    id_afirmacao: int, situacao: str = "confirmada", motivo: str | None = None
) -> dict:
    return {
        "id_afirmacao": id_afirmacao,
        "texto": f"A afirmação {id_afirmacao} descreve um fato sobre o produto.",
        "categoria": "workflow_profundo",
        "polaridade": "presenca",
        "id_documento": 100 + id_afirmacao,
        "trecho_citado": "trecho literal presente no documento citado",
        "situacao": situacao,
        "motivo": motivo,
    }


def perfil_validado(
    situacoes: tuple[str, ...] = ("confirmada", "confirmada"),
    taxa_derrubada: float = 0.0,
) -> PerfilValidado:
    afirmacoes = [
        afirmacao_validada(
            indice,
            situacao,
            None if situacao == "confirmada" else "trecho não encontrado",
        )
        for indice, situacao in enumerate(situacoes, start=1)
    ]
    # As dimensões precisam derivar da evidência confirmada: o contrato de
    # PerfilValidado rejeita um artefato que declare o contrário do que as
    # próprias afirmações sustentam.
    confirmados = [
        indice
        for indice, situacao in enumerate(situacoes, start=1)
        if situacao == "confirmada"
    ]
    return PerfilValidado(
        afirmacoes_validadas=afirmacoes,
        taxa_derrubada=taxa_derrubada,
        hosts_distintos=["fonte-a.example", "fonte-b.example"],
        estado_dimensoes_gap=[
            {
                "dimensao": dimensao,
                "estado": (
                    "capacidade_confirmada"
                    if dimensao == "workflow_profundo" and confirmados
                    else "desconhecido"
                ),
                "ids_evidencias": (
                    confirmados if dimensao == "workflow_profundo" else []
                ),
            }
            for dimensao in DIMENSOES_GAP
        ],
    )


def classificacao(classe: str = "AI-enabled", suporte: list[int] | None = None):
    return Classificacao(
        classe=classe,
        justificativa=(
            "O produto existe sem depender integralmente de modelos. "
            "A inteligência artificial aparece como uma camada funcional."
        ),
        ids_afirmacoes_suporte=suporte or [1],
    )


def estado(
    *,
    perfil: PerfilValidado | None = None,
    classe: str = "AI-enabled",
    suporte: list[int] | None = None,
    tentativas: int = 1,
) -> EstadoRadar:
    return {
        "perfil_validado": perfil or perfil_validado(),
        "classificacao": classificacao(classe, suporte),
        "tentativas_extracao": tentativas,
        "erros": [],
        "trajeto": [],
    }


def test_constantes_de_r2_respeitam_a_arquitetura():
    assert LIMIAR_DERRUBADA == 0.5
    assert MAX_EXTRACOES == 2


def test_r2_reextrai_com_metade_das_afirmacoes_derrubadas():
    perfil = perfil_validado(("confirmada", "derrubada"), taxa_derrubada=0.5)
    assert rotear_r2(estado(perfil=perfil)) == "reextrair"


def test_r2_reextrai_quando_o_suporte_da_classe_foi_derrubado():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert rotear_r2(estado(perfil=perfil, suporte=[1])) == "reextrair"


def test_r2_segue_quando_evidencia_e_suporte_estao_prontos():
    assert rotear_r2(estado()) == "evidencia_pronta"


def test_r2_respeita_o_teto_e_nao_cria_loop_infinito():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert rotear_r2(estado(perfil=perfil, tentativas=2)) == "evidencia_pronta"


def test_r3_prioriza_zero_confirmadas_como_evidencia_insuficiente():
    perfil = perfil_validado(("derrubada",), taxa_derrubada=1.0)
    assert rotear_r3(estado(perfil=perfil, classe="non-AI")) == "evidencia_insuficiente"


def test_r3_prioriza_suporte_derrubado_como_evidencia_insuficiente():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert (
        rotear_r3(estado(perfil=perfil, classe="non-AI", suporte=[1], tentativas=2))
        == "evidencia_insuficiente"
    )


def test_r3_encerra_non_ai_com_suporte_confirmado_como_nao_aderente():
    assert rotear_r3(estado(classe="non-AI")) == "nao_aderente"


def test_r3_libera_classe_aderente_para_a_proxima_etapa():
    assert rotear_r3(estado(classe="AI-native")) == "prosseguir"


@pytest.mark.parametrize("roteador", [rotear_r2, rotear_r3])
def test_roteadores_sao_leitores_puros(roteador):
    entrada = estado()
    antes = deepcopy(entrada)
    roteador(entrada)
    assert entrada == antes


@pytest.mark.parametrize("roteador", [rotear_r2, rotear_r3])
def test_roteadores_falham_sem_contratos_obrigatorios(roteador):
    with pytest.raises((KeyError, TypeError, ValueError)):
        roteador({"tentativas_extracao": 1})
