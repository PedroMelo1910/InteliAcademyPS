"""Roteadores determinísticos do grafo: R1, R2 e R3.

Os três são leitores puros do estado — não escrevem, não chamam provedor e não
tocam banco. R1 decide quando o ranking já é útil, quando relaxar e quando
encerrar sem resultado. R2 decide quando reextrair diante de evidência
derrubada, respeitando o teto que impede laço infinito. R3 decide entre
evidência insuficiente, desfecho não aderente e seguir para a próxima etapa.
"""

from copy import deepcopy

import pytest

from radar.agentes.roteadores import rotear_r1, rotear_r2, rotear_r3
from radar.configuracao import (
    LIMIAR_DERRUBADA,
    MAX_EXTRACOES,
    MINIMO_CANDIDATAS_UTEIS,
    TETO_RELAXAMENTO,
)
from radar.contratos import (
    DIMENSOES_GAP,
    Classificacao,
    EmpresaCandidata,
    EstadoRadar,
    FiltrosEstruturados,
    PerfilValidado,
    ResultadoRecuperacao,
)


# ------------------------------------------------------------------------
# R1 — ranking mínimo útil e escada de relaxamento
# ------------------------------------------------------------------------

def resultado(empresas=None):
    return ResultadoRecuperacao(
        empresas=empresas or [],
        documentos=[],
        filtros_aplicados=FiltrosEstruturados(),
    )


def empresa(id_startup: int = 1):
    return EmpresaCandidata(
        id_startup=id_startup,
        nome=f"Empresa {id_startup}",
        setor="Saúde",
        estagio="seed",
        localizacao="São Paulo, SP",
        descricao_curta="Descrição",
    )


def empresas(quantidade: int):
    return [empresa(indice) for indice in range(1, quantidade + 1)]


def estado_r1(quantidade: int, tentativas: int):
    return {
        "resultado_recuperacao": resultado(empresas(quantidade)),
        "tentativas_relaxamento": tentativas,
    }


# --------------------------------------------------------------------------
# Prioridade absoluta do aprofundamento
# --------------------------------------------------------------------------


def test_startup_selecionada_tem_prioridade_e_dispara_analisar():
    assert rotear_r1(
        {
            "startup_selecionada": 1,
            "resultado_recuperacao": resultado(),
            "tentativas_relaxamento": TETO_RELAXAMENTO,
        }
    ) == "analisar"


def test_startup_selecionada_vence_ate_com_ranking_suficiente():
    """O aprofundamento não pode ser sequestrado pelo tamanho do ranking."""
    assert rotear_r1(
        {
            "startup_selecionada": 7,
            **estado_r1(MINIMO_CANDIDATAS_UTEIS, 0),
        }
    ) == "analisar"


# --------------------------------------------------------------------------
# Ranking mínimo útil: 3 encerra, 0/1/2 relaxam enquanto houver tentativa
# --------------------------------------------------------------------------


def test_minimo_util_e_constante_nomeada_de_configuracao():
    assert MINIMO_CANDIDATAS_UTEIS == 3


def test_zero_candidatas_dispara_relaxar():
    assert rotear_r1(estado_r1(0, 0)) == "relaxar"


def test_uma_candidata_com_tentativa_disponivel_dispara_relaxar():
    """O defeito ao vivo: Pix Force sozinha encerrava a descoberta."""
    assert rotear_r1(estado_r1(1, 0)) == "relaxar"


def test_duas_candidatas_com_tentativa_disponivel_dispara_relaxar():
    assert rotear_r1(estado_r1(2, 0)) == "relaxar"


def test_uma_candidata_no_segundo_degrau_ainda_dispara_relaxar():
    assert rotear_r1(estado_r1(1, TETO_RELAXAMENTO - 1)) == "relaxar"


def test_tres_candidatas_dispara_candidatas_prontas():
    assert rotear_r1(estado_r1(MINIMO_CANDIDATAS_UTEIS, 0)) == "candidatas_prontas"


def test_acima_do_minimo_dispara_candidatas_prontas():
    assert rotear_r1(estado_r1(MINIMO_CANDIDATAS_UTEIS + 5, 0)) == "candidatas_prontas"


# --------------------------------------------------------------------------
# Depois do teto: parcial é entregue, vazio é falha honesta
# --------------------------------------------------------------------------


def test_uma_candidata_apos_o_teto_permanece_candidatas_prontas():
    """Resultado parcial válido nunca vira `sem_resultado`."""
    assert rotear_r1(estado_r1(1, TETO_RELAXAMENTO)) == "candidatas_prontas"


def test_duas_candidatas_apos_o_teto_permanecem_candidatas_prontas():
    assert rotear_r1(estado_r1(2, TETO_RELAXAMENTO)) == "candidatas_prontas"


def test_teto_de_relaxamento_sem_empresas_dispara_sem_resultado():
    assert rotear_r1(estado_r1(0, TETO_RELAXAMENTO)) == "sem_resultado"


def test_r1_nunca_devolve_sem_resultado_com_empresa_no_estado():
    for quantidade in range(1, MINIMO_CANDIDATAS_UTEIS + 2):
        for tentativas in range(0, TETO_RELAXAMENTO + 2):
            assert rotear_r1(estado_r1(quantidade, tentativas)) != "sem_resultado"


# ------------------------------------------------------------------------
# R2 e R3 — evidência derrubada, desfechos terminais
# ------------------------------------------------------------------------

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


def estado_evidencia(
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
    assert rotear_r2(estado_evidencia(perfil=perfil)) == "reextrair"


def test_r2_reextrai_quando_o_suporte_da_classe_foi_derrubado():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert rotear_r2(estado_evidencia(perfil=perfil, suporte=[1])) == "reextrair"


def test_r2_segue_quando_evidencia_e_suporte_estao_prontos():
    assert rotear_r2(estado_evidencia()) == "evidencia_pronta"


def test_r2_respeita_o_teto_e_nao_cria_loop_infinito():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert rotear_r2(estado_evidencia(perfil=perfil, tentativas=2)) == "evidencia_pronta"


def test_r3_prioriza_zero_confirmadas_como_evidencia_insuficiente():
    perfil = perfil_validado(("derrubada",), taxa_derrubada=1.0)
    assert rotear_r3(estado_evidencia(perfil=perfil, classe="non-AI")) == "evidencia_insuficiente"


def test_r3_prioriza_suporte_derrubado_como_evidencia_insuficiente():
    perfil = perfil_validado(("derrubada", "confirmada"), taxa_derrubada=0.5)
    assert (
        rotear_r3(estado_evidencia(perfil=perfil, classe="non-AI", suporte=[1], tentativas=2))
        == "evidencia_insuficiente"
    )


def test_r3_encerra_non_ai_com_suporte_confirmado_como_nao_aderente():
    assert rotear_r3(estado_evidencia(classe="non-AI")) == "nao_aderente"


def test_r3_libera_classe_aderente_para_a_proxima_etapa():
    assert rotear_r3(estado_evidencia(classe="AI-native")) == "prosseguir"


@pytest.mark.parametrize("roteador", [rotear_r2, rotear_r3])
def test_roteadores_sao_leitores_puros(roteador):
    entrada = estado_evidencia()
    antes = deepcopy(entrada)
    roteador(entrada)
    assert entrada == antes


@pytest.mark.parametrize("roteador", [rotear_r2, rotear_r3])
def test_roteadores_falham_sem_contratos_obrigatorios(roteador):
    with pytest.raises((KeyError, TypeError, ValueError)):
        roteador({"tentativas_extracao": 1})
