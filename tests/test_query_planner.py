"""Query Planner: o plano estruturado e a escada de relaxamento.

A primeira parte cobre o nó: saída estruturada válida, o retry único, a falha
neutra de provedor e o relaxamento como cópia de plano — nunca geração.

A segunda parte fecha o laço determinístico Retriever → R1 → Query Planner
sobre a base curada real e **sem nenhuma chamada de LLM**: um setor controlado
estreito demais não pode encerrar a descoberta com uma empresa só, e a
ampliação precisa cruzar o setor original sem perder a empresa de origem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar.agentes.query_planner import ErroQueryPlanner, QueryPlanner
from radar.agentes.retriever import Retriever
from radar.agentes.roteadores import rotear_r1
from radar.aplicacao import construir_ranking
from radar.base_startups import BaseStartups
from radar.configuracao import (
    CAMINHO_DADOS_CURADOS,
    MINIMO_CANDIDATAS_UTEIS,
    TETO_RELAXAMENTO,
)
from radar.contratos import (
    EmpresaCandidata,
    FiltrosEstruturados,
    PlanoConsulta,
    ResultadoRecuperacao,
)
from radar.provedores import falha_operacional
from tests.conftest import (
    CORPO_BRUTO_DO_PROVEDOR,
    ProvedorSequencial,
    exigir_falha_neutra_de_provedor,
)

def plano_valido() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(
            setor="Saúde",
            estagio=["série C"],
            localizacao="São Paulo, SP",
            tamanho_time=["não divulgado"],
        ),
        termos_busca=["saúde"],
        sinais_ia=["inteligência artificial"],
        foco_analise="uso assistivo de IA",
    )


def resultado_com(plano: PlanoConsulta, quantidade: int) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        empresas=[
            EmpresaCandidata(
                id_startup=indice,
                nome=f"Exemplo {indice}",
                setor="Saúde",
                estagio="série C",
                localizacao="São Paulo, SP",
                descricao_curta="exemplo",
            )
            for indice in range(1, quantidade + 1)
        ],
        documentos=[],
        filtros_aplicados=plano.filtros,
    )


def resultado_vazio(filtros=None) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        empresas=[],
        documentos=[],
        filtros_aplicados=filtros or FiltrosEstruturados(),
    )


def test_saida_estruturada_invalida_e_rejeitada_apos_um_retry(base):
    invalida = {
        "filtros": {},
        "termos_busca": [],
        "sinais_ia": [],
        "foco_analise": "teste",
    }
    provedor = ProvedorSequencial(invalida, invalida)
    planejador = QueryPlanner(base, provedor)
    with pytest.raises(ErroQueryPlanner, match="duas vezes fora do contrato"):
        planejador({"consulta_usuario": "startups de saúde"})
    assert len(provedor.chamadas) == 2
    assert "Falha de validação" in provedor.chamadas[1][-1][1]


def test_falha_do_gemini_nao_fabrica_resultado(base):
    provedor = ProvedorSequencial(ConnectionError("segredo que não deve aparecer"))
    planejador = QueryPlanner(base, provedor)
    estado = {"consulta_usuario": "startups de saúde"}
    with pytest.raises(ErroQueryPlanner) as falha:
        planejador(estado)
    assert "segredo" not in str(falha.value)
    assert "resultado_recuperacao" not in estado
    assert "plano_consulta" not in estado


def test_relaxamento_ocorre_com_plano_e_ranking_abaixo_do_minimo(base):
    """A régua da reentrada é a mesma de R1: ranking curto também relaxa."""
    provedor = ProvedorSequencial()
    planejador = QueryPlanner(base, provedor)
    plano = plano_valido()

    relaxado = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_vazio(plano.filtros),
            "tentativas_relaxamento": 0,
        }
    )
    assert relaxado["tentativas_relaxamento"] == 1
    assert relaxado["plano_consulta"].filtros.estagio is None
    assert relaxado["plano_consulta"].filtros.localizacao is None
    assert relaxado["plano_consulta"].filtros.tamanho_time is None
    assert relaxado["plano_consulta"].filtros.setor == "Saúde"
    assert relaxado["plano_consulta"].termos_busca == plano.termos_busca
    assert relaxado["plano_consulta"].sinais_ia == plano.sinais_ia

    reutilizado = planejador({"plano_consulta": plano})
    assert "tentativas_relaxamento" not in reutilizado

    # Uma empresa é ranking curto demais: relaxa, igual a R1.
    relaxado_com_uma = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_com(plano, 1),
            "tentativas_relaxamento": 0,
        }
    )
    assert relaxado_com_uma["tentativas_relaxamento"] == 1

    # No mínimo útil o planejador devolve o plano intacto.
    nao_relaxado = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_com(plano, MINIMO_CANDIDATAS_UTEIS),
        }
    )
    assert "tentativas_relaxamento" not in nao_relaxado
    assert provedor.chamadas == []


def test_segundo_degrau_remove_apenas_setor(base):
    provedor = ProvedorSequencial()
    planejador = QueryPlanner(base, provedor)
    plano = plano_valido()
    saida = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_vazio(plano.filtros),
            "tentativas_relaxamento": 1,
        }
    )
    assert saida["tentativas_relaxamento"] == 2
    assert saida["criterios_relaxados"] == ["setor"]
    assert saida["plano_consulta"].filtros.setor is None
    assert saida["plano_consulta"].filtros.estagio == ["série C"]


# --------------------------------------------------------------------------
# Falha segura: a mensagem não nomeia o provedor
# --------------------------------------------------------------------------


def test_queda_do_provedor_nao_nomeia_gemini_nem_groq(base):
    provedor = ProvedorSequencial(ConnectionError(CORPO_BRUTO_DO_PROVEDOR))
    entrada = {"consulta_usuario": "startups de saúde"}

    with pytest.raises(ErroQueryPlanner) as falha:
        QueryPlanner(base, provedor)(entrada)

    mensagem = str(falha.value)
    exigir_falha_neutra_de_provedor(mensagem)
    assert "nenhum resultado foi fabricado" in mensagem
    # Queda não é resposta fora do contrato: não consome o retry corretivo.
    assert len(provedor.chamadas) == 1
    assert falha_operacional(falha.value) is True
    assert "plano_consulta" not in entrada
    assert "resultado_recuperacao" not in entrada


def test_duas_respostas_fora_do_contrato_nao_nomeiam_o_provedor(base):
    invalida = {
        "filtros": {},
        "termos_busca": [],
        "sinais_ia": [],
        "foco_analise": "teste",
    }
    provedor = ProvedorSequencial(invalida, invalida)
    entrada = {"consulta_usuario": "startups de saúde"}

    with pytest.raises(ErroQueryPlanner) as falha:
        QueryPlanner(base, provedor)(entrada)

    mensagem = str(falha.value)
    exigir_falha_neutra_de_provedor(mensagem)
    assert "duas vezes fora do contrato estruturado" in mensagem
    assert "a execução foi interrompida sem resultados" in mensagem
    # Teto de duas tentativas preservado.
    assert len(provedor.chamadas) == 2
    # Contrato violado é defeito de processamento, não indisponibilidade.
    assert falha_operacional(falha.value) is False
    assert "plano_consulta" not in entrada


# ------------------------------------------------------------------------
# A escada de relaxamento sobre a base curada real, sem LLM
# ------------------------------------------------------------------------

SETOR_ESTREITO = json.loads(
    (Path(CAMINHO_DADOS_CURADOS) / "13_pix_force.json").read_text(encoding="utf-8")
)["setor"]


class ProvedorProibido:
    """Qualquer chamada aqui é violação: a reentrada de relaxamento é pura."""

    def invocar(self, mensagens):  # pragma: no cover - o teste falha se rodar
        raise AssertionError(
            "a reentrada de relaxamento chamou o LLM do Query Planner"
        )


def plano_visao_computacional() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor=SETOR_ESTREITO),
        termos_busca=["visão computacional", "imagem"],
        sinais_ia=["visão computacional"],
        foco_analise="volume significativo de dados de imagem",
    )


def executar_descoberta_deterministica(base: BaseStartups, plano: PlanoConsulta):
    """Roda Retriever → R1 → Query Planner até R1 sair de `relaxar`."""
    planejador = QueryPlanner(base, ProvedorProibido())
    recuperador = Retriever(base)
    estado: dict = {
        "plano_consulta": plano,
        "tentativas_relaxamento": 0,
        "criterios_relaxados": [],
    }
    passos: list[dict] = []
    for _ in range(TETO_RELAXAMENTO + 2):
        estado.update(recuperador(estado))
        rota = rotear_r1(estado)
        passos.append(
            {
                "rota": rota,
                "empresas": [
                    empresa.nome for empresa in estado["resultado_recuperacao"].empresas
                ],
                "tentativas": int(estado.get("tentativas_relaxamento", 0)),
                "criterios": list(estado.get("criterios_relaxados", [])),
            }
        )
        if rota != "relaxar":
            return estado, rota, passos
        estado.update(planejador(estado))
    raise AssertionError("o laço de relaxamento não convergiu")


# --------------------------------------------------------------------------
# O ponto de partida: um setor controlado estreito isola uma empresa
# --------------------------------------------------------------------------


def test_setor_controlado_estreito_devolve_apenas_pix_force(base):
    """Ancora o ponto de partida: o filtro exato realmente isola uma empresa."""
    resultado = base.recuperar(plano_visao_computacional(), None)

    assert [empresa.nome for empresa in resultado.empresas] == ["Pix Force"]


def test_consulta_de_visao_computacional_nao_fica_presa_em_pix_force(base):
    _estado, rota, passos = executar_descoberta_deterministica(
        base, plano_visao_computacional()
    )

    assert passos[0]["rota"] == "relaxar", "uma empresa só não é ranking útil"
    assert passos[0]["empresas"] == ["Pix Force"]
    assert rota == "candidatas_prontas"
    assert len(passos[-1]["empresas"]) >= MINIMO_CANDIDATAS_UTEIS


def test_relaxamento_amplo_recupera_candidatas_de_outros_setores(base):
    estado, _rota, _passos = executar_descoberta_deterministica(
        base, plano_visao_computacional()
    )
    nomes = [empresa.nome for empresa in estado["resultado_recuperacao"].empresas]

    assert "Pix Force" in nomes, "a empresa do setor original não pode sumir"
    assert "Cromai" in nomes, "documento aderente de outro setor deve reaparecer"
    setores = {
        empresa.setor for empresa in estado["resultado_recuperacao"].empresas
    }
    assert setores - {SETOR_ESTREITO}, "a ampliação precisa cruzar o setor original"


def test_escada_preserva_ordem_termos_sinais_e_criterios(base):
    plano = plano_visao_computacional()
    estado, _rota, passos = executar_descoberta_deterministica(base, plano)
    final = estado["plano_consulta"]

    # Degrau 1 remove estágio/localização/tamanho de time; degrau 2 remove setor.
    assert passos[0]["criterios"] == []
    assert "setor" in estado["criterios_relaxados"]
    assert estado["criterios_relaxados"].index("setor") == len(
        estado["criterios_relaxados"]
    ) - 1
    assert final.filtros.setor is None
    assert final.termos_busca == plano.termos_busca
    assert final.sinais_ia == plano.sinais_ia
    assert final.filtros.classe_analisada == plano.filtros.classe_analisada
    assert int(estado["tentativas_relaxamento"]) <= TETO_RELAXAMENTO


def test_relaxamento_nao_chama_o_llm_do_query_planner(base):
    # ``ProvedorProibido`` levanta AssertionError se for invocado; chegar ao fim
    # sem exceção é a prova.
    _estado, rota, _passos = executar_descoberta_deterministica(
        base, plano_visao_computacional()
    )

    assert rota in {"candidatas_prontas", "sem_resultado"}


def test_ranking_permanece_deterministico(base):
    primeira, _r1, _p1 = executar_descoberta_deterministica(
        base, plano_visao_computacional()
    )
    segunda, _r2, _p2 = executar_descoberta_deterministica(
        base, plano_visao_computacional()
    )
    ids = [
        empresa.id_startup for empresa in primeira["resultado_recuperacao"].empresas
    ]

    assert ids == [
        empresa.id_startup for empresa in segunda["resultado_recuperacao"].empresas
    ]
    ranking_a = construir_ranking(primeira["resultado_recuperacao"], {})
    ranking_b = construir_ranking(segunda["resultado_recuperacao"], {})
    assert [item.empresa.id_startup for item in ranking_a] == [
        item.empresa.id_startup for item in ranking_b
    ]
    assert [item.posicao for item in ranking_a] == list(range(1, len(ranking_a) + 1))
