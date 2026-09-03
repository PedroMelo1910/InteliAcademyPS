"""Regras determinísticas do motor de recomendação: tabela, prioridade e complexidade.

Nenhum teste aqui usa LLM, banco ou rede: são funções puras sobre valores.
"""

from __future__ import annotations

import pytest

from radar.contratos import TECNOLOGIAS_NVIDIA
from radar.regras_recomendacao import (
    CATEGORIAS_DE_DOR,
    conferir_gap_sustentado,
    gaps_sustentados,
    COMPLEXIDADE_POR_TECNOLOGIA,
    COMPLEXIDADES_RECOMENDACAO,
    GAPS_ENDERECAVEIS,
    TECNOLOGIAS_POR_GAP,
    ErroRegraRecomendacao,
    calcular_complexidade,
    calcular_prioridade,
    estagio_na_janela_de_inflexao,
    tecnologias_candidatas,
)


# ----------------------------------------------------------------------
# Tabela fixa gap -> tecnologias candidatas
# ----------------------------------------------------------------------


def test_todo_gap_do_enum_tem_conjunto_candidato_nao_vazio():
    assert set(TECNOLOGIAS_POR_GAP) == set(GAPS_ENDERECAVEIS)
    for gap, candidatas in TECNOLOGIAS_POR_GAP.items():
        assert candidatas, f"o gap {gap} ficou sem tecnologia candidata"


def test_tabela_so_usa_as_dezesseis_tecnologias_do_tapi():
    usadas = {
        tecnologia
        for candidatas in TECNOLOGIAS_POR_GAP.values()
        for tecnologia in candidatas
    }
    assert usadas <= set(TECNOLOGIAS_NVIDIA)


def test_nenhuma_das_dezesseis_tecnologias_fica_inalcancavel():
    usadas = {
        tecnologia
        for candidatas in TECNOLOGIAS_POR_GAP.values()
        for tecnologia in candidatas
    }
    assert usadas == set(TECNOLOGIAS_NVIDIA)


def test_candidatas_nao_repetem_tecnologia_dentro_do_mesmo_gap():
    for gap, candidatas in TECNOLOGIAS_POR_GAP.items():
        assert len(set(candidatas)) == len(candidatas), gap


def test_tecnologias_candidatas_devolve_o_conjunto_do_gap():
    assert tecnologias_candidatas("otimizacao_tecnica") == TECNOLOGIAS_POR_GAP[
        "otimizacao_tecnica"
    ]


def test_tecnologias_candidatas_recusa_gap_fora_do_enum():
    with pytest.raises(ErroRegraRecomendacao, match="gap"):
        tecnologias_candidatas("gap_inventado")


def test_dependencia_de_api_externa_oferece_o_pacote_do_tapi():
    candidatas = set(TECNOLOGIAS_POR_GAP["dependencia_api_externa"])
    assert {
        "NVIDIA NIM",
        "NeMo Guardrails",
        "NVIDIA Triton Inference Server",
    } <= candidatas


def test_otimizacao_tecnica_oferece_o_pacote_de_latencia_do_tapi():
    candidatas = set(TECNOLOGIAS_POR_GAP["otimizacao_tecnica"])
    assert {"NVIDIA Triton Inference Server", "TensorRT-LLM"} <= candidatas


def test_dados_proprietarios_oferece_o_pacote_tabular_do_tapi():
    candidatas = set(TECNOLOGIAS_POR_GAP["dados_proprietarios"])
    assert {"NVIDIA RAPIDS", "cuDF", "cuML"} <= candidatas


def test_gap_de_distribuicao_e_o_unico_que_oferece_o_programa_inception():
    gaps_com_inception = {
        gap
        for gap, candidatas in TECNOLOGIAS_POR_GAP.items()
        if "NVIDIA Inception" in candidatas
    }
    assert gaps_com_inception == {"distribuicao"}


# ----------------------------------------------------------------------
# Prioridade (§10.2): determinística sobre evidência citada + estágio
# ----------------------------------------------------------------------


def test_janela_de_inflexao_cobre_seed_e_serie_a():
    assert estagio_na_janela_de_inflexao("Seed")
    assert estagio_na_janela_de_inflexao("Série A")
    assert estagio_na_janela_de_inflexao("serie a")


def test_janela_de_inflexao_cobre_a_grafia_inglesa_series_a():
    """A base curada é livre; "Series A" precisa valer o mesmo que "Série A"."""
    assert estagio_na_janela_de_inflexao("Series A")
    assert estagio_na_janela_de_inflexao("series a")
    assert estagio_na_janela_de_inflexao("SERIES A")


def test_janela_de_inflexao_exclui_pre_seed_e_series_avancadas():
    assert not estagio_na_janela_de_inflexao("Pre-seed")
    assert not estagio_na_janela_de_inflexao("Série B")
    assert not estagio_na_janela_de_inflexao("desconhecido")


def test_janela_de_inflexao_exclui_series_avancadas_em_ingles():
    assert not estagio_na_janela_de_inflexao("Series B")
    assert not estagio_na_janela_de_inflexao("Series C")
    assert not estagio_na_janela_de_inflexao("Series F")


@pytest.mark.parametrize("categoria_de_dor", sorted(CATEGORIAS_DE_DOR))
def test_prioridade_alta_exige_dor_citada_e_janela_de_estagio(categoria_de_dor):
    assert (
        calcular_prioridade(
            categorias_citadas=[categoria_de_dor, "outro"],
            estagio="Seed",
            gap_confirmado=True,
        )
        == "alta"
    )


def test_prioridade_cai_para_media_quando_a_dor_esta_fora_da_janela():
    assert (
        calcular_prioridade(
            categorias_citadas=["escala_e_dor_operacional"],
            estagio="Série C",
            gap_confirmado=True,
        )
        == "media"
    )


def test_prioridade_media_para_gap_confirmado_sem_dor_documentada():
    assert (
        calcular_prioridade(
            categorias_citadas=["workflow_profundo"],
            estagio="Seed",
            gap_confirmado=True,
        )
        == "media"
    )


def test_prioridade_baixa_para_aperfeicoamento_sem_dor_e_sem_gap_confirmado():
    assert (
        calcular_prioridade(
            categorias_citadas=["outro", "momento_e_financiamento"],
            estagio="Seed",
            gap_confirmado=False,
        )
        == "baixa"
    )


def test_prioridade_e_estavel_para_a_mesma_entrada():
    argumentos = {
        "categorias_citadas": ["dependencia_api_externa"],
        "estagio": "Série A",
        "gap_confirmado": True,
    }
    assert calcular_prioridade(**argumentos) == calcular_prioridade(**argumentos)


def test_prioridade_ignora_a_ordem_das_categorias_citadas():
    direta = calcular_prioridade(
        categorias_citadas=["outro", "escala_e_dor_operacional"],
        estagio="Seed",
        gap_confirmado=True,
    )
    inversa = calcular_prioridade(
        categorias_citadas=["escala_e_dor_operacional", "outro"],
        estagio="Seed",
        gap_confirmado=True,
    )
    assert direta == inversa == "alta"


# ----------------------------------------------------------------------
# Complexidade (§10.3): tabela fixa + ajuste único por P1
# ----------------------------------------------------------------------


def test_tabela_de_complexidade_cobre_exatamente_as_dezesseis_tecnologias():
    assert set(COMPLEXIDADE_POR_TECNOLOGIA) == set(TECNOLOGIAS_NVIDIA)
    assert set(COMPLEXIDADE_POR_TECNOLOGIA.values()) <= set(COMPLEXIDADES_RECOMENDACAO)


def test_inception_e_a_unica_adocao_de_complexidade_baixa():
    baixas = {
        tecnologia
        for tecnologia, nivel in COMPLEXIDADE_POR_TECNOLOGIA.items()
        if nivel == "baixa"
    }
    assert baixas == {"NVIDIA Inception"}


def test_complexidade_de_pacote_usa_a_maior_das_tecnologias():
    assert (
        calcular_complexidade(
            ["NVIDIA Inception", "NVIDIA NIM"], pontos_centralidade_ia=8
        )
        == "media"
    )
    assert (
        calcular_complexidade(
            ["NVIDIA NIM", "TensorRT-LLM"], pontos_centralidade_ia=8
        )
        == "alta"
    )


def test_centralidade_de_ia_baixa_sobe_um_degrau_de_complexidade():
    assert (
        calcular_complexidade(["NVIDIA Inception"], pontos_centralidade_ia=3)
        == "media"
    )
    assert calcular_complexidade(["NVIDIA NIM"], pontos_centralidade_ia=3) == "alta"


def test_ajuste_por_centralidade_nunca_ultrapassa_alta():
    assert calcular_complexidade(["CUDA"], pontos_centralidade_ia=0) == "alta"


def test_ajuste_por_centralidade_nunca_desce_um_degrau():
    assert (
        calcular_complexidade(["NVIDIA NIM"], pontos_centralidade_ia=10) == "media"
    )


def test_complexidade_recusa_pacote_vazio():
    with pytest.raises(ErroRegraRecomendacao, match="ao menos uma tecnologia"):
        calcular_complexidade([], pontos_centralidade_ia=5)


def test_complexidade_recusa_tecnologia_fora_das_dezesseis():
    with pytest.raises(ErroRegraRecomendacao, match="fora do catálogo"):
        calcular_complexidade(["Produto inventado"], pontos_centralidade_ia=5)


# ----------------------------------------------------------------------
# Elegibilidade: a evidência citada precisa sustentar o gap escolhido
# ----------------------------------------------------------------------


def _perfil(itens):
    from tests.conftest import perfil_validado_falso

    return perfil_validado_falso(itens)


def _afirmacao(id_afirmacao, categoria, **ajustes):
    from tests.conftest import afirmacao_validada_falsa

    return afirmacao_validada_falsa(id_afirmacao, categoria, **ajustes)


def test_dimensao_desconhecida_nao_entra_nos_gaps_sustentados():
    perfil = _perfil([_afirmacao(1, "outro")])
    assert gaps_sustentados(perfil) == {}


def test_capacidade_confirmada_nao_e_gap_sustentado():
    perfil = _perfil([_afirmacao(1, "distribuicao", polaridade="presenca")])
    assert "distribuicao" not in gaps_sustentados(perfil)


def test_gap_confirmado_traz_os_ids_que_o_sustentam():
    perfil = _perfil(
        [_afirmacao(1, "otimizacao_tecnica", polaridade="ausencia_explicita")]
    )
    assert gaps_sustentados(perfil)["otimizacao_tecnica"] == frozenset({1})


def test_conflito_vira_desconhecido_e_nao_sustenta_gap():
    perfil = _perfil(
        [
            _afirmacao(1, "workflow_profundo", polaridade="presenca"),
            _afirmacao(2, "workflow_profundo", polaridade="ausencia_explicita"),
        ]
    )
    assert "workflow_profundo" not in gaps_sustentados(perfil)


def test_categoria_de_dor_confirmada_sustenta_o_gap_homonimo():
    perfil = _perfil([_afirmacao(1, "escala_e_dor_operacional")])
    assert gaps_sustentados(perfil)["escala_e_dor_operacional"] == frozenset({1})


def test_categoria_de_dor_derrubada_nao_sustenta_gap():
    perfil = _perfil(
        [
            _afirmacao(1, "dependencia_api_externa", situacao="derrubada"),
            _afirmacao(2, "outro"),
        ]
    )
    assert gaps_sustentados(perfil) == {}


def test_gaps_sustentados_seguem_a_ordem_do_contrato():
    perfil = _perfil(
        [
            _afirmacao(1, "dependencia_api_externa"),
            _afirmacao(2, "dados_proprietarios", polaridade="ausencia_explicita"),
        ]
    )
    assert list(gaps_sustentados(perfil)) == [
        "dados_proprietarios",
        "dependencia_api_externa",
    ]


def test_conferir_aceita_id_que_pertence_ao_gap():
    sustentados = {"otimizacao_tecnica": frozenset({1, 4})}
    conferir_gap_sustentado("otimizacao_tecnica", [4, 9], sustentados)


def test_conferir_recusa_gap_sem_sustentacao():
    with pytest.raises(ErroRegraRecomendacao, match="não está sustentado"):
        conferir_gap_sustentado("distribuicao", [1], {"otimizacao_tecnica": frozenset({1})})


def test_conferir_recusa_evidencia_que_nao_pertence_ao_gap():
    sustentados = {
        "workflow_profundo": frozenset({2}),
        "otimizacao_tecnica": frozenset({1}),
    }
    with pytest.raises(ErroRegraRecomendacao, match="não sustentam o gap"):
        conferir_gap_sustentado("workflow_profundo", [1], sustentados)
