"""O que a instrução de cada agente precisa ensinar antes de o modelo escrever.

Prompt é contrato de entrada: se a instrução não nomeia o que conta e o que não
conta, o modelo preenche a lacuna sozinho e a falha é silenciosa. Estes testes
travam o texto, não o comportamento em runtime — as portas determinísticas que
conferem a resposta vivem nos módulos de âncora e de polaridade.

A primeira parte cobre a matriz de sinais técnicos do Extractor: cada sinal
nomeado, o fato que o qualifica, o contraexemplo e as duas proibições que
impedem o salto de setor e de rótulo genérico.

A segunda parte cobre o mapa de fundamentos do Recommendation: gap estrutural,
dor documentada e oportunidade confirmada aparecem com os ids que os sustentam,
e nenhuma oportunidade é descrita como lacuna.
"""

from __future__ import annotations

import pytest

from radar.agentes.extractor import Extractor
from radar.agentes.recommendation import Recommendation
from radar.contratos import SINAIS_TECNICOS, FiltrosEstruturados, PlanoConsulta
from radar.regras_recomendacao import fundamentos_disponiveis
from tests.conftest import afirmacao_validada_falsa, perfil_validado_falso


# ------------------------------------------------------------------------
# Extractor: a matriz dos dez sinais técnicos
# ------------------------------------------------------------------------

def instrucao() -> str:
    plano = PlanoConsulta(
        filtros=FiltrosEstruturados(),
        termos_busca=["startup"],
        sinais_ia=[],
        foco_analise="uso de IA no produto vendido",
    )
    return Extractor._instrucao(7, plano, [1, 2, 3])


@pytest.mark.parametrize("sinal", SINAIS_TECNICOS)
def test_cada_sinal_aparece_nomeado_na_matriz(sinal):
    assert sinal in instrucao()


@pytest.mark.parametrize(
    "termo",
    [
        "modelo de linguagem",
        "ajuste fino",
        "transcrição",
        "volume",
        "modelo preditivo",
        "OCR",
        "robô",
        "radiologia",
        "multiagente",
        "detecção de ameaça",
    ],
)
def test_a_matriz_traz_o_fato_que_qualifica_cada_sinal(termo):
    assert termo in instrucao()


def test_cada_sinal_tem_um_contraexemplo_explicito():
    texto = instrucao()

    assert texto.count("NÃO qualifica") >= len(SINAIS_TECNICOS)


def test_a_matriz_proibe_derivar_sinal_do_setor():
    texto = instrucao().casefold()

    assert "setor" in texto
    assert "não cria sinal" in texto or "nunca cria sinal" in texto


def test_a_matriz_proibe_derivar_sinal_de_mencao_generica_a_ia():
    texto = instrucao()

    assert "usa IA" in texto or "usa inteligência artificial" in texto
    assert "genéric" in texto.casefold()


def test_zero_sinal_so_e_valido_depois_de_auditar_toda_a_matriz():
    texto = instrucao().casefold()

    assert "nenhum sinal" in texto or "zero sinal" in texto
    assert "avalie obrigatoriamente" in texto
    assert "cada afirmação" in texto
    assert "não deixe" in texto or "nao deixe" in texto


def test_extractor_exige_varredura_dos_documentos_por_cargas_tecnicas():
    texto = instrucao().casefold()

    assert "antes de finalizar" in texto
    assert "percorra todos os documentos" in texto
    assert "reconhecimento facial" in texto


def test_a_matriz_exige_todos_os_sinais_qualificados_na_mesma_afirmacao():
    texto = instrucao().casefold()

    assert "todos os sinais" in texto
    assert "mesma afirmação" in texto


def test_a_matriz_exige_que_o_proprio_trecho_citado_sustente_o_sinal():
    texto = instrucao().casefold()

    assert "trecho_citado" in texto


def test_trecho_citado_so_pode_vir_do_conteudo_texto_e_nunca_do_titulo():
    texto = instrucao().casefold()

    assert "exclusivamente do conteudo_texto" in texto
    assert "nunca copie do título" in texto


def test_a_instrucao_nao_manda_derivar_sinal_da_categoria():
    texto = instrucao().casefold()

    assert "categoria" in texto  # a matriz de polaridade continua lá
    assert "sinal a partir da categoria" not in texto


# ------------------------------------------------------------------------
# Recommendation: o mapa de fundamentos
# ------------------------------------------------------------------------

def perfil_completo():
    """Gap estrutural, dor documentada, oportunidade e dimensões bloqueadas."""
    itens = [
        # gap estrutural confirmado em distribuicao
        afirmacao_validada_falsa(1, "distribuicao", polaridade="ausencia_explicita"),
        # dor documentada
        afirmacao_validada_falsa(2, "dependencia_api_externa"),
        # oportunidade confirmada
        afirmacao_validada_falsa(
            3, "stack_propria", sinais_tecnicos=["inferencia_llm"]
        ),
        # capacidade confirmada — precisa aparecer como bloqueada
        afirmacao_validada_falsa(
            4, "dados_proprietarios", polaridade="presenca"
        ),
    ]
    return perfil_validado_falso(itens)


def mapa() -> str:
    perfil = perfil_completo()
    return Recommendation._mapa_de_fundamentos(
        perfil, fundamentos_disponiveis(perfil)
    )


def test_o_gap_estrutural_confirmado_aparece_com_os_ids_que_o_sustentam():
    texto = mapa()

    assert "distribuicao" in texto
    assert "[1]" in texto


def test_a_dor_documentada_aparece_com_os_ids_que_a_sustentam():
    texto = mapa()

    assert "dependencia_api_externa" in texto
    assert "[2]" in texto


def test_a_oportunidade_confirmada_aparece_com_os_ids_que_a_sustentam():
    texto = mapa()

    assert "inferencia_llm" in texto
    assert "[3]" in texto


def test_a_dimensao_com_capacidade_confirmada_aparece_como_bloqueada():
    texto = mapa()

    bloqueadas = texto.split("NÃO")[-1]
    assert "dados_proprietarios" in bloqueadas
    assert "capacidade_confirmada" in bloqueadas


def test_uma_dimensao_confirmada_como_gap_nao_aparece_como_bloqueada():
    texto = mapa()

    bloqueadas = texto.split("NÃO")[-1]
    assert "distribuicao" not in bloqueadas


def test_a_oportunidade_nunca_e_descrita_como_lacuna():
    texto = mapa()

    linhas = [linha for linha in texto.splitlines() if "inferencia_llm" in linha]

    assert linhas, "a oportunidade precisa estar no mapa"
    for linha in linhas:
        assert "gap" not in linha.casefold()
        assert "lacuna" not in linha.casefold()


def test_um_perfil_sem_fundamento_nenhum_nao_inventa_secao():
    perfil = perfil_validado_falso([afirmacao_validada_falsa(1, "outro")])

    texto = Recommendation._mapa_de_fundamentos(
        perfil, fundamentos_disponiveis(perfil)
    )

    assert texto.count("nenhuma") >= 2
