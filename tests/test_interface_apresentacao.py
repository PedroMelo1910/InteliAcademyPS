"""As promessas de apresentação da tela, provadas pelo comportamento.

Estes testes não olham pixel. Eles olham as promessas que a redação da tela faz
ao avaliador: a abertura diz o que o produto é, os exemplos preenchem sem
executar, as duas medidas continuam separadas, a ordem e os números vêm prontos
da aplicação, a volta não refaz trabalho, os seis desfechos continuam
renderizáveis e o único HTML bruto da aplicação é uma folha de estilo literal.
"""

from __future__ import annotations

import ast

import pytest

from radar.configuracao import RAIZ_PROJETO
from radar.interface import mensagens
from radar.interface.tema import CSS_TEMA
from tests.apoio_interface import (
    AplicacaoFalsa,
    CONSULTA_PADRAO,
    abrir_ranking,
    briefing_na_tela,
    descoberta_falsa,
    item_ausente,
    item_concluido_ia,
    item_concluido_non_ai,
    item_falso,
    item_insuficiente,
    montar,
    textos,
)
from tests.conftest import (
    briefing_insuficiente_falso,
    briefing_nao_aderente_falso,
    briefing_normal_falso,
)

def botoes_de_exemplo(teste):
    return [botao for botao in teste.button if str(botao.key).startswith("exemplo_")]


# ----------------------------------------------------------------------
# 1. Abertura e busca
# ----------------------------------------------------------------------


def test_a_abertura_declara_o_produto_e_o_proposito_em_uma_frase():
    teste = montar(AplicacaoFalsa())

    tela = textos(teste)

    assert mensagens.NOME_PRODUTO in tela
    assert mensagens.PROPOSITO in tela


def test_a_abertura_oferece_de_duas_a_tres_perguntas_de_exemplo():
    teste = montar(AplicacaoFalsa())

    exemplos = botoes_de_exemplo(teste)

    assert 2 <= len(exemplos) <= 3
    assert len(mensagens.PERGUNTAS_DE_EXEMPLO) == len(exemplos)


def test_um_exemplo_apenas_preenche_o_campo_e_nao_executa_a_busca():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)

    teste.button(key="exemplo_0").click().run()

    assert aplicacao.consultas == []
    assert teste.session_state["consulta"] == mensagens.PERGUNTAS_DE_EXEMPLO[0]
    assert teste.text_input(key="consulta").value == mensagens.PERGUNTAS_DE_EXEMPLO[0]


def test_os_exemplos_saem_de_cena_depois_que_existe_um_ranking():
    teste = abrir_ranking(AplicacaoFalsa())

    assert botoes_de_exemplo(teste) == []


# ----------------------------------------------------------------------
# 2. Painel do ranking
# ----------------------------------------------------------------------


def test_o_painel_do_ranking_conta_cada_estado_de_analise():
    teste = abrir_ranking(AplicacaoFalsa())

    tela = textos(teste)

    assert f"{mensagens.ROTULO_TOTAL_CANDIDATAS}: 4" in tela
    assert f"{mensagens.ROTULO_TOTAL_ANALISADAS}: 2" in tela
    assert f"{mensagens.ROTULO_TOTAL_SEM_LASTRO}: 1" in tela
    assert f"{mensagens.ROTULO_TOTAL_PENDENTES}: 1" in tela


def test_as_duas_medidas_aparecem_rotuladas_e_explicadas_em_separado():
    teste = abrir_ranking(AplicacaoFalsa())

    tela = textos(teste)

    assert mensagens.ROTULO_FIT_SCORE in tela
    assert mensagens.ROTULO_BM25 in tela
    assert mensagens.EXPLICACAO_FIT_SCORE in tela
    assert mensagens.EXPLICACAO_BM25 in tela


def test_a_ordem_e_os_numeros_do_ranking_vem_prontos_da_aplicacao():
    """Pontuações fora de ordem provam que a tela não reordena nem renormaliza."""
    itens = (
        item_falso(1, 11, "Baixa", fit_score_total=12),
        item_falso(2, 12, "Alta", fit_score_total=95),
        item_falso(3, 13, "Media", fit_score_total=40),
    )
    teste = abrir_ranking(AplicacaoFalsa(descoberta=descoberta_falsa(itens=itens)))

    tela = textos(teste)

    assert tela.index("Baixa") < tela.index("Alta") < tela.index("Media")
    for pontuacao in ("12/100", "95/100", "40/100"):
        assert pontuacao in tela


def test_o_gate_non_ai_aparece_como_zero_deliberado_e_nao_como_erro():
    itens = (item_concluido_non_ai(posicao=1, id_startup=2, nome="Wine"),)
    teste = abrir_ranking(AplicacaoFalsa(descoberta=descoberta_falsa(itens=itens)))

    tela = textos(teste)

    assert "0/100" in tela
    assert "non-AI" in tela
    assert "erro" not in tela.casefold()
    assert not teste.error


# ----------------------------------------------------------------------
# 3. Tela da análise
# ----------------------------------------------------------------------


def test_a_analise_organiza_o_briefing_em_quatro_seccoes_claras():
    teste = briefing_na_tela(briefing_normal_falso())

    abas = [aba.label for aba in teste.tabs]

    assert abas == [
        mensagens.ABA_VISAO_GERAL,
        mensagens.ABA_EVIDENCIAS,
        mensagens.ABA_RECOMENDACOES,
        mensagens.ABA_RASTRO,
    ]


@pytest.mark.parametrize(
    "construtor", [briefing_nao_aderente_falso, briefing_insuficiente_falso]
)
def test_uma_variante_sem_recomendacao_nao_abre_a_aba_de_recomendacao(construtor):
    teste = briefing_na_tela(construtor())

    abas = [aba.label for aba in teste.tabs]

    assert mensagens.ABA_RECOMENDACOES not in abas
    assert mensagens.ABA_VISAO_GERAL in abas
    assert mensagens.ABA_RASTRO in abas


def test_a_tese_e_a_sintese_aparecem_antes_do_detalhe_tecnico():
    briefing = briefing_normal_falso()

    teste = briefing_na_tela(briefing)
    tela = textos(teste)

    assert tela.index(briefing.veredito.tese) < tela.index(
        briefing.rodape.versao_rubrica
    )


def test_voltar_ao_ranking_nao_refaz_a_busca_nem_a_analise():
    aplicacao = AplicacaoFalsa()
    teste = abrir_ranking(aplicacao)
    teste.button(key="aprofundar_1").click().run()

    teste.button(key="voltar_para_candidatas").click().run()

    assert aplicacao.consultas == [CONSULTA_PADRAO]
    assert len(aplicacao.aprofundamentos) == 1
    assert "Maritaca AI" in textos(teste)
    assert not teste.exception


# ----------------------------------------------------------------------
# 4. Os seis desfechos continuam renderizáveis
# ----------------------------------------------------------------------


def desfecho_normal():
    return briefing_na_tela(briefing_normal_falso())


def desfecho_non_ai():
    return briefing_na_tela(briefing_nao_aderente_falso())


def desfecho_evidencia_insuficiente():
    return briefing_na_tela(briefing_insuficiente_falso())


def desfecho_sem_resultado():
    return abrir_ranking(AplicacaoFalsa(descoberta=descoberta_falsa(itens=())))


def desfecho_analise_ausente():
    itens = (item_ausente(posicao=1, id_startup=4, nome="Mombak"),)
    return abrir_ranking(AplicacaoFalsa(descoberta=descoberta_falsa(itens=itens)))


def desfecho_falha_de_provedor():
    aplicacao = AplicacaoFalsa(erro_aprofundamento=RuntimeError("indisponível"))
    teste = abrir_ranking(aplicacao)
    return teste.button(key="aprofundar_1").click().run()


DESFECHOS = (
    desfecho_normal,
    desfecho_non_ai,
    desfecho_evidencia_insuficiente,
    desfecho_sem_resultado,
    desfecho_analise_ausente,
    desfecho_falha_de_provedor,
)


@pytest.mark.parametrize("desfecho", DESFECHOS, ids=lambda f: f.__name__)
def test_cada_desfecho_terminal_renderiza_com_texto_legivel(desfecho):
    teste = desfecho()

    assert not teste.exception
    assert textos(teste).strip()


def test_o_desfecho_sem_resultado_orienta_a_saida_sem_inventar_candidata():
    teste = desfecho_sem_resultado()

    tela = textos(teste)

    assert mensagens.SEM_RESULTADO in tela
    assert mensagens.SEM_RESULTADO_SAIDA in tela
    assert not teste.download_button


def test_o_desfecho_de_falha_e_legivel_e_nao_expoe_rastreamento():
    teste = desfecho_falha_de_provedor()

    tela = textos(teste)

    assert teste.error
    assert mensagens.MENSAGEM_FALHA_APROFUNDAMENTO in tela
    assert mensagens.DIAGNOSTICO_TECNICO in tela
    assert "Traceback" not in tela
    assert not teste.download_button


def test_uma_candidata_fora_da_busca_atual_tem_estado_proprio():
    aplicacao = AplicacaoFalsa()
    teste = abrir_ranking(aplicacao)

    teste.session_state["startup_selecionada"] = 999
    teste.run()

    assert not teste.exception
    assert mensagens.CANDIDATA_FORA_DA_BUSCA in textos(teste)
    assert len(aplicacao.aprofundamentos) == 0


# ----------------------------------------------------------------------
# 5. A fronteira de HTML bruto
# ----------------------------------------------------------------------


def arvore(caminho_relativo: str) -> ast.Module:
    caminho = RAIZ_PROJETO / caminho_relativo
    return ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))


def chamadas_de_html_bruto(modulo: ast.Module) -> list[ast.Call]:
    """Toda chamada que insere HTML: ``st.html`` ou ``unsafe_allow_html=True``."""
    achadas = []
    for no in ast.walk(modulo):
        if not isinstance(no, ast.Call):
            continue
        liberou_html = any(
            palavra.arg == "unsafe_allow_html"
            and isinstance(palavra.value, ast.Constant)
            and palavra.value.value is True
            for palavra in no.keywords
        )
        nome = no.func.attr if isinstance(no.func, ast.Attribute) else ""
        if liberou_html or nome == "html":
            achadas.append(no)
    return achadas


def test_o_html_bruto_da_tela_recebe_apenas_a_folha_de_estilo_estatica():
    chamadas = chamadas_de_html_bruto(arvore("app.py"))

    assert len(chamadas) == 1, "a tela abriu mais de uma fronteira de HTML bruto"
    (chamada,) = chamadas
    assert len(chamada.args) == 1
    argumento = chamada.args[0]
    assert isinstance(argumento, ast.Name), (
        "o HTML bruto só pode receber uma constante nomeada, nunca uma expressão"
    )
    assert argumento.id == "CSS_TEMA"


def test_nenhum_modulo_de_apoio_abre_fronteira_de_html_bruto():
    for caminho in sorted((RAIZ_PROJETO / "radar" / "interface").glob("*.py")):
        modulo = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
        assert chamadas_de_html_bruto(modulo) == [], caminho.name


def test_a_folha_de_estilo_e_um_literal_sem_nenhuma_interpolacao():
    modulo = arvore("radar/interface/tema.py")
    atribuicoes = [
        no
        for no in ast.walk(modulo)
        if isinstance(no, ast.Assign)
        and any(
            isinstance(alvo, ast.Name) and alvo.id == "CSS_TEMA" for alvo in no.targets
        )
    ]

    assert len(atribuicoes) == 1
    valor = atribuicoes[0].value
    assert isinstance(valor, ast.Constant) and isinstance(valor.value, str)


def test_a_folha_de_estilo_nao_carrega_script_nem_recurso_remoto():
    minusculo = CSS_TEMA.casefold()

    assert minusculo.lstrip().startswith("<style>")
    assert minusculo.rstrip().endswith("</style>")
    for proibido in ("<script", "javascript:", "http://", "https://", "@import", "url("):
        assert proibido not in minusculo


def test_a_folha_de_estilo_nunca_vaza_para_o_texto_da_tela():
    teste = abrir_ranking(AplicacaoFalsa())

    assert "st-key-" not in textos(teste)
    assert "<style>" not in textos(teste)


# ----------------------------------------------------------------------
# 6. A tela continua sem decidir nada
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "proibido",
    ["sorted(", ".sort(", "classe_referencia", "import sqlite3", "fit_score_total /"],
)
def test_a_tela_nao_ordena_nao_recalcula_e_nao_consulta_o_banco(proibido):
    fonte = (RAIZ_PROJETO / "app.py").read_text(encoding="utf-8")

    assert proibido not in fonte


# ----------------------------------------------------------------------
# 7. O título e a legenda do ranking descrevem a ordem que a aplicação usa
# ----------------------------------------------------------------------
#
# A ordem real de ``construir_ranking`` é (grupo, -fit_score, bm25, nome, id):
# análise concluída antes de insuficiente/ausente, fit-score NVIDIA decrescente,
# e relevância lexical só como desempate. Chamar isso de "ordem de recuperação"
# contradiz a própria tela, que define recuperação como o eixo BM25 em
# ``EXPLICACAO_BM25``. Estes testes prendem a redação ao algoritmo.


def test_titulo_do_ranking_nao_promete_ordem_de_recuperacao():
    assert mensagens.TITULO_RANKING == "Candidatas priorizadas"
    assert "recupera" not in mensagens.TITULO_RANKING.casefold()


def test_legenda_do_ranking_descreve_a_ordem_real():
    legenda = mensagens.LEGENDA_RANKING.casefold()

    # o critério primário, nomeado como o usuário o vê na tela
    assert "fit-score" in legenda
    # o desempate, nomeado sem se confundir com o critério primário
    assert "desempat" in legenda
    assert "relevância lexical" in legenda
    # o que acontece com quem não tem análise gravada
    assert "depois" in legenda
    # a promessa de honestidade que já existia não pode se perder
    assert "recalcula" in legenda


def test_a_legenda_do_ranking_e_verdadeira_sobre_a_ordem_construida():
    """A promessa da legenda, conferida contra o ranking real.

    O caso é construído para que a empresa lexicalmente mais relevante (BM25
    mais negativo) **não** seja a primeira: se a legenda prometesse ordem de
    recuperação, ela mentiria exatamente aqui.
    """
    from radar.aplicacao import construir_ranking
    from radar.contratos import FiltrosEstruturados, ResultadoRecuperacao
    from tests.apoio_interface import documento_falso, empresa_falsa
    from tests.test_lote_ranking import _analise

    resultado = ResultadoRecuperacao(
        empresas=[
            empresa_falsa(1, "Lexicalmente Melhor"),
            empresa_falsa(2, "Fit Score Melhor"),
            empresa_falsa(3, "Sem Analise"),
        ],
        documentos=[
            documento_falso(10, 1, score_bm25=-9.0),
            documento_falso(20, 2, score_bm25=-0.5),
            documento_falso(30, 3, score_bm25=-9.9),
        ],
        filtros_aplicados=FiltrosEstruturados(),
    )
    ranking = construir_ranking(
        resultado,
        {1: _analise(1, (3, 0, 1, 3)), 2: _analise(2, (5, 4, 5, 5))},
    )

    # 1º: concluída de maior fit-score, apesar do pior BM25 entre as concluídas
    assert ranking[0].empresa.id_startup == 2
    assert ranking[0].melhor_score_bm25 > ranking[1].melhor_score_bm25
    # 2º: a outra concluída
    assert ranking[1].empresa.id_startup == 1
    # 3º: sem análise vem depois, mesmo sendo a mais relevante lexicalmente
    assert ranking[2].empresa.id_startup == 3
    assert ranking[2].status_analise == "ausente"
    assert ranking[2].melhor_score_bm25 == min(
        item.melhor_score_bm25 for item in ranking
    )
