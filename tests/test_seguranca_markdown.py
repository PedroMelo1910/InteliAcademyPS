"""Texto dinâmico e destino de link não podem virar estrutura de Markdown.

Duas superfícies de ataque, o mesmo dever: o rótulo visível e o destino do
link. O contrato de evidência exige que o ``trecho_citado`` seja substring
literal da fonte pública; ele não exige — nem poderia — que a fonte seja
inofensiva em Markdown. A primeira metade ataca tela e arquivo baixado com
colchetes, parênteses, imagem, HTML, crase e destino ``javascript:``.

A segunda metade cobre o destino: escapar o rótulo não basta, porque uma URL
validada pelo Pydantic ainda pode conter parêntese, espaço ou quebra de linha
e, com isso, fechar o destino mais cedo e abrir um segundo link ou uma imagem
remota. O destino é percent-codificado, não escapado como texto visível — ele
precisa continuar clicável.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from radar.interface.exportacao import exportar_briefing_markdown
from radar.interface.texto import destino_markdown, escapar_markdown
from tests.apoio_interface import (
    AplicacaoFalsa,
    CONSULTA_PADRAO,
    aprofundamento_falso,
    descoberta_falsa,
    documento_falso,
    item_concluido_ia,
    montar,
    submeter,
    textos,
)
from tests.conftest import (
    briefing_normal_falso,
    cabecalho_falso,
    citacao_nvidia_falsa,
    fonte_falsa,
    recomendacao_falsa,
)


# ------------------------------------------------------------------------
# O rótulo visível: escapar_markdown
# ------------------------------------------------------------------------

LINK = "[clique aqui](javascript:alert(1))"
IMAGEM = "![logo](http://exemplo.test/pixel.png)"
HTML = "<script>alert(1)</script>"
CRASE = "use `codigo inline` e ```bloco cercado```"
TITULO = "# Título falso injetado"
LISTA = "- item falso injetado"
ENFASE = "*negrito* e _italico_ e |tabela|"

CARGAS = (LINK, IMAGEM, HTML, CRASE, TITULO, LISTA, ENFASE)


# ----------------------------------------------------------------------
# A fronteira de escape em si
# ----------------------------------------------------------------------


@pytest.mark.parametrize("carga", CARGAS)
def test_o_escape_neutraliza_a_sintaxe_sem_perder_a_redacao(carga):
    escapado = escapar_markdown(carga)

    assert escapado != carga
    assert escapado.replace("\\", "") == carga  # a leitura devolve o original


@pytest.mark.parametrize("carga", CARGAS)
def test_o_escape_e_deterministico(carga):
    assert escapar_markdown(carga) == escapar_markdown(carga)


def test_o_escape_impede_link_imagem_html_e_codigo():
    escapado = escapar_markdown(f"{LINK} {IMAGEM} {HTML} {CRASE}")

    assert "](javascript:" not in escapado
    assert "![" not in escapado
    assert "<script>" not in escapado
    assert "```" not in escapado


@pytest.mark.parametrize("carga", (TITULO, LISTA, "1. item ordenado", "=== setext"))
def test_o_escape_impede_abertura_de_bloco_no_inicio_da_linha(carga):
    escapado = escapar_markdown(carga)

    assert not escapado.lstrip().startswith(("#", "-", "+", "=", "1."))


def test_o_escape_preserva_hifen_interno_para_nao_deformar_classes():
    assert escapar_markdown("non-AI tem fit-score zero") == "non-AI tem fit-score zero"


def test_o_escape_preserva_identificadores_de_evidencia():
    assert escapar_markdown("Afirmação 1 documento 2 chunk 101") == (
        "Afirmação 1 documento 2 chunk 101"
    )


# ----------------------------------------------------------------------
# Markdown exportado
# ----------------------------------------------------------------------


def briefing_hostil():
    from radar.contratos import ConclusaoAncorada, VereditoBriefing

    return briefing_normal_falso(
        cabecalho=cabecalho_falso(nome=f"Caju {HTML}", setor=f"Fintech {LINK}"),
        veredito=VereditoBriefing(
            classe="AI-enabled",
            fit_score_total=61,
            tese=f"A empresa usa IA. {LINK}",
            ids_afirmacoes_suporte=[1],
        ),
        sintese_executiva=ConclusaoAncorada(
            texto=f"{TITULO} e depois {IMAGEM}",
            ids_afirmacoes_suporte=[1, 2],
        ),
        pontos_de_conversa=[
            ConclusaoAncorada(texto=f"{LISTA} no roteiro", ids_afirmacoes_suporte=[2]),
            ConclusaoAncorada(texto=f"{CRASE} na conversa", ids_afirmacoes_suporte=[1]),
        ],
        recomendacoes=[
            recomendacao_falsa(
                trecho_citado=f"Segundo a matéria, {LINK} para detalhes.",
                citacao=citacao_nvidia_falsa(
                    101,
                    topico=f"NIM {HTML}",
                    breadcrumb=f"trilha {LINK}",
                ),
            )
        ],
        fontes=[fonte_falsa(titulo=f"Matéria {LINK}")],
        avisos=[f"Aviso com {IMAGEM}"],
    )


def test_o_arquivo_nao_deixa_texto_dinamico_virar_link_ou_imagem():
    markdown = exportar_briefing_markdown(briefing_hostil())

    assert LINK not in markdown
    assert IMAGEM not in markdown
    assert "](javascript:" not in markdown
    assert "![logo]" not in markdown


def test_o_arquivo_nao_deixa_texto_dinamico_virar_html_ou_codigo():
    markdown = exportar_briefing_markdown(briefing_hostil())

    assert "<script>" not in markdown
    assert "</script>" not in markdown
    assert "```" not in markdown


def test_o_arquivo_preserva_a_redacao_original_sob_escape():
    briefing = briefing_hostil()

    markdown = exportar_briefing_markdown(briefing)

    assert escapar_markdown(briefing.veredito.tese) in markdown
    assert escapar_markdown(briefing.sintese_executiva.texto) in markdown
    assert escapar_markdown(briefing.avisos[0]) in markdown
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]
    assert escapar_markdown(evidencia.trecho_citado) in markdown
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]
    assert escapar_markdown(citacao.breadcrumb) in markdown
    assert escapar_markdown(citacao.topico) in markdown


def test_o_arquivo_mantem_as_urls_validadas_como_destino_de_link():
    briefing = briefing_hostil()

    markdown = exportar_briefing_markdown(briefing)

    fonte = briefing.fontes[0]
    assert f"]({fonte.url_fonte})" in markdown
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]
    assert f"]({evidencia.url_fonte})" in markdown
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]
    assert f"]({citacao.fonte_url})" in markdown


def test_o_arquivo_preserva_os_identificadores_de_evidencia():
    markdown = exportar_briefing_markdown(briefing_hostil())

    assert "Afirmação 1 (documento 1)" in markdown
    assert "Chunk 101" in markdown


def test_o_escape_nao_muda_o_determinismo_da_exportacao():
    assert exportar_briefing_markdown(briefing_hostil()) == exportar_briefing_markdown(
        briefing_hostil()
    )


# ----------------------------------------------------------------------
# Tela do Streamlit
# ----------------------------------------------------------------------


def descoberta_hostil():
    item = item_concluido_ia()
    empresa = item.empresa.model_copy(
        update={"descricao_curta": f"Descrição com {LINK} e {HTML}"}
    )
    hostil = replace(
        item,
        empresa=empresa,
        justificativa_fit_score=f"Justificativa com {IMAGEM}",
    )
    return descoberta_falsa(itens=(hostil,))


def test_a_tela_do_ranking_nao_renderiza_texto_dinamico_como_sintaxe():
    teste = montar(AplicacaoFalsa(descoberta=descoberta_hostil()))

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert not teste.exception
    assert LINK not in tela
    assert IMAGEM not in tela
    assert "<script>" not in tela
    assert "](javascript:" not in tela
    assert escapar_markdown(LINK) in tela


def test_a_tela_do_briefing_nao_renderiza_texto_dinamico_como_sintaxe():
    briefing = briefing_hostil()
    aplicacao = AplicacaoFalsa(
        aprofundamento=aprofundamento_falso(1, briefing=briefing)
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert not teste.exception
    assert LINK not in tela
    assert IMAGEM not in tela
    assert "<script>" not in tela
    assert "```" not in tela
    assert "](javascript:" not in tela
    assert escapar_markdown(briefing.veredito.tese) in tela


# ------------------------------------------------------------------------
# O destino do link: destino_markdown
# ------------------------------------------------------------------------

CARGA_BUGBOT = "https://example.com/a)%20![x](https://attacker.example/pixel)"


# ----------------------------------------------------------------------
# O ajudante
# ----------------------------------------------------------------------


def test_a_carga_do_bugbot_nao_consegue_abrir_segundo_link_nem_imagem():
    destino = destino_markdown(CARGA_BUGBOT)

    assert "(" not in destino and ")" not in destino
    assert "![" not in destino
    linha = f"[fonte]({destino})"
    assert linha.count("(") == 1 and linha.count(")") == 1


def test_espacos_e_quebras_de_linha_nao_quebram_o_destino():
    destino = destino_markdown("https://ex.test/a b\nc\r\nd")

    assert " " not in destino
    assert "\n" not in destino and "\r" not in destino
    assert "%20" in destino


def test_parenteses_sao_percent_codificados():
    assert destino_markdown("https://ex.test/a(b)c") == "https://ex.test/a%28b%29c"


def test_angulares_e_barra_invertida_sao_neutralizados():
    destino = destino_markdown("https://ex.test/x<y>z\\w")

    assert "<" not in destino and ">" not in destino and "\\" not in destino


def test_urls_normais_continuam_utilizaveis():
    for url in (
        "https://www.caju.com.br/",
        "http://fonte-a.example/materia",
        "https://nvidia.example/101",
    ):
        assert destino_markdown(url) == url


def test_query_e_fragmento_sao_preservados():
    url = "https://ex.test/caminho?a=1&b=2;c=3#secao-4"

    assert destino_markdown(url) == url


def test_percent_existente_nao_vira_duplo_percent():
    destino = destino_markdown("https://ex.test/a%20b%C3%A9")

    assert destino == "https://ex.test/a%20b%C3%A9"
    assert "%2520" not in destino


def test_o_destino_e_deterministico():
    assert destino_markdown(CARGA_BUGBOT) == destino_markdown(CARGA_BUGBOT)


@pytest.mark.parametrize(
    "hostil", ["javascript:alert(1)", "data:text/html,x", "ftp://ex.test/a"]
)
def test_esquema_fora_de_http_e_recusado(hostil):
    with pytest.raises(ValueError):
        destino_markdown(hostil)


# ----------------------------------------------------------------------
# Markdown exportado
# ----------------------------------------------------------------------


def test_o_arquivo_usa_o_destino_protegido_na_evidencia():
    briefing = briefing_normal_falso(
        recomendacoes=[recomendacao_falsa(url_fonte=CARGA_BUGBOT)]
    )
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]

    markdown = exportar_briefing_markdown(briefing)

    assert f"]({destino_markdown(str(evidencia.url_fonte))})" in markdown
    assert "attacker.example/pixel)" not in markdown


def test_o_arquivo_protege_site_fontes_e_citacoes():
    briefing = briefing_normal_falso()

    markdown = exportar_briefing_markdown(briefing)

    fonte = briefing.fontes[0]
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]
    assert f"]({destino_markdown(str(fonte.url_fonte))})" in markdown
    assert f"]({destino_markdown(str(citacao.fonte_url))})" in markdown


# ----------------------------------------------------------------------
# Tela do Streamlit
# ----------------------------------------------------------------------


def test_documento_recuperado_nao_injeta_link_quando_nao_e_exibido_no_ranking():
    documento = documento_falso(10, 1, url_fonte=CARGA_BUGBOT)
    item = item_concluido_ia()
    from dataclasses import replace

    hostil = replace(item, documentos=(documento,))
    teste = montar(AplicacaoFalsa(descoberta=descoberta_falsa(itens=(hostil,))))

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert not teste.exception
    assert destino_markdown(CARGA_BUGBOT) not in tela
    assert "attacker.example/pixel)" not in tela
    assert "![x]" not in tela
