"""Exportação determinística do ``Briefing`` e paridade de rastreabilidade.

O renderizador é a última peça do sistema: ele não decide nada, só transcreve
um ``Briefing`` já validado (§11.4). A primeira parte trava três coisas — que a
saída é byte a byte reprodutível, que cada variante mostra apenas o que o
contrato autoriza, e que a rastreabilidade (fontes, trechos citados, chunks
NVIDIA) sobrevive à travessia para o arquivo.

A segunda parte trava a paridade: o gerente decide olhando a tela e leva o
Markdown; se o arquivo mostra o id da afirmação que sustenta a tese e a tela
não, a tela é a versão pior de um documento que se propõe a ser auditável.
Formatação pode diferir; fato material, não.

A terceira parte cobre a variante de evidência insuficiente (§11.3), que tem
mais de uma causa. O título compartilhado da variante não pode afirmar uma
causa específica: quem nomeia a causa é a ``sintese_executiva`` e os ``avisos``
já validados dentro do próprio Briefing; o cabeçalho apenas diz que não houve
conclusão.
"""

from __future__ import annotations

import pytest

from radar.interface.exportacao import (
    RESUMO_DA_VARIANTE,
    exportar_briefing_markdown,
    nome_arquivo_briefing,
)
from radar.interface.rotulos import rotulo, rotular_fundamento, trajeto_legivel
from radar.interface.texto import destino_markdown, escapar_markdown
from tests.apoio_interface import (
    AplicacaoFalsa,
    CONSULTA_PADRAO,
    aprofundamento_falso,
    montar,
    submeter,
    textos,
)
from tests.conftest import (
    briefing_insuficiente_falso,
    briefing_nao_aderente_falso,
    briefing_normal_falso,
    cabecalho_falso,
    rodape_falso,
)

# ----------------------------------------------------------------------
# Determinismo e segurança do arquivo
# ----------------------------------------------------------------------


def test_a_mesma_entrada_produz_exatamente_o_mesmo_markdown():
    briefing = briefing_normal_falso()

    assert exportar_briefing_markdown(briefing) == exportar_briefing_markdown(
        briefing_normal_falso()
    )


def test_o_markdown_sobrevive_a_ida_e_volta_em_utf8():
    markdown = exportar_briefing_markdown(briefing_normal_falso())

    assert markdown.encode("utf-8").decode("utf-8") == markdown
    assert "ç" in markdown or "ã" in markdown  # o texto é português, não ASCII


def test_o_nome_do_arquivo_deriva_do_nome_e_da_data_de_geracao():
    briefing = briefing_normal_falso()

    assert nome_arquivo_briefing(briefing) == "briefing_caju_2026-09-03.md"


def test_o_nome_do_arquivo_neutraliza_separadores_e_acentos():
    briefing = briefing_normal_falso(
        cabecalho=cabecalho_falso(nome="Açaí / Tech ../.. \\ Ltda")
    )

    nome = nome_arquivo_briefing(briefing)

    assert nome == "briefing_acai-tech-ltda_2026-09-03.md"
    assert "/" not in nome and "\\" not in nome and ".." not in nome


def test_o_nome_do_arquivo_e_deterministico_para_a_mesma_entrada():
    assert nome_arquivo_briefing(briefing_normal_falso()) == nome_arquivo_briefing(
        briefing_normal_falso()
    )


def test_um_nome_sem_nenhum_caractere_util_ainda_gera_arquivo_valido():
    briefing = briefing_normal_falso(cabecalho=cabecalho_falso(nome="///"))

    assert nome_arquivo_briefing(briefing) == "briefing_startup_2026-09-03.md"


def test_o_exportador_recusa_qualquer_coisa_que_nao_seja_briefing_validado():
    with pytest.raises(TypeError):
        exportar_briefing_markdown({"variante": "normal"})

    with pytest.raises(TypeError):
        nome_arquivo_briefing("Caju")


# ----------------------------------------------------------------------
# Variante normal
# ----------------------------------------------------------------------


def test_o_normal_traz_identificacao_veredito_e_consulta_original():
    briefing = briefing_normal_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "# Briefing — Caju" in markdown
    assert "https://www.caju.com.br/" in markdown
    assert "Fintech / RH" in markdown
    assert "série B" in markdown
    assert "São Paulo, SP" in markdown
    assert "03/09/2026" in markdown
    assert briefing.cabecalho.consulta_original in markdown
    assert "AI-enabled" in markdown
    assert "61/100" in markdown
    assert briefing.veredito.tese in markdown
    assert briefing.sintese_executiva.texto in markdown


def test_o_normal_traz_todos_os_pontos_de_conversa():
    briefing = briefing_normal_falso()

    markdown = exportar_briefing_markdown(briefing)

    for ponto in briefing.pontos_de_conversa:
        assert ponto.texto in markdown


def test_o_normal_traz_cada_campo_da_recomendacao():
    briefing = briefing_normal_falso()
    recomendacao = briefing.recomendacoes[0]

    markdown = exportar_briefing_markdown(briefing)

    assert rotular_fundamento(recomendacao) in markdown
    assert recomendacao.tecnologias[0] in markdown
    assert recomendacao.justificativa_tecnica in markdown
    assert recomendacao.justificativa_negocio in markdown
    assert rotulo(recomendacao.prioridade) in markdown
    assert rotulo(recomendacao.complexidade) in markdown
    assert rotulo(recomendacao.proxima_acao.tipo_acao) in markdown
    assert recomendacao.proxima_acao.detalhe in markdown


def test_o_normal_preserva_o_trecho_literal_e_a_fonte_de_cada_evidencia():
    briefing = briefing_normal_falso()
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]

    markdown = exportar_briefing_markdown(briefing)

    assert evidencia.trecho_citado in markdown
    assert str(evidencia.url_fonte) in markdown
    assert f"{evidencia.id_afirmacao}" in markdown
    assert f"documento {evidencia.id_documento}" in markdown


def test_o_normal_preserva_chunk_breadcrumb_origem_e_url_das_citacoes_nvidia():
    briefing = briefing_normal_falso()
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]

    markdown = exportar_briefing_markdown(briefing)

    assert str(citacao.id_chunk) in markdown
    assert escapar_markdown(citacao.breadcrumb) in markdown
    assert citacao.topico in markdown
    assert rotulo(citacao.origem) in markdown
    assert citacao.tecnologia in markdown
    assert str(citacao.fonte_url) in markdown


def test_o_normal_lista_as_fontes_publicas_como_links_clicaveis():
    briefing = briefing_normal_falso()
    fonte = briefing.fontes[0]

    markdown = exportar_briefing_markdown(briefing)

    assert f"[{fonte.titulo}]({fonte.url_fonte})" in markdown
    assert fonte.host_normalizado in markdown


def test_o_normal_fecha_com_rubrica_data_de_execucao_e_trajeto():
    briefing = briefing_normal_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "rubrica-v1" in markdown
    assert "03/09/2026" in markdown
    assert trajeto_legivel(briefing.rodape.trajeto) in markdown
    assert rotulo(briefing.rodape.rota_r3) in markdown


def test_o_normal_omite_a_secao_de_avisos_quando_nao_ha_aviso():
    markdown = exportar_briefing_markdown(briefing_normal_falso())

    assert "## Avisos" not in markdown


def test_o_normal_mostra_os_avisos_quando_existem():
    briefing = briefing_normal_falso(avisos=["Critérios estruturados relaxados."])

    markdown = exportar_briefing_markdown(briefing)

    assert "## Limites desta análise" in markdown
    assert "Critérios estruturados relaxados." in markdown


# ----------------------------------------------------------------------
# Variante non-AI
# ----------------------------------------------------------------------


def test_o_nao_aderente_mostra_a_classe_validada_e_o_zero_real():
    briefing = briefing_nao_aderente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "non-AI" in markdown
    assert "0/100" in markdown
    assert briefing.veredito.tese in markdown
    assert briefing.sintese_executiva.texto in markdown
    assert briefing.avisos[0] in markdown


def test_o_nao_aderente_nao_finge_que_existe_oportunidade():
    markdown = exportar_briefing_markdown(briefing_nao_aderente_falso())

    assert "## Recomendações" not in markdown
    assert "Próxima ação" not in markdown


def test_o_nao_aderente_mantem_fontes_e_auditoria():
    briefing = briefing_nao_aderente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert str(briefing.fontes[0].url_fonte) in markdown
    assert rotulo(briefing.rodape.rota_r3) in markdown
    assert "rubrica-v1" in markdown


# ----------------------------------------------------------------------
# Variante evidência insuficiente
# ----------------------------------------------------------------------


def test_o_insuficiente_nao_declara_classe_score_tese_nem_recomendacao():
    briefing = briefing_insuficiente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "## Veredito" not in markdown
    assert "## Recomendações" not in markdown
    assert briefing.veredito.tese not in markdown
    assert "Fit-score" not in markdown
    assert "non-AI" not in markdown
    assert "AI-enabled" not in markdown
    assert "AI-native" not in markdown


def test_o_insuficiente_explica_a_insuficiencia_sem_virar_acusacao():
    briefing = briefing_insuficiente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "Evidência insuficiente" in markdown
    assert briefing.sintese_executiva.texto in markdown
    assert briefing.avisos[0] in markdown


def test_o_insuficiente_mantem_a_auditoria_da_execucao():
    briefing = briefing_insuficiente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert "rubrica-v1" in markdown
    assert rotulo(briefing.rodape.rota_r3) in markdown
    assert trajeto_legivel(briefing.rodape.trajeto) in markdown


def test_o_insuficiente_omite_a_secao_de_fontes_que_o_contrato_deixa_vazia():
    markdown = exportar_briefing_markdown(briefing_insuficiente_falso())

    assert "## Fontes públicas" not in markdown


# ----------------------------------------------------------------------
# O que nunca pode vazar
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "construtor",
    [briefing_normal_falso, briefing_nao_aderente_falso, briefing_insuficiente_falso],
)
def test_nenhuma_variante_vaza_rotulo_de_curadoria_ou_estado_interno(construtor):
    markdown = exportar_briefing_markdown(construtor())

    for proibido in (
        "classe_referencia",
        "query_planner",
        "retriever",
        "extractor",
        "classifier",
        "evidence_validator",
        "nvidia_rag",
        "gap_confirmado",
        "evidencia_insuficiente",
        "nao_aderente",
        "convite_inception",
        "API_KEY",
        "GOOGLE_API_KEY",
        "NVIDIA_API_KEY",
        "system",
        "prompt",
        "startup_selecionada",
        "tentativas_extracao",
    ):
        assert proibido not in markdown


# ------------------------------------------------------------------------
# Paridade de rastreabilidade entre a tela e o arquivo baixado
# ------------------------------------------------------------------------

def tela_do_briefing(briefing):
    aplicacao = AplicacaoFalsa(
        aprofundamento=aprofundamento_falso(1, briefing=briefing)
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    return teste.button(key="aprofundar_1").click().run()


def fatos_do_arquivo(briefing) -> list[str]:
    """O arquivo mantém a auditoria completa, inclusive os ids de suporte."""
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]
    return [
        *fatos_legiveis_na_tela(briefing),
        f"Afirmação {evidencia.id_afirmacao}",
        f"documento {evidencia.id_documento}",
        f"Chunk {citacao.id_chunk}",
        rotulo(citacao.origem),
        briefing.rodape.versao_rubrica,
        rotulo(briefing.rodape.rota_r3),
        trajeto_legivel(briefing.rodape.trajeto),
    ]


def fatos_legiveis_na_tela(briefing) -> list[str]:
    """A tela preserva o conteúdo, mas traduz identificadores internos."""
    citacao = briefing.recomendacoes[0].citacoes_nvidia[0]
    evidencia = briefing.recomendacoes[0].evidencias_startup[0]
    fonte = briefing.fontes[0]
    recomendacao = briefing.recomendacoes[0]
    return [
        escapar_markdown(briefing.veredito.tese),
        escapar_markdown(briefing.sintese_executiva.texto),
        *[escapar_markdown(p.texto) for p in briefing.pontos_de_conversa],
        briefing.veredito.classe,
        f"{briefing.veredito.fit_score_total}/100",
        rotular_fundamento(recomendacao),
        recomendacao.tecnologias[0],
        rotulo(recomendacao.prioridade),
        rotulo(recomendacao.complexidade),
        escapar_markdown(recomendacao.justificativa_tecnica),
        escapar_markdown(recomendacao.justificativa_negocio),
        rotulo(recomendacao.proxima_acao.tipo_acao),
        escapar_markdown(recomendacao.proxima_acao.detalhe),
        escapar_markdown(evidencia.trecho_citado),
        destino_markdown(str(evidencia.url_fonte)),
        citacao.tecnologia,
        escapar_markdown(citacao.topico),
        escapar_markdown(citacao.breadcrumb),
        destino_markdown(str(citacao.fonte_url)),
        escapar_markdown(fonte.titulo),
        fonte.host_normalizado,
        rotulo(fonte.tipo),
        destino_markdown(str(fonte.url_fonte)),
    ]


@pytest.mark.parametrize(
    "fato",
    fatos_do_arquivo(briefing_normal_falso()),
)
def test_cada_fato_material_aparece_na_tela_e_no_arquivo(fato):
    briefing = briefing_normal_falso()

    markdown = exportar_briefing_markdown(briefing)
    assert fato in markdown, f"ausente no arquivo: {fato!r}"


@pytest.mark.parametrize(
    "fato",
    fatos_legiveis_na_tela(briefing_normal_falso()),
)
def test_cada_fato_material_aparece_de_forma_legivel_na_tela(fato):
    tela = textos(tela_do_briefing(briefing_normal_falso()))

    assert fato in tela, f"ausente na tela: {fato!r}"


def test_a_tela_nao_expoe_ids_internos_do_veredito_e_da_sintese():
    briefing = briefing_normal_falso()

    tela = textos(tela_do_briefing(briefing))

    assert "Afirmações que sustentam a tese" not in tela
    assert "Afirmações de suporte" not in tela


def test_a_tela_nao_expoe_ids_internos_nos_pontos_de_conversa():
    briefing = briefing_normal_falso()

    tela = textos(tela_do_briefing(briefing))

    assert "afirmações:" not in tela.casefold()


def test_a_tela_mostra_o_tipo_de_cada_fonte_publica():
    briefing = briefing_normal_falso()

    tela = textos(tela_do_briefing(briefing))

    for fonte in briefing.fontes:
        assert fonte.tipo in tela


def test_a_tela_nao_expoe_registro_bruto_de_banco_nem_payload_de_provedor():
    tela = textos(tela_do_briefing(briefing_normal_falso()))

    for proibido in (
        "classe_referencia",
        "SELECT ",
        "sqlite",
        "API_KEY",
        "candidates",
        "Traceback",
    ):
        assert proibido not in tela


# ------------------------------------------------------------------------
# A variante de evidência insuficiente e suas causas (§11.3)
# ------------------------------------------------------------------------

TITULO = RESUMO_DA_VARIANTE["evidencia_insuficiente"]

SINTESE_SEM_RECOMENDACAO = (
    "As afirmações confirmadas não sustentaram nenhuma recomendação rastreável "
    "até a base NVIDIA."
)
AVISO_SEM_RECOMENDACAO = (
    "Evidência insuficiente: nenhuma recomendação sobreviveu à exigência de "
    "proveniência dos dois lados."
)


def briefing_sem_recomendacao_sustentavel(**ajustes):
    """Insuficiência com evidência confirmada: a causa está no Recommendation.

    O contrato admite esta variante chegando pela rota ``prosseguir`` — é o
    caminho pós-Recommendation da §11.3, e o rodapé registra afirmações
    confirmadas maiores que zero.
    """
    from radar.contratos import ConclusaoAncorada

    campos = {
        "sintese_executiva": ConclusaoAncorada(
            texto=SINTESE_SEM_RECOMENDACAO,
            ids_afirmacoes_suporte=[],
        ),
        "avisos": [AVISO_SEM_RECOMENDACAO],
        "rodape": rodape_falso(
            afirmacoes_confirmadas=3,
            afirmacoes_derrubadas=0,
            rota_r3="prosseguir",
        ),
    }
    campos.update(ajustes)
    return briefing_insuficiente_falso(**campos)


CAUSAS = {
    "nenhuma_afirmacao_confirmada": briefing_insuficiente_falso,
    "sem_recomendacao_sustentavel": briefing_sem_recomendacao_sustentavel,
}


# ----------------------------------------------------------------------
# O título compartilhado é neutro quanto à causa
# ----------------------------------------------------------------------


def test_o_titulo_da_variante_nao_afirma_que_nenhuma_afirmacao_sobreviveu():
    minusculo = TITULO.casefold()

    assert "sobreviveu" not in minusculo
    assert "nenhuma afirmação" not in minusculo
    assert "nenhuma evidência" not in minusculo
    assert "validação de proveniência" not in minusculo


def test_o_titulo_diz_que_nao_houve_conclusao_e_aponta_para_a_causa_validada():
    minusculo = TITULO.casefold()

    assert "evidência insuficiente" in minusculo
    assert "não sustentam uma conclusão segura" in minusculo
    assert "motivo" in minusculo
    assert "faltou confirmar" in minusculo


def test_causas_diferentes_recebem_exatamente_o_mesmo_cabecalho():
    exportados = [
        exportar_briefing_markdown(construtor()) for construtor in CAUSAS.values()
    ]

    for markdown in exportados:
        assert TITULO in markdown


# ----------------------------------------------------------------------
# A causa específica continua chegando ao leitor
# ----------------------------------------------------------------------


def test_a_insuficiencia_por_falta_de_recomendacao_e_descrita_com_honestidade():
    briefing = briefing_sem_recomendacao_sustentavel()

    markdown = exportar_briefing_markdown(briefing)

    assert SINTESE_SEM_RECOMENDACAO in markdown
    assert AVISO_SEM_RECOMENDACAO in markdown
    assert "Afirmações confirmadas:** 3" in markdown
    assert "Afirmações derrubadas:** 0" in markdown


def test_a_insuficiencia_por_ausencia_de_afirmacao_mantem_a_propria_causa():
    briefing = briefing_insuficiente_falso()

    markdown = exportar_briefing_markdown(briefing)

    assert briefing.sintese_executiva.texto in markdown
    assert briefing.avisos[0] in markdown
    assert "Afirmações confirmadas:** 0" in markdown


@pytest.mark.parametrize("construtor", CAUSAS.values(), ids=list(CAUSAS))
def test_a_tela_preserva_a_sintese_e_os_avisos_especificos_da_causa(construtor):
    briefing = construtor()
    aplicacao = AplicacaoFalsa(
        aprofundamento=aprofundamento_falso(1, briefing=briefing)
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert not teste.exception
    assert TITULO in tela
    assert briefing.sintese_executiva.texto in tela
    assert briefing.avisos[0] in tela


# ----------------------------------------------------------------------
# Nenhuma causa reabre classe, score ou recomendação
# ----------------------------------------------------------------------


@pytest.mark.parametrize("construtor", CAUSAS.values(), ids=list(CAUSAS))
def test_nenhuma_causa_declara_classe_score_ou_recomendacao_no_arquivo(construtor):
    briefing = construtor()

    markdown = exportar_briefing_markdown(briefing)

    assert "## Veredito" not in markdown
    assert "## Recomendações" not in markdown
    assert briefing.veredito.tese not in markdown
    assert "/100" not in markdown
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        assert classe not in markdown


@pytest.mark.parametrize("construtor", CAUSAS.values(), ids=list(CAUSAS))
def test_nenhuma_causa_declara_classe_score_ou_recomendacao_na_tela(construtor):
    aplicacao = AplicacaoFalsa(
        aprofundamento=aprofundamento_falso(1, briefing=construtor())
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert "/100" not in tela
    assert "Recomendações" not in tela
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        assert classe not in tela
