"""Jornada da interface, exercitada pelo ``AppTest`` do Streamlit.

Nenhum teste aqui toca rede, SQLite ou provedor: a fronteira da aplicação é
injetada em ``session_state`` e o script roda em processo. O que se prova é a
jornada — busca explícita, ranking preservado entre reruns, seleção que chama a
fronteira com a descoberta corrente e briefing que nunca é fabricado.

A segunda parte cobre a memória entre telas quando há várias candidatas e
várias buscas: cada candidata guarda o próprio resultado, a falha de uma não
contamina o briefing de outra, uma busca nova não deixa nada da anterior vazar
e nenhum rerun repete trabalho já feito.

A terceira parte cobre o cartão do ranking depois da análise. A descoberta é
uma fotografia do cache no instante da busca; se o usuário aprofunda uma
candidata marcada como ``ausente`` e volta para a lista, insistir na fotografia
antiga é mentir para ele. A atualização vive só na sessão: não reordena, não
recalcula fit-score e não escreve no SQLite.
"""

from __future__ import annotations

import pytest

from radar.agentes.query_planner import ErroQueryPlanner
from radar.configuracao import ErroConfiguracao, RAIZ_PROJETO
from radar.interface import estado as sessao
from radar.interface.texto import escapar_markdown
from tests.apoio_interface import (
    AplicacaoFalsa,
    CONSULTA_PADRAO,
    ROTULO_BUSCAR,
    abrir_ranking,
    aprofundamento_falso,
    briefing_na_tela,
    chamadas_por_startup,
    descoberta_falsa,
    item_ausente,
    item_concluido_ia,
    item_concluido_non_ai,
    item_insuficiente,
    montar,
    submeter,
    textos,
)
from tests.conftest import (
    briefing_insuficiente_falso,
    briefing_nao_aderente_falso,
    briefing_normal_falso,
    cabecalho_falso,
)

# ----------------------------------------------------------------------
# Busca explícita
# ----------------------------------------------------------------------


def test_a_tela_abre_sem_executar_descoberta_nenhuma():
    aplicacao = AplicacaoFalsa()

    teste = montar(aplicacao)

    assert not teste.exception
    assert aplicacao.consultas == []


def test_digitar_sem_submeter_nao_dispara_a_descoberta():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)

    teste.text_input(key="consulta").input(CONSULTA_PADRAO).run()

    assert aplicacao.consultas == []


def test_a_submissao_explicita_dispara_a_descoberta_uma_vez():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)

    submeter(teste, CONSULTA_PADRAO)

    assert aplicacao.consultas == [CONSULTA_PADRAO]
    assert not teste.exception


def test_consulta_em_branco_avisa_e_nao_chama_a_aplicacao():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)

    submeter(teste, "   ")

    assert aplicacao.consultas == []
    assert teste.warning


def test_a_legenda_obsoleta_sobre_o_fit_score_nao_existe_mais():
    teste = montar(AplicacaoFalsa())

    assert "ainda não é o fit-score" not in textos(teste)


# ----------------------------------------------------------------------
# Ranking preservado, ordenado e honesto
# ----------------------------------------------------------------------


def test_a_descoberta_sobrevive_a_reruns_sem_reexecutar():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.run()
    teste.run()

    assert aplicacao.consultas == [CONSULTA_PADRAO]
    assert "Maritaca AI" in textos(teste)


def test_o_ranking_sai_na_ordem_devolvida_pela_aplicacao():
    itens = (
        item_ausente(posicao=1, id_startup=4, nome="Zeta"),
        item_concluido_ia(posicao=2, id_startup=1, nome="Alfa"),
    )
    teste = montar(AplicacaoFalsa(descoberta=descoberta_falsa(itens=itens)))

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert tela.index("Zeta") < tela.index("Alfa")


def test_os_quatro_estados_de_analise_aparecem_com_leituras_distintas():
    teste = montar(AplicacaoFalsa())

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert "72/100" in tela  # concluída aderente
    assert "0/100" in tela  # concluída non-AI, zero validado
    assert item_insuficiente().motivo_evidencia_insuficiente in tela
    assert "AI-native" in tela and "non-AI" in tela


def test_a_candidata_sem_analise_persistida_nao_ganha_zero_nem_classe():
    itens = (item_ausente(posicao=1, id_startup=4, nome="Mombak"),)
    teste = montar(AplicacaoFalsa(descoberta=descoberta_falsa(itens=itens)))

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert "Mombak" in tela
    assert "0/100" not in tela
    assert "AI-native" not in tela and "non-AI" not in tela


def test_o_bm25_aparece_como_informacao_secundaria_e_explicada():
    teste = montar(AplicacaoFalsa())

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert "BM25" in tela
    assert "relevância textual" in tela
    assert "não entra no fit-score" in tela


def test_o_estado_vazio_e_explicito_e_nao_inventa_candidata():
    teste = montar(AplicacaoFalsa(descoberta=descoberta_falsa(itens=())))

    submeter(teste, CONSULTA_PADRAO)

    assert teste.warning
    assert not [botao for botao in teste.button if botao.label != ROTULO_BUSCAR]


def test_o_relaxamento_de_criterios_e_anunciado_ao_usuario():
    descoberta = descoberta_falsa(
        criterios_relaxados=("setor", "estagio"), tentativas_relaxamento=2
    )
    teste = montar(AplicacaoFalsa(descoberta=descoberta))

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert "setor" in tela and "estagio" in tela
    assert teste.info


def test_toda_candidata_tem_acao_habilitada_de_aprofundamento():
    descoberta = descoberta_falsa()
    teste = montar(AplicacaoFalsa(descoberta=descoberta))

    submeter(teste, CONSULTA_PADRAO)

    for item in descoberta.ranking:
        botao = teste.button(key=f"aprofundar_{item.empresa.id_startup}")
        assert botao.disabled is False


# ----------------------------------------------------------------------
# Seleção e execução única do aprofundamento
# ----------------------------------------------------------------------


def test_clicar_numa_candidata_chama_a_fronteira_com_a_descoberta_corrente():
    descoberta = descoberta_falsa()
    aplicacao = AplicacaoFalsa(descoberta=descoberta)
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.button(key="aprofundar_2").click().run()

    assert len(aplicacao.aprofundamentos) == 1
    descoberta_usada, id_usado = aplicacao.aprofundamentos[0]
    assert descoberta_usada is descoberta
    assert id_usado == 2
    assert not teste.exception


def test_reruns_nao_repetem_um_aprofundamento_ja_concluido():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_1").click().run()

    teste.run()
    teste.run()

    assert len(aplicacao.aprofundamentos) == 1


def test_a_descoberta_continua_disponivel_enquanto_o_briefing_e_exibido():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_1").click().run()

    teste.button(key="voltar_para_candidatas").click().run()

    assert aplicacao.consultas == [CONSULTA_PADRAO]
    assert "Maritaca AI" in textos(teste)


def test_uma_nova_busca_apaga_a_selecao_e_o_briefing_anteriores():
    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_1").click().run()
    assert teste.download_button

    submeter(teste, "outra consulta completamente diferente")

    assert not teste.download_button
    assert sessao.CHAVE_SELECIONADA not in teste.session_state
    assert "Maritaca AI" in textos(teste)


# ----------------------------------------------------------------------
# Falha segura
# ----------------------------------------------------------------------


def test_uma_falha_no_aprofundamento_nao_fabrica_briefing():
    aplicacao = AplicacaoFalsa(
        erro_aprofundamento=RuntimeError("chave-secreta-do-provedor vazou aqui")
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)

    teste.button(key="aprofundar_1").click().run()

    assert teste.error
    assert not teste.download_button
    tela = textos(teste)
    assert "chave-secreta-do-provedor" not in tela
    assert "Tese" not in tela


def test_uma_falha_nao_e_reexecutada_a_cada_rerun():
    aplicacao = AplicacaoFalsa(erro_aprofundamento=RuntimeError("indisponível"))
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_1").click().run()

    teste.run()
    teste.run()

    assert len(aplicacao.aprofundamentos) == 1


def test_uma_falha_na_descoberta_nao_deixa_ranking_antigo_na_tela():
    class AplicacaoQueFalha(AplicacaoFalsa):
        def executar_descoberta(self, consulta):
            self.consultas.append(consulta)
            raise RuntimeError("falha simulada da descoberta")

    aplicacao = AplicacaoFalsa()
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    assert "Maritaca AI" in textos(teste)

    teste.session_state[sessao.CHAVE_APLICACAO] = AplicacaoQueFalha()
    submeter(teste, "consulta que quebra")

    assert teste.error
    assert "Maritaca AI" not in textos(teste)


# ----------------------------------------------------------------------
# As três variantes do briefing na tela
# ----------------------------------------------------------------------


def test_o_briefing_normal_mostra_veredito_recomendacoes_e_rastreabilidade():
    briefing = briefing_normal_falso()

    teste = briefing_na_tela(briefing)
    tela = textos(teste)

    assert not teste.exception
    assert briefing.veredito.tese in tela
    assert "61/100" in tela
    assert "AI-enabled" in tela
    assert briefing.sintese_executiva.texto in tela
    assert briefing.pontos_de_conversa[0].texto in tela
    recomendacao = briefing.recomendacoes[0]
    assert recomendacao.justificativa_tecnica in tela
    assert recomendacao.justificativa_negocio in tela
    assert recomendacao.evidencias_startup[0].trecho_citado in tela
    assert escapar_markdown(recomendacao.citacoes_nvidia[0].breadcrumb) in tela
    assert str(briefing.fontes[0].url_fonte) in tela
    assert teste.download_button


def test_o_briefing_nao_aderente_mostra_o_zero_sem_prometer_oportunidade():
    briefing = briefing_nao_aderente_falso()

    teste = briefing_na_tela(briefing)
    tela = textos(teste)

    assert not teste.exception
    assert "non-AI" in tela
    assert "0/100" in tela
    assert briefing.avisos[0] in tela
    assert "Recomendações" not in tela
    assert "insuficiente" not in tela.casefold()


def test_o_briefing_de_evidencia_insuficiente_nao_declara_classe_nem_score():
    briefing = briefing_insuficiente_falso()

    teste = briefing_na_tela(briefing)
    tela = textos(teste)

    assert not teste.exception
    assert briefing.sintese_executiva.texto in tela
    assert briefing.avisos[0] in tela
    assert "Recomendações" not in tela
    assert "/100" not in tela
    assert "AI-native" not in tela and "AI-enabled" not in tela
    assert "non-AI" not in tela


@pytest.mark.parametrize(
    "construtor",
    [briefing_normal_falso, briefing_nao_aderente_falso, briefing_insuficiente_falso],
)
def test_toda_variante_oferece_download_e_nenhuma_quebra_por_campo_ausente(construtor):
    teste = briefing_na_tela(construtor())

    assert not teste.exception
    assert teste.download_button
    assert teste.download_button[0].disabled is False


def test_o_download_recebe_exatamente_o_briefing_exibido(monkeypatch):
    from radar.interface import exportacao

    capturados = []
    original = exportacao.exportar_briefing_markdown

    def espiao(briefing):
        capturados.append(briefing)
        return original(briefing)

    monkeypatch.setattr(exportacao, "exportar_briefing_markdown", espiao)
    briefing = briefing_normal_falso()

    briefing_na_tela(briefing)

    assert capturados and capturados[-1] == briefing


# ----------------------------------------------------------------------
# Conteúdo gerado nunca é HTML confiável
# ----------------------------------------------------------------------


def test_a_interface_nunca_habilita_html_bruto():
    fonte = (RAIZ_PROJETO / "app.py").read_text(encoding="utf-8")

    assert "unsafe_allow_html" not in fonte


# ------------------------------------------------------------------------
# Várias candidatas, várias buscas e falhas parciais
# ------------------------------------------------------------------------

TESE_A = "A candidata A tem lacuna de distribuição comprovada."
TESE_B = "A candidata B tem lacuna de otimização técnica comprovada."


def briefing_de(nome: str, tese: str):
    from radar.contratos import VereditoBriefing

    return briefing_normal_falso(
        cabecalho=cabecalho_falso(nome=nome),
        veredito=VereditoBriefing(
            classe="AI-enabled",
            fit_score_total=61,
            tese=tese,
            ids_afirmacoes_suporte=[1],
        ),
    )


def aplicacao_com_duas_candidatas(**ajustes) -> AplicacaoFalsa:
    padrao = {
        "descoberta": descoberta_falsa(
            itens=(item_concluido_ia(), item_concluido_non_ai())
        ),
        "aprofundamentos_por_startup": {
            1: aprofundamento_falso(1, briefing=briefing_de("Candidata A", TESE_A)),
            2: aprofundamento_falso(2, briefing=briefing_de("Candidata B", TESE_B)),
        },
    }
    padrao.update(ajustes)
    return AplicacaoFalsa(**padrao)


# ----------------------------------------------------------------------
# 1. Ida e volta entre candidatas
# ----------------------------------------------------------------------


def test_ir_e_voltar_entre_candidatas_mantem_cada_resultado_isolado():
    aplicacao = aplicacao_com_duas_candidatas()
    teste = abrir_ranking(aplicacao)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)
    assert TESE_A in tela and TESE_B not in tela

    teste.button(key="voltar_para_candidatas").click().run()
    teste.button(key="aprofundar_2").click().run()
    tela = textos(teste)
    assert TESE_B in tela and TESE_A not in tela

    teste.button(key="voltar_para_candidatas").click().run()
    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert not teste.exception
    assert TESE_A in tela and TESE_B not in tela
    assert chamadas_por_startup(aplicacao) == [1, 2]  # a volta não recalculou


# ----------------------------------------------------------------------
# 2. Falha de uma candidata não contamina a outra
# ----------------------------------------------------------------------


def test_a_falha_de_uma_candidata_nao_contamina_o_briefing_da_outra():
    aplicacao = aplicacao_com_duas_candidatas(
        erros_por_startup={2: RuntimeError("indisponível")}
    )
    teste = abrir_ranking(aplicacao)
    teste.button(key="aprofundar_1").click().run()
    assert TESE_A in textos(teste)

    teste.button(key="voltar_para_candidatas").click().run()
    teste.button(key="aprofundar_2").click().run()
    tela = textos(teste)

    assert teste.error
    assert not teste.download_button
    assert TESE_A not in tela and TESE_B not in tela

    teste.button(key="voltar_para_candidatas").click().run()
    teste.button(key="aprofundar_1").click().run()

    assert TESE_A in textos(teste)
    assert teste.download_button
    assert chamadas_por_startup(aplicacao) == [1, 2]


# ----------------------------------------------------------------------
# 3. Mesmo id de startup depois de uma busca nova
# ----------------------------------------------------------------------


def test_reusar_o_mesmo_id_apos_nova_busca_nao_expoe_o_resultado_anterior():
    primeira = descoberta_falsa(
        itens=(item_concluido_ia(1, 1, "Alfa"),), consulta="primeira consulta"
    )
    segunda = descoberta_falsa(
        itens=(item_concluido_ia(1, 1, "Beta"),), consulta="segunda consulta"
    )
    aplicacao = AplicacaoFalsa(descobertas=[primeira, segunda])
    teste = abrir_ranking(aplicacao)
    teste.button(key="aprofundar_1").click().run()
    assert teste.download_button

    submeter(teste, "segunda consulta bem diferente")
    tela = textos(teste)

    assert not teste.download_button
    assert "Beta" in tela and "Alfa" not in tela
    assert sessao.startup_selecionada(teste.session_state) is None

    teste.button(key="aprofundar_1").click().run()

    assert not teste.exception
    assert len(aplicacao.aprofundamentos) == 2  # nada foi reaproveitado
    assert aplicacao.aprofundamentos[1][0] is segunda


# ----------------------------------------------------------------------
# 4. Busca nova que falha não deixa resíduo
# ----------------------------------------------------------------------


def test_uma_busca_nova_que_falha_nao_deixa_ranking_nem_briefing_antigos():
    primeira = descoberta_falsa(itens=(item_concluido_ia(1, 1, "Alfa"),))
    aplicacao = AplicacaoFalsa(
        descobertas=[primeira, RuntimeError("falha simulada da descoberta")],
        aprofundamentos_por_startup={
            1: aprofundamento_falso(1, briefing=briefing_de("Alfa", TESE_A))
        },
    )
    teste = abrir_ranking(aplicacao)
    teste.button(key="aprofundar_1").click().run()
    assert TESE_A in textos(teste)

    submeter(teste, "consulta que quebra")
    tela = textos(teste)

    assert teste.error
    assert "Alfa" not in tela
    assert TESE_A not in tela
    assert not teste.download_button
    assert sessao.descoberta_atual(teste.session_state) is None
    assert sessao.startup_selecionada(teste.session_state) is None


# ----------------------------------------------------------------------
# 5. Nada repete depois de uma análise concluída
# ----------------------------------------------------------------------


def test_reruns_download_e_navegacao_nao_repetem_a_analise_concluida():
    aplicacao = aplicacao_com_duas_candidatas()
    teste = abrir_ranking(aplicacao)
    teste.button(key="aprofundar_1").click().run()

    teste.run()
    teste.run()
    teste.download_button[0].click().run()
    teste.button(key="voltar_para_candidatas").click().run()
    teste.button(key="aprofundar_1").click().run()

    assert not teste.exception
    assert chamadas_por_startup(aplicacao) == [1]
    assert aplicacao.consultas == [CONSULTA_PADRAO]
    assert TESE_A in textos(teste)


# ----------------------------------------------------------------------
# 6. Rótulos de ação identificam a candidata
# ----------------------------------------------------------------------


def test_os_rotulos_de_aprofundamento_sao_unicos_e_nomeiam_a_startup():
    descoberta = descoberta_falsa()
    teste = abrir_ranking(AplicacaoFalsa(descoberta=descoberta))

    rotulos = [
        teste.button(key=f"aprofundar_{item.empresa.id_startup}").label
        for item in descoberta.ranking
    ]

    assert len(set(rotulos)) == len(rotulos)
    for item, rotulo in zip(descoberta.ranking, rotulos, strict=True):
        assert item.empresa.nome in rotulo
    assert ROTULO_BUSCAR not in rotulos


# ----------------------------------------------------------------------
# 7 a 9. Falhas mostram só a mensagem segura
# ----------------------------------------------------------------------


SEGREDO = "AIzaSyA-chave-secreta-do-provedor"
PAYLOAD = "resposta bruta do provedor: {'candidates': [{'content': 'x'}]}"


def com_causa(erro: Exception) -> Exception:
    """Anexa uma causa hostil, como o `raise ... from ...` do runtime faria."""
    erro.__cause__ = RuntimeError(f"{PAYLOAD} — {SEGREDO}")
    return erro


def test_erro_de_configuracao_na_busca_mostra_apenas_a_mensagem_segura():
    mensagem = "GOOGLE_API_KEY não está configurada no .env local."
    aplicacao = AplicacaoFalsa(descobertas=[com_causa(ErroConfiguracao(mensagem))])
    teste = montar(aplicacao)

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert teste.error
    assert mensagem in tela
    assert SEGREDO not in tela and PAYLOAD not in tela
    assert "Traceback" not in tela


def test_erro_de_configuracao_no_aprofundamento_mostra_apenas_a_mensagem_segura():
    mensagem = "NVIDIA_API_KEY não está configurada no .env local."
    aplicacao = aplicacao_com_duas_candidatas(
        erros_por_startup={1: com_causa(ErroConfiguracao(mensagem))}
    )
    teste = abrir_ranking(aplicacao)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert teste.error
    assert mensagem in tela
    assert SEGREDO not in tela and PAYLOAD not in tela
    assert not teste.download_button


def test_erro_do_query_planner_mostra_apenas_a_mensagem_segura():
    mensagem = "O provedor não devolveu um plano de consulta válido."
    aplicacao = AplicacaoFalsa(descobertas=[com_causa(ErroQueryPlanner(mensagem))])
    teste = montar(aplicacao)

    submeter(teste, CONSULTA_PADRAO)
    tela = textos(teste)

    assert teste.error
    assert mensagem in tela
    assert SEGREDO not in tela and PAYLOAD not in tela
    assert "Traceback" not in tela


def test_uma_falha_generica_nao_revela_causa_credencial_nem_payload():
    erro = com_causa(RuntimeError(f"falha interna com {SEGREDO}"))
    aplicacao = aplicacao_com_duas_candidatas(erros_por_startup={1: erro})
    teste = abrir_ranking(aplicacao)

    teste.button(key="aprofundar_1").click().run()
    tela = textos(teste)

    assert teste.error
    assert SEGREDO not in tela
    assert PAYLOAD not in tela
    assert "falha interna" not in tela
    assert "Traceback" not in tela
    assert not teste.download_button
    assert TESE_A not in tela


# ------------------------------------------------------------------------
# O cartão do ranking depois da análise desta sessão
# ------------------------------------------------------------------------

ROTULO_AUSENTE = "Análise ainda não disponível"


def descoberta_so_com_ausente():
    return descoberta_falsa(itens=(item_ausente(posicao=1, id_startup=4),))


def analisar_a_ausente(briefing=None, erro=None):
    """Busca, aprofunda a candidata ausente e volta para o ranking."""
    aplicacao = AplicacaoFalsa(
        descoberta=descoberta_so_com_ausente(),
        aprofundamentos_por_startup=(
            {4: aprofundamento_falso(4, briefing=briefing)}
            if briefing is not None
            else None
        ),
        erros_por_startup={4: erro} if erro is not None else None,
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_4").click().run()
    teste.button(key="voltar_para_candidatas").click().run()
    return aplicacao, teste


# ----------------------------------------------------------------------
# As três variantes atualizam o cartão
# ----------------------------------------------------------------------


def test_uma_ausente_analisada_com_sucesso_deixa_de_ser_ausente():
    _, teste = analisar_a_ausente(briefing_normal_falso())
    tela = textos(teste)

    assert not teste.exception
    assert ROTULO_AUSENTE not in tela
    assert "AI-enabled" in tela
    assert "61/100" in tela


def test_um_briefing_nao_aderente_mostra_o_zero_validado_no_cartao():
    _, teste = analisar_a_ausente(briefing_nao_aderente_falso())
    tela = textos(teste)

    assert ROTULO_AUSENTE not in tela
    assert "non-AI" in tela
    assert "0/100" in tela
    assert "insuficiente" not in tela.casefold()


def test_um_briefing_de_evidencia_insuficiente_nao_declara_classe_nem_score():
    briefing = briefing_insuficiente_falso()
    _, teste = analisar_a_ausente(briefing)
    tela = textos(teste)

    assert ROTULO_AUSENTE not in tela
    assert "Evidência insuficiente" in tela
    assert briefing.avisos[0] in tela
    assert "/100" not in tela
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        assert classe not in tela


# ----------------------------------------------------------------------
# O que a atualização não pode tocar
# ----------------------------------------------------------------------


def test_voltar_para_o_ranking_nao_repete_a_analise():
    aplicacao, teste = analisar_a_ausente(briefing_normal_falso())

    teste.run()

    assert len(aplicacao.aprofundamentos) == 1
    assert aplicacao.consultas == [CONSULTA_PADRAO]


def test_posicao_bm25_e_documentos_sobrevivem_a_atualizacao():
    descoberta = descoberta_falsa()
    aplicacao = AplicacaoFalsa(
        descoberta=descoberta,
        aprofundamentos_por_startup={
            4: aprofundamento_falso(4, briefing=briefing_normal_falso())
        },
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_4").click().run()
    teste.button(key="voltar_para_candidatas").click().run()

    visivel = sessao.ranking_visivel(teste.session_state)

    assert [item.posicao for item in visivel] == [1, 2, 3, 4]
    for antigo, novo in zip(descoberta.ranking, visivel, strict=True):
        assert novo.empresa == antigo.empresa
        assert novo.melhor_score_bm25 == antigo.melhor_score_bm25
        assert novo.documentos == antigo.documentos
    assert visivel[3].status_analise == "concluida"
    # as outras candidatas continuam exatamente como o cache as devolveu
    assert visivel[0] == descoberta.ranking[0]
    assert visivel[1] == descoberta.ranking[1]
    assert visivel[2] == descoberta.ranking[2]


def test_uma_analise_que_falha_nao_altera_o_cartao():
    _, teste = analisar_a_ausente(erro=RuntimeError("indisponível"))
    tela = textos(teste)

    assert ROTULO_AUSENTE in tela
    assert "/100" not in tela
    visivel = sessao.ranking_visivel(teste.session_state)
    assert visivel[0].status_analise == "ausente"
    assert visivel[0].classe is None
    assert visivel[0].fit_score_total is None


def test_uma_busca_nova_descarta_a_atualizacao_de_sessao():
    aplicacao = AplicacaoFalsa(
        descobertas=[descoberta_so_com_ausente(), descoberta_so_com_ausente()],
        aprofundamentos_por_startup={
            4: aprofundamento_falso(4, briefing=briefing_normal_falso())
        },
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_4").click().run()
    teste.button(key="voltar_para_candidatas").click().run()
    assert ROTULO_AUSENTE not in textos(teste)

    submeter(teste, "uma busca completamente nova")
    tela = textos(teste)

    assert ROTULO_AUSENTE in tela
    assert "61/100" not in tela
    assert sessao.ranking_visivel(teste.session_state)[0].status_analise == "ausente"


def test_a_atualizacao_de_uma_candidata_nao_contamina_as_outras():
    descoberta = descoberta_falsa(
        itens=(
            item_concluido_ia(),
            item_concluido_non_ai(),
            item_insuficiente(),
            item_ausente(),
        )
    )
    aplicacao = AplicacaoFalsa(
        descoberta=descoberta,
        aprofundamentos_por_startup={
            4: aprofundamento_falso(4, briefing=briefing_insuficiente_falso())
        },
    )
    teste = montar(aplicacao)
    submeter(teste, CONSULTA_PADRAO)
    teste.button(key="aprofundar_4").click().run()
    teste.button(key="voltar_para_candidatas").click().run()

    visivel = sessao.ranking_visivel(teste.session_state)

    assert visivel[0].fit_score_total == 72
    assert visivel[1].fit_score_total == 0
    assert visivel[2].status_analise == "evidencia_insuficiente"
    assert visivel[3].status_analise == "evidencia_insuficiente"
    assert visivel[3].classe is None and visivel[3].fit_score_total is None
