"""Estado de sessão e rótulos da interface, testados sem Streamlit.

Estas duas peças concentram tudo que a tela decide: o que sobrevive a um
rerun, o que uma nova busca invalida, quando o aprofundamento pode rodar e
como cada um dos quatro estados de análise é nomeado. Como são funções puras
sobre um dicionário, o teste é direto e mais forte do que inspecionar widgets.
"""

from __future__ import annotations

from dataclasses import replace

from radar.interface import estado as sessao
from radar.interface.rotulos import resumir_candidata, resumir_ranking
from tests.apoio_interface import (
    aprofundamento_falso,
    descoberta_falsa,
    item_ausente,
    item_concluido_ia,
    item_concluido_non_ai,
    item_insuficiente,
)


# ----------------------------------------------------------------------
# Descoberta: sobrevivência e invalidação
# ----------------------------------------------------------------------


def test_a_descoberta_registrada_sobrevive_a_reruns_sucessivos():
    memoria: dict = {}
    descoberta = descoberta_falsa()

    sessao.registrar_descoberta(memoria, descoberta)
    for _ in range(3):  # cada iteração simula um rerun do Streamlit
        assert sessao.descoberta_atual(memoria) is descoberta


def test_sem_busca_nenhuma_a_sessao_comeca_vazia():
    memoria: dict = {}

    assert sessao.descoberta_atual(memoria) is None
    assert sessao.startup_selecionada(memoria) is None
    assert sessao.aprofundamento_pendente(memoria) is None
    assert sessao.aprofundamento_selecionado(memoria) is None


def test_uma_nova_busca_apaga_selecao_e_briefing_anteriores():
    memoria: dict = {}
    primeira = descoberta_falsa()
    sessao.registrar_descoberta(memoria, primeira)
    sessao.selecionar_startup(memoria, 1)
    sessao.registrar_aprofundamento(memoria, aprofundamento_falso(1))

    sessao.iniciar_busca(memoria)

    assert sessao.descoberta_atual(memoria) is None
    assert sessao.startup_selecionada(memoria) is None
    assert sessao.aprofundamento_selecionado(memoria) is None
    assert sessao.aprofundamento_pendente(memoria) is None


def test_registrar_uma_nova_descoberta_tambem_invalida_o_briefing_antigo():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)
    sessao.registrar_aprofundamento(memoria, aprofundamento_falso(1))

    segunda = descoberta_falsa(consulta="outra consulta")
    sessao.registrar_descoberta(memoria, segunda)

    assert sessao.descoberta_atual(memoria) is segunda
    assert sessao.startup_selecionada(memoria) is None
    assert sessao.aprofundamento_selecionado(memoria) is None


def test_a_descoberta_permanece_enquanto_uma_candidata_e_analisada():
    memoria: dict = {}
    descoberta = descoberta_falsa()
    sessao.registrar_descoberta(memoria, descoberta)

    sessao.selecionar_startup(memoria, 2)
    sessao.registrar_aprofundamento(memoria, aprofundamento_falso(2))

    assert sessao.descoberta_atual(memoria) is descoberta


# ----------------------------------------------------------------------
# Aprofundamento: uma execução por seleção
# ----------------------------------------------------------------------


def test_a_selecao_marca_o_aprofundamento_como_pendente_uma_unica_vez():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())

    sessao.selecionar_startup(memoria, 3)

    assert sessao.aprofundamento_pendente(memoria) == 3


def test_um_aprofundamento_concluido_nao_e_repetido_por_rerun():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)
    saida = aprofundamento_falso(1)

    sessao.registrar_aprofundamento(memoria, saida)

    for _ in range(3):  # reruns sucessivos não reabrem a execução
        assert sessao.aprofundamento_pendente(memoria) is None
        assert sessao.aprofundamento_selecionado(memoria) is saida


def test_voltar_para_a_lista_preserva_o_briefing_ja_calculado():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)
    saida = aprofundamento_falso(1)
    sessao.registrar_aprofundamento(memoria, saida)

    sessao.limpar_selecao(memoria)

    assert sessao.startup_selecionada(memoria) is None
    sessao.selecionar_startup(memoria, 1)
    assert sessao.aprofundamento_pendente(memoria) is None
    assert sessao.aprofundamento_selecionado(memoria) is saida


def test_cada_candidata_guarda_o_proprio_aprofundamento():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)
    primeira = aprofundamento_falso(1)
    sessao.registrar_aprofundamento(memoria, primeira)

    sessao.selecionar_startup(memoria, 2)

    assert sessao.aprofundamento_pendente(memoria) == 2
    assert sessao.aprofundamento_selecionado(memoria) is None


def test_uma_falha_registrada_nao_vira_briefing_nem_reexecuta_sozinha():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)

    sessao.registrar_falha_aprofundamento(memoria, 1, "Falha ao consultar o grafo.")

    assert sessao.aprofundamento_selecionado(memoria) is None
    assert sessao.falha_selecionada(memoria) == "Falha ao consultar o grafo."
    for _ in range(3):
        assert sessao.aprofundamento_pendente(memoria) is None


def test_um_novo_clique_na_mesma_candidata_permite_tentar_de_novo():
    memoria: dict = {}
    sessao.registrar_descoberta(memoria, descoberta_falsa())
    sessao.selecionar_startup(memoria, 1)
    sessao.registrar_falha_aprofundamento(memoria, 1, "Falha ao consultar o grafo.")

    sessao.selecionar_startup(memoria, 1)

    assert sessao.aprofundamento_pendente(memoria) == 1
    assert sessao.falha_selecionada(memoria) is None


# ----------------------------------------------------------------------
# Rótulos: quatro estados, quatro leituras diferentes
# ----------------------------------------------------------------------


def test_o_resumo_preserva_a_ordem_do_ranking_da_aplicacao():
    descoberta = descoberta_falsa()

    resumos = resumir_ranking(descoberta.ranking)

    assert [resumo.posicao for resumo in resumos] == [1, 2, 3, 4]
    assert [resumo.nome for resumo in resumos] == [
        item.empresa.nome for item in descoberta.ranking
    ]


def test_o_resumo_nao_reordena_um_ranking_fora_da_ordem_alfabetica_ou_de_score():
    itens = (
        item_ausente(posicao=1, id_startup=4, nome="Zeta"),
        item_concluido_ia(posicao=2, id_startup=1, nome="Alfa"),
    )

    resumos = resumir_ranking(itens)

    assert [resumo.nome for resumo in resumos] == ["Zeta", "Alfa"]


def test_concluida_aderente_mostra_classe_e_fit_score_numerico():
    resumo = resumir_candidata(item_concluido_ia())

    assert resumo.classe == "AI-native"
    assert resumo.fit_score == "72/100"
    assert resumo.justificativa_fit_score is not None
    assert resumo.motivo_evidencia_insuficiente is None


def test_concluida_non_ai_mostra_o_zero_como_resultado_validado():
    resumo = resumir_candidata(item_concluido_non_ai())

    assert resumo.classe == "non-AI"
    assert resumo.fit_score == "0/100"
    assert "insuficiente" not in resumo.explicacao_status.casefold()
    assert "indisponível" not in resumo.explicacao_status.casefold()


def test_evidencia_insuficiente_nao_mostra_classe_nem_score_e_diz_o_motivo():
    item = item_insuficiente()

    resumo = resumir_candidata(item)

    assert resumo.classe is None
    assert resumo.fit_score is None
    assert resumo.motivo_evidencia_insuficiente == item.motivo_evidencia_insuficiente


def test_analise_ausente_nao_inventa_zero_nem_classe():
    resumo = resumir_candidata(item_ausente())

    assert resumo.classe is None
    assert resumo.fit_score is None
    assert resumo.motivo_evidencia_insuficiente is None
    assert "0" not in (resumo.fit_score or "")


def test_os_quatro_estados_recebem_rotulos_distintos():
    rotulos = [
        resumir_candidata(item).rotulo_status
        for item in (
            item_concluido_ia(),
            item_concluido_non_ai(),
            item_insuficiente(),
            item_ausente(),
        )
    ]

    assert len(set(rotulos)) == 4


def test_o_resumo_copia_os_dados_de_identificacao_da_candidata():
    item = item_concluido_ia()

    resumo = resumir_candidata(item)

    assert resumo.nome == item.empresa.nome
    assert resumo.setor == item.empresa.setor
    assert resumo.estagio == item.empresa.estagio
    assert resumo.localizacao == item.empresa.localizacao
    assert resumo.descricao == item.empresa.descricao_curta


def test_localizacao_e_descricao_ausentes_viram_texto_explicito():
    item = item_concluido_ia()
    sem_dados = replace(
        item,
        empresa=item.empresa.model_copy(
            update={"localizacao": None, "descricao_curta": None}
        ),
    )

    resumo = resumir_candidata(sem_dados)

    assert resumo.localizacao == "não informada"
    assert resumo.descricao == "sem descrição curta na base"
