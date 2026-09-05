"""Contratos congelados do Briefing (§11 da arquitetura).

O Briefing é o único payload que sai do sistema. O contrato precisa recusar,
por construção, tudo que a §11.2 e a §11.3 declaram impossível em cada
variante — antes de qualquer nó, prompt ou provedor.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from radar.contratos import (
    Briefing,
    BriefingRascunho,
    CabecalhoBriefing,
    ConclusaoAncorada,
    FonteBriefing,
    TextoAncoradoRascunho,
    VereditoBriefing,
)
from tests.conftest import (
    briefing_normal_falso,
    cabecalho_falso,
    fonte_falsa,
    recomendacao_falsa,
    rodape_falso,
)


# ----------------------------------------------------------------------
# Cabeçalho determinístico
# ----------------------------------------------------------------------


def test_cabecalho_carrega_os_sete_campos_deterministicos():
    cabecalho = cabecalho_falso()

    assert cabecalho.nome == "Caju"
    assert str(cabecalho.site).startswith("https://")
    assert cabecalho.setor == "Fintech / RH"
    assert cabecalho.estagio == "série B"
    assert cabecalho.localizacao == "São Paulo, SP"
    assert cabecalho.data_geracao == date(2026, 9, 3)
    assert cabecalho.consulta_original


def test_cabecalho_nao_aceita_classe_referencia():
    """O rótulo de curadoria não existe em nenhum contrato de saída."""
    with pytest.raises(ValidationError):
        CabecalhoBriefing(
            nome="Caju",
            site="https://www.caju.com.br/",
            setor="Fintech / RH",
            estagio="série B",
            localizacao=None,
            data_geracao=date(2026, 9, 3),
            consulta_original="consulta",
            classe_referencia="AI-enabled",
        )


# ----------------------------------------------------------------------
# Conclusões ancoradas
# ----------------------------------------------------------------------


def test_conclusao_recusa_ids_repetidos_ou_fora_de_ordem():
    with pytest.raises(ValidationError, match="repet"):
        ConclusaoAncorada(texto="Texto.", ids_afirmacoes_suporte=[1, 1])
    with pytest.raises(ValidationError, match="ordem"):
        ConclusaoAncorada(texto="Texto.", ids_afirmacoes_suporte=[2, 1])


def test_conclusao_recusa_texto_em_branco():
    with pytest.raises(ValidationError):
        ConclusaoAncorada(texto="   ", ids_afirmacoes_suporte=[1])


# ----------------------------------------------------------------------
# Variante normal
# ----------------------------------------------------------------------


def test_briefing_normal_valido_e_construido():
    briefing = briefing_normal_falso()

    assert briefing.variante == "normal"
    assert briefing.veredito.classe == "AI-enabled"
    assert briefing.veredito.fit_score_total == 61
    assert len(briefing.pontos_de_conversa) == 2
    assert len(briefing.recomendacoes) == 1
    assert briefing.rodape.trajeto.count("briefing") == 1


def test_normal_exige_classe_e_fit_score_no_veredito():
    with pytest.raises(ValidationError, match="classe"):
        briefing_normal_falso(
            veredito=VereditoBriefing(
                classe=None,
                fit_score_total=61,
                tese="Tese sem classe.",
                ids_afirmacoes_suporte=[1],
            )
        )
    with pytest.raises(ValidationError, match="fit_score_total"):
        briefing_normal_falso(
            veredito=VereditoBriefing(
                classe="AI-enabled",
                fit_score_total=None,
                tese="Tese sem score.",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_normal_exige_suporte_proprio_na_tese():
    """A tese não herda a proveniência da recomendação: precisa dos ids dela."""
    with pytest.raises(ValidationError, match="veredito"):
        briefing_normal_falso(
            veredito=VereditoBriefing(
                classe="AI-enabled",
                fit_score_total=61,
                tese="Tese sem lastro próprio.",
                ids_afirmacoes_suporte=[],
            )
        )


def test_normal_exige_suporte_proprio_na_sintese():
    with pytest.raises(ValidationError, match="sintese_executiva"):
        briefing_normal_falso(
            sintese_executiva=ConclusaoAncorada(
                texto="Síntese sem lastro próprio.", ids_afirmacoes_suporte=[]
            )
        )


def test_normal_exige_suporte_proprio_em_cada_ponto_de_conversa():
    with pytest.raises(ValidationError, match="pontos_de_conversa"):
        briefing_normal_falso(
            pontos_de_conversa=[
                ConclusaoAncorada(texto="Ponto com lastro.", ids_afirmacoes_suporte=[1]),
                ConclusaoAncorada(texto="Ponto sem lastro.", ids_afirmacoes_suporte=[]),
            ]
        )


@pytest.mark.parametrize("quantidade", [0, 1, 5])
def test_normal_exige_de_dois_a_quatro_pontos_de_conversa(quantidade):
    pontos = [
        ConclusaoAncorada(texto=f"Ponto {indice}.", ids_afirmacoes_suporte=[1])
        for indice in range(quantidade)
    ]
    with pytest.raises(ValidationError):
        briefing_normal_falso(pontos_de_conversa=pontos)


@pytest.mark.parametrize("quantidade", [2, 3, 4])
def test_normal_aceita_de_dois_a_quatro_pontos_de_conversa(quantidade):
    pontos = [
        ConclusaoAncorada(texto=f"Ponto {indice}.", ids_afirmacoes_suporte=[1])
        for indice in range(quantidade)
    ]
    briefing = briefing_normal_falso(pontos_de_conversa=pontos)
    assert len(briefing.pontos_de_conversa) == quantidade


def test_sintese_executiva_respeita_o_teto_de_cento_e_vinte_palavras():
    texto_no_limite = " ".join(["palavra"] * 120) + "."
    briefing = briefing_normal_falso(
        sintese_executiva=ConclusaoAncorada(
            texto=texto_no_limite, ids_afirmacoes_suporte=[1]
        )
    )
    assert briefing.sintese_executiva.texto == texto_no_limite

    with pytest.raises(ValidationError, match="120 palavras"):
        briefing_normal_falso(
            sintese_executiva=ConclusaoAncorada(
                texto=" ".join(["palavra"] * 121) + ".",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_normal_exige_de_uma_a_cinco_recomendacoes():
    with pytest.raises(ValidationError, match="recomendaç"):
        briefing_normal_falso(recomendacoes=[])


def test_briefing_nao_repete_gap_entre_recomendacoes():
    with pytest.raises(ValidationError, match="gap"):
        briefing_normal_falso(
            recomendacoes=[
                recomendacao_falsa("distribuicao"),
                recomendacao_falsa("distribuicao", id_chunk=102),
            ]
        )


# ----------------------------------------------------------------------
# Variante non-AI
# ----------------------------------------------------------------------


def briefing_nao_aderente(**ajustes):
    campos = {
        "variante": "nao_aderente",
        "veredito": VereditoBriefing(
            classe="non-AI",
            fit_score_total=0,
            tese="A base pública não descreve uso de IA no produto vendido.",
            ids_afirmacoes_suporte=[1],
        ),
        "pontos_de_conversa": [
            ConclusaoAncorada(
                texto="Confirmar se há projeto de IA fora do material público.",
                ids_afirmacoes_suporte=[1],
            )
        ],
        "recomendacoes": [],
        "avisos": ["Veredito non-AI: a stack NVIDIA não é recomendada para esta empresa."],
        "rodape": rodape_falso(rota_r3="nao_aderente"),
    }
    campos.update(ajustes)
    return briefing_normal_falso(**campos)


def test_nao_aderente_valido_tem_score_zero_e_nenhuma_recomendacao():
    briefing = briefing_nao_aderente()

    assert briefing.veredito.classe == "non-AI"
    assert briefing.veredito.fit_score_total == 0
    assert briefing.recomendacoes == []
    assert 1 <= len(briefing.pontos_de_conversa) <= 2


def test_nao_aderente_recusa_recomendacao():
    with pytest.raises(ValidationError, match="recomendaç"):
        briefing_nao_aderente(recomendacoes=[recomendacao_falsa()])


def test_nao_aderente_recusa_score_diferente_de_zero():
    with pytest.raises(ValidationError, match="fit_score_total"):
        briefing_nao_aderente(
            veredito=VereditoBriefing(
                classe="non-AI",
                fit_score_total=42,
                tese="Tese com score indevido.",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_nao_aderente_recusa_classe_diferente_de_non_ai():
    with pytest.raises(ValidationError, match="classe"):
        briefing_nao_aderente(
            veredito=VereditoBriefing(
                classe="AI-enabled",
                fit_score_total=0,
                tese="Tese com classe incompatível.",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_nao_aderente_exige_o_aviso_explicito_de_nao_aderencia():
    with pytest.raises(ValidationError, match="aviso"):
        briefing_nao_aderente(avisos=[])


def test_nao_aderente_aceita_no_maximo_dois_pontos():
    with pytest.raises(ValidationError, match="pontos_de_conversa"):
        briefing_nao_aderente(
            pontos_de_conversa=[
                ConclusaoAncorada(texto=f"Ponto {i}.", ids_afirmacoes_suporte=[1])
                for i in range(3)
            ]
        )


# ----------------------------------------------------------------------
# Variante evidência insuficiente
# ----------------------------------------------------------------------


def briefing_insuficiente(**ajustes):
    campos = {
        "variante": "evidencia_insuficiente",
        "veredito": VereditoBriefing(
            classe=None,
            fit_score_total=None,
            tese="A base disponível não sustenta uma conclusão sobre a empresa.",
            ids_afirmacoes_suporte=[],
        ),
        "sintese_executiva": ConclusaoAncorada(
            texto="Nenhuma afirmação sobreviveu à conferência de proveniência.",
            ids_afirmacoes_suporte=[],
        ),
        "pontos_de_conversa": [],
        "recomendacoes": [],
        "fontes": [],
        "avisos": ["Evidência insuficiente: nenhuma afirmação confirmada."],
        "rodape": rodape_falso(
            afirmacoes_confirmadas=0,
            afirmacoes_derrubadas=2,
            rota_r3="evidencia_insuficiente",
        ),
    }
    campos.update(ajustes)
    return briefing_normal_falso(**campos)


def test_insuficiente_valido_tem_classe_e_score_nulos_e_listas_vazias():
    briefing = briefing_insuficiente()

    assert briefing.veredito.classe is None
    assert briefing.veredito.fit_score_total is None
    assert briefing.veredito.ids_afirmacoes_suporte == []
    assert briefing.sintese_executiva.ids_afirmacoes_suporte == []
    assert briefing.pontos_de_conversa == []
    assert briefing.recomendacoes == []
    assert briefing.avisos


def test_insuficiente_recusa_classe_ou_score_fabricados():
    with pytest.raises(ValidationError, match="classe"):
        briefing_insuficiente(
            veredito=VereditoBriefing(
                classe="AI-native",
                fit_score_total=None,
                tese="Tese com classe fabricada.",
                ids_afirmacoes_suporte=[],
            )
        )
    with pytest.raises(ValidationError, match="fit_score_total"):
        briefing_insuficiente(
            veredito=VereditoBriefing(
                classe=None,
                fit_score_total=0,
                tese="Tese com score fabricado.",
                ids_afirmacoes_suporte=[],
            )
        )


def test_insuficiente_recusa_conclusao_ancorada_em_evidencia():
    """Sem lastro não há tese: nem positiva, nem negativa."""
    with pytest.raises(ValidationError, match="veredito"):
        briefing_insuficiente(
            veredito=VereditoBriefing(
                classe=None,
                fit_score_total=None,
                tese="Tese apoiada em id que não deveria existir aqui.",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_insuficiente_recusa_pontos_de_conversa():
    with pytest.raises(ValidationError, match="pontos_de_conversa"):
        briefing_insuficiente(
            pontos_de_conversa=[
                ConclusaoAncorada(texto="Ponto indevido.", ids_afirmacoes_suporte=[])
            ]
        )


def test_insuficiente_exige_ao_menos_um_aviso():
    with pytest.raises(ValidationError, match="aviso"):
        briefing_insuficiente(avisos=[])


# ----------------------------------------------------------------------
# Fontes e rodapé
# ----------------------------------------------------------------------


def test_fonte_exige_host_normalizado_correspondente_a_url():
    fonte = fonte_falsa("https://www.Fonte-B.example/pagina")
    assert fonte.host_normalizado == "fonte-b.example"

    with pytest.raises(ValidationError, match="host"):
        FonteBriefing(
            url_fonte="https://fonte-a.example/materia",
            host_normalizado="outra-fonte.example",
            tipo="notícia",
            titulo="Título",
            data_publicacao=None,
        )


def test_fontes_nao_repetem_url_e_vem_em_ordem_deterministica():
    with pytest.raises(ValidationError, match="repet"):
        briefing_normal_falso(fontes=[fonte_falsa(), fonte_falsa()])

    with pytest.raises(ValidationError, match="ordem"):
        briefing_normal_falso(
            fontes=[
                fonte_falsa("https://z-fonte.example/a"),
                fonte_falsa("https://a-fonte.example/b"),
            ]
        )


def test_rodape_exige_o_no_briefing_exatamente_uma_vez_no_trajeto():
    with pytest.raises(ValidationError, match="briefing"):
        rodape_falso(trajeto=["extractor", "classifier"])
    with pytest.raises(ValidationError, match="briefing"):
        rodape_falso(trajeto=["extractor", "briefing", "briefing"])


def test_rodape_preserva_os_dados_de_auditoria():
    rodape = rodape_falso(afirmacoes_confirmadas=3, afirmacoes_derrubadas=1)

    assert rodape.versao_rubrica == "rubrica-v1"
    assert rodape.data_execucao == date(2026, 9, 3)
    assert rodape.afirmacoes_confirmadas == 3
    assert rodape.afirmacoes_derrubadas == 1
    assert rodape.rota_r3 == "prosseguir"
    assert "briefing" in rodape.trajeto


def test_briefing_atravessa_a_forma_json_do_checkpoint():
    original = briefing_normal_falso()

    reidratado = Briefing.model_validate(original.model_dump(mode="json"))

    assert reidratado == original


# ----------------------------------------------------------------------
# Contrato reduzido do LLM
# ----------------------------------------------------------------------


def rascunho_valido(**ajustes):
    campos = {
        "tese": {"texto": "Tese curta.", "ids_afirmacoes_suporte": [1]},
        "sintese_executiva": {
            "texto": "Síntese curta.",
            "ids_afirmacoes_suporte": [2],
        },
        "pontos_de_conversa": [
            {"texto": "Ponto um.", "ids_afirmacoes_suporte": [1]},
            {"texto": "Ponto dois.", "ids_afirmacoes_suporte": [2]},
        ],
    }
    campos.update(ajustes)
    return campos


def test_rascunho_valido_traz_apenas_o_que_o_llm_pode_escrever():
    rascunho = BriefingRascunho.model_validate(rascunho_valido())

    assert rascunho.tese.ids_afirmacoes_suporte == [1]
    assert len(rascunho.pontos_de_conversa) == 2


@pytest.mark.parametrize(
    "campo_proibido",
    [
        "classe",
        "fit_score_total",
        "cabecalho",
        "recomendacoes",
        "prioridade",
        "complexidade",
        "fontes",
        "avisos",
        "rodape",
        "data_geracao",
        "trajeto",
    ],
)
def test_rascunho_recusa_todo_campo_que_o_llm_nao_pode_produzir(campo_proibido):
    with pytest.raises(ValidationError):
        BriefingRascunho.model_validate(rascunho_valido(**{campo_proibido: "x"}))


@pytest.mark.parametrize("quantidade", [0, 1, 5])
def test_rascunho_exige_de_dois_a_quatro_pontos(quantidade):
    pontos = [
        {"texto": f"Ponto {i}.", "ids_afirmacoes_suporte": [1]}
        for i in range(quantidade)
    ]
    with pytest.raises(ValidationError):
        BriefingRascunho.model_validate(rascunho_valido(pontos_de_conversa=pontos))


def test_rascunho_exige_ao_menos_um_id_por_conclusao():
    with pytest.raises(ValidationError):
        TextoAncoradoRascunho(texto="Sem lastro.", ids_afirmacoes_suporte=[])


# ----------------------------------------------------------------------
# Combinações impossíveis de variante e classe (auditoria)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("classe", ["AI-native", "AI-enabled"])
def test_normal_aceita_somente_as_duas_classes_aderentes(classe):
    briefing = briefing_normal_falso(
        veredito=VereditoBriefing(
            classe=classe,
            fit_score_total=61,
            tese="Tese com classe aderente.",
            ids_afirmacoes_suporte=[1],
        )
    )
    assert briefing.veredito.classe == classe


def test_normal_recusa_classe_non_ai():
    """non-AI só pode sair pela variante de não aderência, com score zero."""
    with pytest.raises(ValidationError, match="non-AI"):
        briefing_normal_falso(
            veredito=VereditoBriefing(
                classe="non-AI",
                fit_score_total=61,
                tese="Tese incoerente com a classe.",
                ids_afirmacoes_suporte=[1],
            )
        )


def test_insuficiente_recusa_indice_de_fontes():
    """Sem id citado não há união de onde derivar fonte alguma."""
    with pytest.raises(ValidationError, match="fontes"):
        briefing_insuficiente(fontes=[fonte_falsa()])
