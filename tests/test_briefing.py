"""O nó Briefing: montagem determinística com escrita controlada pelo LLM.

O contrato reduzido deixa o modelo escrever apenas tese, síntese e pontos, e
escolher ids de afirmações **confirmadas**. Classe, fit-score, cabeçalho,
recomendações, fontes, avisos e rodapé são montagem determinística sobre
objetos que já passaram por validação. Tudo offline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from radar.agentes.briefing import (
    AgenteBriefing,
    ErroBriefing,
)
from radar.base_startups import BaseStartups
from radar.contratos import (
    Briefing,
    Classificacao,
    EntradaFitScore,
    MetadadoDocumentoFitScore,
    PerfilValidado,
    Recomendacao,
    ResultadoRecuperacao,
)
from radar.recomendacao import calcular_fit_score
from tests.conftest import (
    ProvedorSequencialFalso,
    afirmacao_validada_falsa,
    citacao_nvidia_falsa,
    contexto_nvidia_falso,
    perfil_validado_falso,
    recomendacao_falsa,
)


DATA_FIXA = date(2026, 9, 3)


def relogio_fixo():
    return DATA_FIXA


# ----------------------------------------------------------------------
# Estado real montado sobre a base curada
# ----------------------------------------------------------------------


def documentos_caju(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        return conexao.execute(
            """
            SELECT s.id AS id_startup, s.nome, s.setor, s.estagio, s.localizacao,
                   s.descricao_curta, d.id AS id_documento, d.tipo, d.titulo,
                   d.url_fonte, d.dominio_fonte, d.data_publicacao
            FROM startups s
            JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = 'Caju'
            ORDER BY d.id
            """
        ).fetchall()


def outra_startup(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        return conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.url_fonte
            FROM startups s
            JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome != 'Caju'
            ORDER BY d.id
            LIMIT 1
            """
        ).fetchone()


def perfil_de_dois_documentos(linhas, *, derrubar_segunda=False):
    """Duas afirmações confirmadas, cada uma num documento distinto."""
    primeira = afirmacao_validada_falsa(
        1, "distribuicao", polaridade="ausencia_explicita",
        id_documento=linhas[0]["id_documento"],
    )
    segunda = afirmacao_validada_falsa(
        2,
        "escala_e_dor_operacional",
        id_documento=linhas[1]["id_documento"],
        situacao="derrubada" if derrubar_segunda else "confirmada",
    )
    hosts = sorted(
        {linhas[0]["dominio_fonte"]}
        if derrubar_segunda
        else {linhas[0]["dominio_fonte"], linhas[1]["dominio_fonte"]}
    )
    return perfil_validado_falso([primeira, segunda], hosts=hosts)


def recuperacao(linhas):
    return ResultadoRecuperacao(
        empresas=[
            {
                "id_startup": linhas[0]["id_startup"],
                "nome": linhas[0]["nome"],
                "setor": linhas[0]["setor"],
                "estagio": linhas[0]["estagio"],
                "localizacao": linhas[0]["localizacao"],
                "descricao_curta": linhas[0]["descricao_curta"],
            }
        ],
        documentos=[
            {
                "id_documento": linha["id_documento"],
                "id_startup": linha["id_startup"],
                "tipo": linha["tipo"],
                "titulo": linha["titulo"],
                "url_fonte": linha["url_fonte"],
                "dominio_fonte": linha["dominio_fonte"],
                "data_acesso": "2026-08-01",
                "score_bm25": -1.0,
            }
            for linha in linhas
        ],
        filtros_aplicados={},
    )


def fit_score_de(perfil, classificacao, linhas):
    entrada = EntradaFitScore(
        classe=classificacao.classe,
        ids_afirmacoes_suporte_classe=sorted(classificacao.ids_afirmacoes_suporte),
        perfil_validado=perfil,
        setor=linhas[0]["setor"],
        estagio=linhas[0]["estagio"],
        documentos=[
            MetadadoDocumentoFitScore(
                id_documento=linha["id_documento"],
                url_fonte=linha["url_fonte"],
                host_normalizado=linha["dominio_fonte"],
                data_publicacao=linha["data_publicacao"],
            )
            for linha in linhas
        ],
        data_referencia=DATA_FIXA,
    )
    return calcular_fit_score(entrada)


def estado_normal(caminho_banco, **ajustes):
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_de_dois_documentos(linhas)
    classificacao = Classificacao(
        classe="AI-enabled",
        justificativa=(
            "A plataforma de benefícios é o produto contratado pelas empresas. "
            "O material público não descreve modelos próprios como o produto."
        ),
        ids_afirmacoes_suporte=[1],
    )
    recomendacao = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[0]["url_fonte"],
    )
    estado = {
        "consulta_usuario": "fintech brasileira de benefícios com cartão",
        "startup_selecionada": linhas[0]["id_startup"],
        "resultado_recuperacao": recuperacao(linhas),
        "classificacao": classificacao,
        "perfil_validado": perfil,
        "confianca_perfil": "normal",
        "criterios_relaxados": [],
        "contexto_nvidia": contexto_nvidia_falso().model_dump(mode="json"),
        "recomendacoes": [recomendacao.model_dump(mode="json")],
        "fit_score": fit_score_de(perfil, classificacao, linhas),
        "erros": [],
        "trajeto": [
            "query_planner",
            "retriever",
            "extractor",
            "classifier",
            "evidence_validator",
            "nvidia_rag",
            "recommendation",
        ],
    }
    estado.update(ajustes)
    return estado


def rascunho(
    *,
    ids_tese=(1,),
    ids_sintese=(2,),
    pontos=((1,), (2,)),
    texto_sintese="A empresa opera benefícios corporativos com dor de escala documentada.",
):
    return {
        "tese": {
            "texto": "A empresa é AI-enabled com lacuna de distribuição confirmada.",
            "ids_afirmacoes_suporte": list(ids_tese),
        },
        "sintese_executiva": {
            "texto": texto_sintese,
            "ids_afirmacoes_suporte": list(ids_sintese),
        },
        "pontos_de_conversa": [
            {
                "texto": f"Ponto de conversa número {indice}.",
                "ids_afirmacoes_suporte": list(ids),
            }
            for indice, ids in enumerate(pontos, start=1)
        ],
    }


def montar_no(caminho_banco, *respostas):
    provedor = ProvedorSequencialFalso(*respostas)
    no = AgenteBriefing(
        BaseStartups(caminho_banco), provedor, relogio=relogio_fixo
    )
    return no, provedor


def briefing_de(saida) -> Briefing:
    return Briefing.model_validate(saida["briefing"])


# ----------------------------------------------------------------------
# Caminho normal
# ----------------------------------------------------------------------


def test_briefing_normal_valido_e_construido_do_estado(caminho_banco):
    no, provedor = montar_no(caminho_banco, rascunho())

    saida = no(estado_normal(caminho_banco))

    briefing = briefing_de(saida)
    assert provedor.chamadas == 1
    assert briefing.variante == "normal"
    assert briefing.cabecalho.nome == "Caju"
    assert str(briefing.cabecalho.site).startswith("https://")
    assert briefing.cabecalho.setor == "Fintech / RH"
    assert briefing.cabecalho.estagio == "série B"
    assert briefing.cabecalho.data_geracao == DATA_FIXA
    assert briefing.cabecalho.consulta_original == (
        "fintech brasileira de benefícios com cartão"
    )
    assert saida["trajeto"] == ["briefing"]


def test_veredito_normal_traz_classe_score_e_suporte_proprio(caminho_banco):
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert briefing.veredito.classe == "AI-enabled"
    assert isinstance(briefing.veredito.fit_score_total, int)
    assert briefing.veredito.ids_afirmacoes_suporte == [1]


def test_veredito_usa_o_total_do_fit_score_do_estado(caminho_banco):
    estado = estado_normal(caminho_banco)
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado))

    assert briefing.veredito.fit_score_total == estado["fit_score"].total


def test_sintese_e_pontos_tem_cada_um_o_proprio_suporte(caminho_banco):
    no, _ = montar_no(
        caminho_banco, rascunho(ids_sintese=(1, 2), pontos=((2,), (1, 2)))
    )

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert briefing.sintese_executiva.ids_afirmacoes_suporte == [1, 2]
    assert [p.ids_afirmacoes_suporte for p in briefing.pontos_de_conversa] == [
        [2],
        [1, 2],
    ]
    assert briefing.sintese_executiva.texto != briefing.veredito.tese


# ----------------------------------------------------------------------
# Rejeição de ids: ausente, derrubado e estrangeiro
# ----------------------------------------------------------------------


def test_id_ausente_no_perfil_e_recusado_apos_a_correcao_unica(caminho_banco):
    no, provedor = montar_no(
        caminho_banco,
        rascunho(ids_tese=(999,)),
        rascunho(ids_tese=(999,)),
    )

    with pytest.raises(ErroBriefing, match="999"):
        no(estado_normal(caminho_banco))

    assert provedor.chamadas == 2


def test_id_derrubado_e_recusado(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_de_dois_documentos(linhas, derrubar_segunda=True)
    estado = estado_normal(caminho_banco, perfil_validado=perfil)
    no, provedor = montar_no(
        caminho_banco, rascunho(ids_sintese=(2,)), rascunho(ids_sintese=(2,))
    )

    with pytest.raises(ErroBriefing, match="confirmad"):
        no(estado)

    assert provedor.chamadas == 2


def test_id_de_outro_perfil_e_recusado(caminho_banco):
    """Um id plausível de outra análise não vira suporte desta."""
    linhas = documentos_caju(caminho_banco)
    unica = afirmacao_validada_falsa(
        1, "distribuicao", polaridade="ausencia_explicita",
        id_documento=linhas[0]["id_documento"],
    )
    perfil = perfil_validado_falso([unica], hosts=[linhas[0]["dominio_fonte"]])
    estado = estado_normal(caminho_banco, perfil_validado=perfil)
    no, provedor = montar_no(
        caminho_banco,
        rascunho(ids_sintese=(2,)),
        rascunho(ids_sintese=(2,)),
    )

    with pytest.raises(ErroBriefing):
        no(estado)

    assert provedor.chamadas == 2


def test_uma_unica_correcao_estruturada_recupera_o_briefing(caminho_banco):
    no, provedor = montar_no(
        caminho_banco, rascunho(ids_tese=(999,)), rascunho()
    )

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert provedor.chamadas == 2
    assert briefing.variante == "normal"
    # A segunda mensagem carrega o erro da primeira, sem texto livre.
    assert any("Falha" in texto for _papel, texto in provedor.mensagens[1])


def test_resposta_fora_do_contrato_consome_a_mesma_correcao_unica(caminho_banco):
    no, provedor = montar_no(
        caminho_banco, {"tese": "texto solto"}, {"tese": "texto solto"}
    )

    with pytest.raises(ErroBriefing):
        no(estado_normal(caminho_banco))

    assert provedor.chamadas == 2


def test_sintese_acima_de_cento_e_vinte_palavras_e_recusada(caminho_banco):
    longa = " ".join(["palavra"] * 121)
    no, provedor = montar_no(
        caminho_banco,
        rascunho(texto_sintese=longa),
        rascunho(texto_sintese=longa),
    )

    with pytest.raises(ErroBriefing, match="120"):
        no(estado_normal(caminho_banco))

    assert provedor.chamadas == 2


@pytest.mark.parametrize("quantidade", [1, 5])
def test_normal_exige_de_dois_a_quatro_pontos_vindos_do_llm(
    caminho_banco, quantidade
):
    pontos = tuple((1,) for _ in range(quantidade))
    no, provedor = montar_no(
        caminho_banco, rascunho(pontos=pontos), rascunho(pontos=pontos)
    )

    with pytest.raises(ErroBriefing):
        no(estado_normal(caminho_banco))

    assert provedor.chamadas == 2


def test_falha_do_provedor_nao_produz_briefing_parcial(caminho_banco):
    no, provedor = montar_no(
        caminho_banco, RuntimeError("provedor indisponível")
    )

    with pytest.raises(ErroBriefing, match="não respondeu"):
        no(estado_normal(caminho_banco))

    assert provedor.chamadas == 1


# ----------------------------------------------------------------------
# Recomendações: revalidadas, nunca reescritas
# ----------------------------------------------------------------------


def test_recomendacoes_sao_revalidadas_e_embutidas_sem_reescrita(caminho_banco):
    estado = estado_normal(caminho_banco)
    original = estado["recomendacoes"][0]
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado))

    assert len(briefing.recomendacoes) == 1
    embutida = briefing.recomendacoes[0]
    assert embutida.model_dump(mode="json") == original
    assert embutida.prioridade == original["prioridade"]
    assert embutida.complexidade == original["complexidade"]
    assert embutida.tecnologias == original["tecnologias"]
    assert [c["id_chunk"] for c in original["citacoes_nvidia"]] == [
        c.id_chunk for c in embutida.citacoes_nvidia
    ]


def test_recomendacao_fora_do_contrato_no_estado_falha_com_seguranca(caminho_banco):
    estado = estado_normal(caminho_banco, recomendacoes=[{"gap_enderecado": "x"}])
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing):
        no(estado)

    assert provedor.chamadas == 0


def test_o_llm_do_briefing_nunca_recebe_as_recomendacoes_para_reescrever(
    caminho_banco,
):
    no, provedor = montar_no(caminho_banco, rascunho())

    no(estado_normal(caminho_banco))

    prompt = json.dumps(provedor.mensagens[0], ensure_ascii=False)
    assert "justificativa_tecnica" not in prompt
    assert "proxima_acao" not in prompt
    assert "prioridade" not in prompt.replace("NÃO define prioridade", "")


# ----------------------------------------------------------------------
# Fontes: união determinística, deduplicada e isolada por startup
# ----------------------------------------------------------------------


def test_fontes_saem_da_uniao_dos_ids_citados_por_toda_conclusao(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    no, _ = montar_no(caminho_banco, rascunho(ids_tese=(1,), ids_sintese=(2,)))

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    urls = {str(fonte.url_fonte) for fonte in briefing.fontes}
    assert urls == {linhas[0]["url_fonte"], linhas[1]["url_fonte"]}


def test_fontes_deduplicam_o_mesmo_documento_citado_varias_vezes(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    no, _ = montar_no(
        caminho_banco,
        rascunho(ids_tese=(1,), ids_sintese=(1,), pontos=((1,), (1,))),
    )

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert len(briefing.fontes) == 1
    assert str(briefing.fontes[0].url_fonte) == linhas[0]["url_fonte"]


def test_fontes_trazem_a_projecao_publica_exigida_pela_secao_11_2(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    por_url = {linha["url_fonte"]: linha for linha in linhas}
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    for fonte in briefing.fontes:
        original = por_url[str(fonte.url_fonte)]
        assert fonte.titulo == original["titulo"]
        assert fonte.tipo == original["tipo"]
        assert fonte.host_normalizado == original["dominio_fonte"]
        esperada = original["data_publicacao"]
        assert (
            fonte.data_publicacao.isoformat() if fonte.data_publicacao else None
        ) == esperada


def test_resolucao_de_fontes_e_isolada_por_startup(caminho_banco):
    """Documento de outra empresa nunca entra no índice desta análise."""
    linhas = documentos_caju(caminho_banco)
    intrusa = outra_startup(caminho_banco)
    afirmacao_intrusa = afirmacao_validada_falsa(
        2, "escala_e_dor_operacional", id_documento=intrusa["id_documento"]
    )
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "distribuicao", polaridade="ausencia_explicita",
                id_documento=linhas[0]["id_documento"],
            ),
            afirmacao_intrusa,
        ],
        hosts=[linhas[0]["dominio_fonte"]],
    )
    estado = estado_normal(caminho_banco, perfil_validado=perfil)
    no, _ = montar_no(caminho_banco, rascunho(ids_sintese=(2,)), rascunho())

    with pytest.raises(ErroBriefing, match="outra startup|não pertence"):
        no(estado)


def test_base_recusa_documento_de_outra_startup_na_fronteira_de_fontes(
    caminho_banco,
):
    linhas = documentos_caju(caminho_banco)
    intrusa = outra_startup(caminho_banco)
    base = BaseStartups(caminho_banco)

    fontes = base.carregar_fontes_briefing(
        linhas[0]["id_startup"],
        [linhas[0]["id_documento"], intrusa["id_documento"]],
    )

    assert set(fontes) == {linhas[0]["id_documento"]}


def test_base_le_o_site_oficial_sem_tocar_classe_referencia(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    base = BaseStartups(caminho_banco)

    site = base.carregar_site_oficial(linhas[0]["id_startup"])

    assert site.startswith("https://")
    assert "caju" in site


# ----------------------------------------------------------------------
# Segredos e rótulo de curadoria fora do prompt
# ----------------------------------------------------------------------


def test_classe_referencia_nunca_alcanca_o_no_nem_o_prompt(caminho_banco):
    no, provedor = montar_no(caminho_banco, rascunho())

    saida = no(estado_normal(caminho_banco))

    prompt = json.dumps(provedor.mensagens[0], ensure_ascii=False)
    assert "classe_referencia" not in prompt
    assert "classe_referencia" not in json.dumps(saida["briefing"], ensure_ascii=False)


def test_prompt_nao_carrega_documentos_inteiros_da_startup(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    with sqlite3.connect(caminho_banco) as conexao:
        conteudo = conexao.execute(
            "SELECT conteudo_texto FROM documentos WHERE id = ?",
            (linhas[0]["id_documento"],),
        ).fetchone()[0]
    no, provedor = montar_no(caminho_banco, rascunho())

    no(estado_normal(caminho_banco))

    prompt = json.dumps(provedor.mensagens[0], ensure_ascii=False)
    assert conteudo[:120] not in prompt


# ----------------------------------------------------------------------
# Variante non-AI
# ----------------------------------------------------------------------


def estado_nao_aderente(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "outro", id_documento=linhas[0]["id_documento"]
            ),
            afirmacao_validada_falsa(
                2, "outro", id_documento=linhas[1]["id_documento"]
            ),
        ],
        hosts=sorted({linhas[0]["dominio_fonte"], linhas[1]["dominio_fonte"]}),
    )
    return estado_normal(
        caminho_banco,
        perfil_validado=perfil,
        classificacao=Classificacao(
            classe="non-AI",
            justificativa=(
                "O material público descreve um produto operacional sem IA. "
                "Não há descrição de modelos próprios nem de inferência."
            ),
            ids_afirmacoes_suporte=[1, 2],
        ),
        recomendacoes=None,
        fit_score=None,
        trajeto=[
            "query_planner",
            "retriever",
            "extractor",
            "classifier",
            "evidence_validator",
        ],
    )


def test_variante_non_ai_nao_chama_o_provedor_do_briefing(caminho_banco):
    no, provedor = montar_no(caminho_banco)

    saida = no(estado_nao_aderente(caminho_banco))

    assert provedor.chamadas == 0
    assert briefing_de(saida).variante == "nao_aderente"


def test_variante_non_ai_tem_score_zero_e_nenhuma_recomendacao(caminho_banco):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_nao_aderente(caminho_banco)))

    assert briefing.veredito.classe == "non-AI"
    assert briefing.veredito.fit_score_total == 0
    assert briefing.recomendacoes == []
    assert briefing.veredito.ids_afirmacoes_suporte == [1, 2]
    assert 1 <= len(briefing.pontos_de_conversa) <= 2
    assert all(p.ids_afirmacoes_suporte for p in briefing.pontos_de_conversa)


def test_variante_non_ai_avisa_a_nao_aderencia_e_nao_cita_tecnologia_nvidia(
    caminho_banco,
):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_nao_aderente(caminho_banco)))

    assert any("non-AI" in aviso for aviso in briefing.avisos)
    texto = json.dumps(briefing.model_dump(mode="json"), ensure_ascii=False)
    assert "NVIDIA" not in texto


# ----------------------------------------------------------------------
# Variante evidência insuficiente
# ----------------------------------------------------------------------


def estado_insuficiente(caminho_banco, **ajustes):
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "outro", id_documento=linhas[0]["id_documento"],
                situacao="derrubada",
            )
        ],
        hosts=[],
    )
    base = estado_normal(
        caminho_banco,
        perfil_validado=perfil,
        recomendacoes=None,
        fit_score=None,
        trajeto=[
            "query_planner",
            "retriever",
            "extractor",
            "classifier",
            "evidence_validator",
        ],
    )
    base.update(ajustes)
    return base


def test_variante_insuficiente_nao_chama_nenhum_llm(caminho_banco):
    no, provedor = montar_no(caminho_banco)

    saida = no(estado_insuficiente(caminho_banco))

    assert provedor.chamadas == 0
    assert briefing_de(saida).variante == "evidencia_insuficiente"


def test_variante_insuficiente_tem_classe_e_score_nulos_e_listas_vazias(
    caminho_banco,
):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_insuficiente(caminho_banco)))

    assert briefing.veredito.classe is None
    assert briefing.veredito.fit_score_total is None
    assert briefing.veredito.ids_afirmacoes_suporte == []
    assert briefing.sintese_executiva.ids_afirmacoes_suporte == []
    assert briefing.pontos_de_conversa == []
    assert briefing.recomendacoes == []
    assert briefing.fontes == []


def test_variante_insuficiente_explica_a_causa_e_conta_as_derrubadas(caminho_banco):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_insuficiente(caminho_banco)))

    assert briefing.avisos
    assert any("derrubada" in aviso for aviso in briefing.avisos)
    assert briefing.rodape.afirmacoes_confirmadas == 0
    assert briefing.rodape.afirmacoes_derrubadas == 1


def test_variante_insuficiente_nao_formula_tese_sobre_a_startup(caminho_banco):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_insuficiente(caminho_banco)))

    assert "não sustenta" in briefing.veredito.tese


def test_recomendacoes_vazias_no_caminho_aderente_viram_briefing_insuficiente(
    caminho_banco,
):
    """§11.3: todas as recomendações descartadas terminam na variante terminal."""
    estado = estado_normal(caminho_banco, recomendacoes=[], fit_score=None)
    no, provedor = montar_no(caminho_banco)

    briefing = briefing_de(no(estado))

    assert provedor.chamadas == 0
    assert briefing.variante == "evidencia_insuficiente"
    assert briefing.rodape.rota_r3 == "prosseguir"


# ----------------------------------------------------------------------
# Avisos e rodapé determinísticos
# ----------------------------------------------------------------------


def test_confianca_baixa_e_criterios_relaxados_viram_avisos(caminho_banco):
    estado = estado_normal(
        caminho_banco,
        confianca_perfil="baixa",
        criterios_relaxados=["estagio", "localizacao"],
    )
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado))

    assert any("confiança" in aviso.lower() for aviso in briefing.avisos)
    assert any("estagio" in aviso for aviso in briefing.avisos)


def test_avisos_sao_deterministicos_entre_execucoes_identicas(caminho_banco):
    estado = estado_normal(caminho_banco, confianca_perfil="baixa")
    primeiro, _ = montar_no(caminho_banco, rascunho())
    segundo, _ = montar_no(caminho_banco, rascunho())

    um = briefing_de(primeiro(estado))
    dois = briefing_de(segundo(estado_normal(caminho_banco, confianca_perfil="baixa")))

    assert um.avisos == dois.avisos
    assert um.fontes == dois.fontes


def test_rodape_preserva_rubrica_datas_contagens_e_trajeto(caminho_banco):
    estado = estado_normal(caminho_banco)
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado))

    assert briefing.rodape.versao_rubrica == "rubrica-v1"
    assert briefing.rodape.data_execucao == DATA_FIXA
    assert briefing.rodape.afirmacoes_confirmadas == 2
    assert briefing.rodape.afirmacoes_derrubadas == 0
    assert briefing.rodape.rota_r3 == "prosseguir"
    assert briefing.rodape.trajeto == list(estado["trajeto"]) + ["briefing"]
    assert briefing.rodape.trajeto.count("briefing") == 1


def test_o_relogio_injetado_governa_as_duas_datas(caminho_banco):
    no = AgenteBriefing(
        BaseStartups(caminho_banco),
        ProvedorSequencialFalso(rascunho()),
        relogio=lambda: date(2027, 1, 15),
    )

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert briefing.cabecalho.data_geracao == date(2027, 1, 15)
    assert briefing.rodape.data_execucao == date(2027, 1, 15)


def test_briefing_do_estado_atravessa_a_forma_json_do_checkpoint(caminho_banco):
    no, _ = montar_no(caminho_banco, rascunho())

    saida = no(estado_normal(caminho_banco))

    assert isinstance(saida["briefing"], dict)
    assert Recomendacao.model_validate(
        saida["briefing"]["recomendacoes"][0]
    ).gap_enderecado == "distribuicao"
    assert isinstance(
        PerfilValidado.model_validate(
            estado_normal(caminho_banco)["perfil_validado"]
        ),
        PerfilValidado,
    )


# ----------------------------------------------------------------------
# Rastreabilidade da recomendação embutida contra o estado ATUAL
# ----------------------------------------------------------------------
#
# Revalidar só a forma Pydantic deixa passar uma recomendação estruturalmente
# válida porém envenenada ou velha. Cada teste abaixo altera exatamente um elo
# da cadeia de proveniência e exige recusa antes de qualquer chamada de LLM.


def estado_com_recomendacao(caminho_banco, recomendacao, **ajustes):
    return estado_normal(
        caminho_banco,
        recomendacoes=[recomendacao.model_dump(mode="json")],
        **ajustes,
    )


def test_recomendacao_com_documento_trocado_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    # A afirmação 1 vive no documento 0; a recomendação aponta para o 1.
    envenenada = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[1]["id_documento"],
        url_fonte=linhas[1]["url_fonte"],
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="documento"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_com_url_trocada_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    envenenada = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[1]["url_fonte"],
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="url_fonte"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_com_trecho_adulterado_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    envenenada = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[0]["url_fonte"],
        trecho_citado="Trecho adulterado que a afirmação validada não sustenta.",
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="trecho_citado"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_apoiada_em_afirmacao_derrubada_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_de_dois_documentos(linhas, derrubar_segunda=True)
    envenenada = recomendacao_falsa(
        id_afirmacao=2,
        id_documento=linhas[1]["id_documento"],
        url_fonte=linhas[1]["url_fonte"],
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="confirmad"):
        no(
            estado_com_recomendacao(
                caminho_banco, envenenada, perfil_validado=perfil
            )
        )

    assert provedor.chamadas == 0


def test_recomendacao_com_afirmacao_inexistente_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    envenenada = recomendacao_falsa(
        id_afirmacao=999,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[0]["url_fonte"],
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="999"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_com_documento_de_outra_startup_e_recusada(caminho_banco):
    """Proveniência coerente entre si, mas de outra empresa: recusada."""
    linhas = documentos_caju(caminho_banco)
    intrusa = outra_startup(caminho_banco)
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "distribuicao", polaridade="ausencia_explicita",
                id_documento=linhas[0]["id_documento"],
            ),
            afirmacao_validada_falsa(
                2, "escala_e_dor_operacional",
                id_documento=intrusa["id_documento"],
            ),
        ],
        hosts=[linhas[0]["dominio_fonte"]],
    )
    envenenada = recomendacao_falsa(
        id_afirmacao=2,
        id_documento=intrusa["id_documento"],
        url_fonte=intrusa["url_fonte"],
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="outra startup|não pertence"):
        no(
            estado_com_recomendacao(
                caminho_banco, envenenada, perfil_validado=perfil
            )
        )

    assert provedor.chamadas == 0


def test_recomendacao_com_chunk_nvidia_inexistente_e_recusada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    envenenada = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[0]["url_fonte"],
        citacao=citacao_nvidia_falsa(999),
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="999"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_com_citacao_divergente_do_chunk_atual_e_recusada(
    caminho_banco,
):
    """O id existe no contexto atual, mas os metadados são de outra coisa."""
    linhas = documentos_caju(caminho_banco)
    envenenada = recomendacao_falsa(
        id_afirmacao=1,
        id_documento=linhas[0]["id_documento"],
        url_fonte=linhas[0]["url_fonte"],
        citacao=citacao_nvidia_falsa(
            101, breadcrumb="Trilha inventada > seção falsa"
        ),
    )
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="breadcrumb|não corresponde"):
        no(estado_com_recomendacao(caminho_banco, envenenada))

    assert provedor.chamadas == 0


def test_recomendacao_com_contexto_nvidia_ausente_e_recusada(caminho_banco):
    """Contexto velho ou removido não sustenta a citação embutida."""
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing):
        no(estado_normal(caminho_banco, contexto_nvidia=None))

    assert provedor.chamadas == 0


def test_recomendacao_integra_atravessa_a_conferencia(caminho_banco):
    """A cadeia coerente continua passando: a guarda não é um bloqueio geral."""
    no, provedor = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert provedor.chamadas == 1
    assert len(briefing.recomendacoes) == 1


# ----------------------------------------------------------------------
# A consulta original nunca é fabricada
# ----------------------------------------------------------------------


def test_consulta_ausente_no_estado_falha_sem_chamar_provedor(caminho_banco):
    estado = estado_normal(caminho_banco)
    del estado["consulta_usuario"]
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="consulta"):
        no(estado)

    assert provedor.chamadas == 0


@pytest.mark.parametrize("consulta", ["", "   ", None])
def test_consulta_em_branco_ou_nula_falha_sem_chamar_provedor(
    caminho_banco, consulta
):
    no, provedor = montar_no(caminho_banco, rascunho())

    with pytest.raises(ErroBriefing, match="consulta"):
        no(estado_normal(caminho_banco, consulta_usuario=consulta))

    assert provedor.chamadas == 0


def test_nenhum_briefing_carrega_texto_de_consulta_fabricado(caminho_banco):
    no, _ = montar_no(caminho_banco, rascunho())

    briefing = briefing_de(no(estado_normal(caminho_banco)))

    assert "não registrada" not in briefing.cabecalho.consulta_original


# ----------------------------------------------------------------------
# §11.3: as afirmações derrubadas chegam ao público com id e motivo
# ----------------------------------------------------------------------


def perfil_com_duas_derrubadas(linhas):
    return perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "outro", id_documento=linhas[0]["id_documento"],
                situacao="derrubada",
                motivo="Trecho ausente no documento citado.",
            ),
            afirmacao_validada_falsa(
                2, "outro", id_documento=linhas[1]["id_documento"],
                situacao="derrubada",
                motivo="Citação diverge da fonte pública.",
            ),
        ],
        hosts=[],
    )


def test_avisos_preservam_id_e_motivo_de_cada_afirmacao_derrubada(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    estado = estado_insuficiente(
        caminho_banco, perfil_validado=perfil_com_duas_derrubadas(linhas)
    )
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado))

    texto = "\n".join(briefing.avisos)
    assert "Trecho ausente no documento citado." in texto
    assert "Citação diverge da fonte pública." in texto
    # Cada motivo viaja colado ao id que ele explica.
    por_id = {
        identificador: aviso
        for identificador in ("1", "2")
        for aviso in briefing.avisos
        if f"afirmação {identificador}" in aviso.casefold()
    }
    assert "Trecho ausente no documento citado." in por_id["1"]
    assert "Citação diverge da fonte pública." in por_id["2"]


def test_avisos_de_derrubada_seguem_ordem_deterministica_por_id(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    estado = estado_insuficiente(
        caminho_banco, perfil_validado=perfil_com_duas_derrubadas(linhas)
    )
    primeiro, _ = montar_no(caminho_banco)
    segundo, _ = montar_no(caminho_banco)

    um = briefing_de(primeiro(estado))
    dois = briefing_de(segundo(estado))

    assert um.avisos == dois.avisos
    posicoes = [
        indice
        for indice, aviso in enumerate(um.avisos)
        if "Trecho ausente" in aviso or "Citação diverge" in aviso
    ]
    assert posicoes == sorted(posicoes)
    assert um.avisos.index(
        next(a for a in um.avisos if "Trecho ausente" in a)
    ) < um.avisos.index(
        next(a for a in um.avisos if "Citação diverge" in a)
    )


def test_motivo_da_derrubada_nao_expoe_documento_completo(caminho_banco):
    linhas = documentos_caju(caminho_banco)
    with sqlite3.connect(caminho_banco) as conexao:
        conteudo = conexao.execute(
            "SELECT conteudo_texto FROM documentos WHERE id = ?",
            (linhas[0]["id_documento"],),
        ).fetchone()[0]
    estado = estado_insuficiente(
        caminho_banco, perfil_validado=perfil_com_duas_derrubadas(linhas)
    )
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado))

    assert conteudo[:120] not in "\n".join(briefing.avisos)


# ----------------------------------------------------------------------
# §11.3: a síntese de insuficiência precisa dizer a causa REAL
# ----------------------------------------------------------------------
#
# São três situações distintas, e só a primeira é falha de proveniência. Uma
# síntese fixa afirmando que "as afirmações não passaram na conferência" é
# falsa quando R3 devolveu prosseguir sobre evidência confirmada e o que
# faltou foi recomendação sustentada.


def estado_suporte_derrubado(caminho_banco, **ajustes):
    """R3 encerra por suporte da classe derrubado — com evidência viva."""
    linhas = documentos_caju(caminho_banco)
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "distribuicao", polaridade="ausencia_explicita",
                id_documento=linhas[0]["id_documento"],
            ),
            afirmacao_validada_falsa(
                2, "outro", id_documento=linhas[1]["id_documento"],
                situacao="derrubada",
                motivo="Trecho não ocorre na fonte indicada.",
            ),
        ],
        hosts=[linhas[0]["dominio_fonte"]],
    )
    return estado_normal(
        caminho_banco,
        perfil_validado=perfil,
        classificacao=Classificacao(
            classe="AI-enabled",
            justificativa=(
                "A plataforma de benefícios é o produto contratado pelas empresas. "
                "O material público não descreve modelos próprios como o produto."
            ),
            ids_afirmacoes_suporte=[2],
        ),
        recomendacoes=None,
        fit_score=None,
        **ajustes,
    )


def estado_sem_recomendacao_sustentada(caminho_banco, **ajustes):
    """R3 devolveu prosseguir, mas nenhuma recomendação sobreviveu."""
    return estado_normal(
        caminho_banco, recomendacoes=[], fit_score=None, **ajustes
    )


CAUSAS_DE_INSUFICIENCIA = (
    "sem_afirmacao_confirmada",
    "suporte_da_classe_derrubado",
    "sem_recomendacao_sustentada",
)


def estado_da_causa(caminho_banco, causa):
    if causa == "sem_afirmacao_confirmada":
        return estado_insuficiente(caminho_banco)
    if causa == "suporte_da_classe_derrubado":
        return estado_suporte_derrubado(caminho_banco)
    return estado_sem_recomendacao_sustentada(caminho_banco)


@pytest.mark.parametrize("causa", CAUSAS_DE_INSUFICIENCIA)
def test_cada_causa_de_insuficiencia_tem_sintese_e_aviso_da_mesma_origem(
    caminho_banco, causa
):
    """Síntese e aviso principal derivam do mesmo valor de causa."""
    from radar.agentes.briefing import AVISO_POR_CAUSA, SINTESE_POR_CAUSA

    no, provedor = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_da_causa(caminho_banco, causa)))

    assert provedor.chamadas == 0
    assert briefing.sintese_executiva.texto == SINTESE_POR_CAUSA[causa]
    assert AVISO_POR_CAUSA[causa] in briefing.avisos


def test_as_tres_causas_produzem_sinteses_distintas(caminho_banco):
    textos = set()
    for causa in CAUSAS_DE_INSUFICIENCIA:
        no, provedor = montar_no(caminho_banco)
        briefing = briefing_de(no(estado_da_causa(caminho_banco, causa)))
        assert provedor.chamadas == 0
        textos.add(briefing.sintese_executiva.texto)

    assert len(textos) == len(CAUSAS_DE_INSUFICIENCIA)


def test_sintese_sem_recomendacao_nao_alega_falha_de_proveniencia(caminho_banco):
    """O defeito corrigido: a evidência passou, o que faltou foi recomendação."""
    no, provedor = montar_no(caminho_banco)

    briefing = briefing_de(
        no(estado_sem_recomendacao_sustentada(caminho_banco))
    )

    texto = briefing.sintese_executiva.texto.casefold()
    assert "não passaram" not in texto
    assert "proveniência" not in texto
    assert "recomendaç" in texto
    assert briefing.rodape.rota_r3 == "prosseguir"
    assert briefing.rodape.afirmacoes_confirmadas >= 1
    assert provedor.chamadas == 0


def test_sintese_do_suporte_derrubado_nomeia_o_teto_de_reextracao(caminho_banco):
    no, provedor = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_suporte_derrubado(caminho_banco)))

    texto = briefing.sintese_executiva.texto.casefold()
    assert "classificação" in texto
    assert "reextração" in texto
    assert briefing.rodape.rota_r3 == "evidencia_insuficiente"
    assert provedor.chamadas == 0


@pytest.mark.parametrize("causa", CAUSAS_DE_INSUFICIENCIA)
def test_sintese_de_insuficiencia_concorda_com_as_contagens_do_rodape(
    caminho_banco, causa
):
    no, _ = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_da_causa(caminho_banco, causa)))

    texto = briefing.sintese_executiva.texto.casefold()
    afirma_nenhuma_confirmada = "nenhuma afirmação" in texto
    assert afirma_nenhuma_confirmada == (
        briefing.rodape.afirmacoes_confirmadas == 0
    )
    if not afirma_nenhuma_confirmada:
        assert briefing.rodape.afirmacoes_confirmadas >= 1


@pytest.mark.parametrize("causa", CAUSAS_DE_INSUFICIENCIA)
def test_as_tres_causas_preservam_o_contrato_da_variante_insuficiente(
    caminho_banco, causa
):
    no, provedor = montar_no(caminho_banco)

    briefing = briefing_de(no(estado_da_causa(caminho_banco, causa)))

    assert provedor.chamadas == 0
    assert briefing.variante == "evidencia_insuficiente"
    assert briefing.veredito.classe is None
    assert briefing.veredito.fit_score_total is None
    assert briefing.veredito.ids_afirmacoes_suporte == []
    assert briefing.sintese_executiva.ids_afirmacoes_suporte == []
    assert briefing.pontos_de_conversa == []
    assert briefing.recomendacoes == []
    assert briefing.fontes == []
    assert briefing.avisos
    # A tese genérica do veredito é preservada nas três causas.
    assert briefing.veredito.tese == (
        "A base disponível não sustenta uma conclusão sobre esta empresa."
    )
