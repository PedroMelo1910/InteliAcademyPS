from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from radar.contratos import (
    AfirmacaoValidada,
    EntradaFitScore,
    EstadoDimensaoGap,
    MetadadoDocumentoFitScore,
    PerfilValidado,
)
from radar.recomendacao import (
    _aplicar_travas,
    _faixa,
    _pilar,
    calcular_fit_score,
)


DIMENSOES = (
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
)


def afirmacao(
    id_afirmacao: int,
    categoria: str,
    *,
    polaridade: str | None = None,
    situacao: str = "confirmada",
    id_documento: int | None = None,
) -> AfirmacaoValidada:
    if polaridade is None:
        polaridade = "presenca" if categoria in DIMENSOES else "neutro"
    return AfirmacaoValidada(
        id_afirmacao=id_afirmacao,
        texto=f"A evidência número {id_afirmacao} está documentada.",
        categoria=categoria,
        polaridade=polaridade,
        id_documento=id_documento or id_afirmacao,
        trecho_citado=f"Trecho público verificável para a evidência {id_afirmacao} citada.",
        situacao=situacao,
        motivo=None if situacao == "confirmada" else "Trecho não ocorre na fonte.",
    )


def perfil(itens: list[AfirmacaoValidada]) -> PerfilValidado:
    estados = []
    for dimensao in DIMENSOES:
        presencas = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "presenca"
        )
        ausencias = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "ausencia_explicita"
        )
        if presencas and ausencias:
            estado, ids = "desconhecido", sorted(presencas + ausencias)
        elif presencas:
            estado, ids = "capacidade_confirmada", presencas
        elif ausencias:
            estado, ids = "gap_confirmado", ausencias
        else:
            estado, ids = "desconhecido", []
        estados.append(
            EstadoDimensaoGap(
                dimensao=dimensao, estado=estado, ids_evidencias=ids
            )
        )
    derrubadas = sum(item.situacao == "derrubada" for item in itens)
    return PerfilValidado(
        afirmacoes_validadas=itens,
        taxa_derrubada=derrubadas / len(itens),
        hosts_distintos=[],
        estado_dimensoes_gap=estados,
    )


def entrada(
    itens: list[AfirmacaoValidada],
    *,
    classe: str = "AI-native",
    setor: str = "Saúde",
    estagio: str = "seed",
    suporte_classe: list[int] | None = None,
    hosts: dict[int, str] | None = None,
    datas: dict[int, date | None] | None = None,
    data_referencia: date = date(2026, 9, 2),
) -> EntradaFitScore:
    hosts = hosts or {}
    datas = datas or {}
    documentos = []
    for item in itens:
        host = hosts.get(item.id_afirmacao, "fonte-a.example")
        documentos.append(
            MetadadoDocumentoFitScore(
                id_documento=item.id_documento,
                url_fonte=f"https://{host}/documento/{item.id_documento}",
                host_normalizado=host,
                data_publicacao=datas.get(item.id_afirmacao),
            )
        )
    return EntradaFitScore(
        classe=classe,
        ids_afirmacoes_suporte_classe=suporte_classe or [1],
        perfil_validado=perfil(itens),
        setor=setor,
        estagio=estagio,
        documentos=documentos,
        data_referencia=data_referencia,
    )


def pontos(resultado, pilar: str) -> int:
    return next(item.pontos for item in resultado.pilares if item.pilar == pilar)


def trava(resultado, pilar: str) -> list[str]:
    return next(
        item.travas_aplicadas for item in resultado.pilares if item.pilar == pilar
    )


def test_mesma_entrada_produz_saida_idêntica():
    itens = [afirmacao(1, "stack_propria")]
    dados = entrada(itens)
    primeira = calcular_fit_score(dados)
    segunda = calcular_fit_score(dados)
    assert primeira.model_dump(mode="json") == segunda.model_dump(mode="json")


def test_maximo_bruto_36_normaliza_exatamente_para_100():
    itens = [
        afirmacao(1, "stack_propria"),
        afirmacao(2, "stack_propria"),
        afirmacao(3, "stack_propria"),
        afirmacao(4, "equipe_e_contratacao"),
        afirmacao(5, "equipe_e_contratacao"),
        afirmacao(6, "dados_proprietarios", polaridade="ausencia_explicita"),
        afirmacao(7, "workflow_profundo", polaridade="ausencia_explicita"),
        afirmacao(8, "distribuicao", polaridade="ausencia_explicita"),
        afirmacao(9, "otimizacao_tecnica", polaridade="ausencia_explicita"),
        afirmacao(10, "dependencia_api_externa"),
        afirmacao(11, "momento_e_financiamento"),
    ]
    hosts = {
        item.id_afirmacao: (
            "fonte-a.example" if item.id_afirmacao % 2 else "fonte-b.example"
        )
        for item in itens
    }
    datas = {
        4: date(2026, 8, 1),
        5: date(2026, 7, 1),
        11: date(2026, 6, 1),
    }
    resultado = calcular_fit_score(
        entrada(itens, hosts=hosts, datas=datas, suporte_classe=[1, 2])
    )
    assert [item.pontos for item in resultado.pilares] == [10, 10, 9, 7]
    assert resultado.total == 100


def test_normalizacao_intermediaria_usa_a_soma_bruta_e_maximo_36():
    resultado = calcular_fit_score(entrada([afirmacao(1, "outro")]))
    assert [item.pontos for item in resultado.pilares] == [5, 0, 5, 7]
    assert resultado.total == round(100 * 17 / 36) == 47


def test_non_ai_aciona_gate_global_e_zerado():
    resultado = calcular_fit_score(
        entrada([afirmacao(1, "outro")], classe="non-AI", setor="Saúde")
    )
    assert resultado.total == 0
    assert [item.pontos for item in resultado.pilares] == [0, 0, 0, 0]
    assert all(item.travas_aplicadas == ["gate_non_ai"] for item in resultado.pilares)


def test_teto_de_corrobacao_impede_faixa_alta_com_um_host():
    itens = [
        afirmacao(1, "stack_propria"),
        afirmacao(2, "stack_propria"),
        afirmacao(3, "stack_propria"),
        afirmacao(4, "equipe_e_contratacao"),
        afirmacao(5, "equipe_e_contratacao"),
    ]
    resultado = calcular_fit_score(
        entrada(itens, classe="AI-enabled", suporte_classe=[1])
    )
    assert pontos(resultado, "centralidade_ia") == 7
    assert trava(resultado, "centralidade_ia") == ["teto_corrobacao"]


def test_fonte_de_evidencia_que_nao_contribui_nao_remove_teto_do_pilar():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "dados_proprietarios", polaridade="ausencia_explicita"),
        afirmacao(3, "workflow_profundo", polaridade="ausencia_explicita"),
        afirmacao(4, "distribuicao", polaridade="ausencia_explicita"),
        afirmacao(5, "otimizacao_tecnica", polaridade="ausencia_explicita"),
        afirmacao(6, "dependencia_api_externa"),
    ]
    resultado = calcular_fit_score(
        entrada(
            itens,
            suporte_classe=[1],
            hosts={
                1: "fonte-nao-contribuinte.example",
                2: "fonte-gap.example",
                3: "fonte-gap.example",
                4: "fonte-gap.example",
                5: "fonte-gap.example",
                6: "fonte-gap.example",
            },
        )
    )
    assert pontos(resultado, "gap_enderecavel") == 7
    assert trava(resultado, "gap_enderecavel") == ["teto_corrobacao"]


def test_afirmacao_derrubada_nao_pontua_nem_cria_gap():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "stack_propria", situacao="derrubada"),
        afirmacao(
            3,
            "dados_proprietarios",
            polaridade="ausencia_explicita",
            situacao="derrubada",
        ),
    ]
    resultado = calcular_fit_score(
        entrada(itens, classe="AI-enabled", suporte_classe=[1])
    )
    assert pontos(resultado, "centralidade_ia") == 3
    assert pontos(resultado, "gap_enderecavel") == 0
    assert resultado.estado_dimensoes_gap[0].estado == "desconhecido"


def test_conflito_presenca_ausencia_vira_desconhecido_e_nao_pontua_gap():
    itens = [
        afirmacao(1, "dados_proprietarios", polaridade="presenca"),
        afirmacao(2, "dados_proprietarios", polaridade="ausencia_explicita"),
    ]
    resultado = calcular_fit_score(entrada(itens, suporte_classe=[1]))
    estado = resultado.estado_dimensoes_gap[0]
    assert estado.estado == "desconhecido"
    assert estado.ids_evidencias == [1, 2]
    assert pontos(resultado, "gap_enderecavel") == 0


def test_gap_so_pontua_quando_ausencia_explicita_foi_confirmada():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "workflow_profundo", polaridade="ausencia_explicita"),
    ]
    resultado = calcular_fit_score(entrada(itens, suporte_classe=[1]))
    assert pontos(resultado, "gap_enderecavel") == 2
    assert resultado.estado_dimensoes_gap[1].estado == "gap_confirmado"


def test_datas_exatamente_nos_limites_de_18_e_12_meses_pontuam():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "momento_e_financiamento"),
        afirmacao(3, "equipe_e_contratacao"),
    ]
    resultado = calcular_fit_score(
        entrada(
            itens,
            datas={2: date(2025, 3, 2), 3: date(2025, 9, 2)},
            hosts={
                1: "fonte-a.example",
                2: "fonte-a.example",
                3: "fonte-b.example",
            },
            suporte_classe=[1],
        )
    )
    assert pontos(resultado, "momento") == 9
    assert trava(resultado, "momento") == []


def test_datas_antigas_futuras_e_ausentes_nao_pontuam():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "momento_e_financiamento"),
        afirmacao(3, "equipe_e_contratacao"),
        afirmacao(4, "equipe_e_contratacao"),
    ]
    resultado = calcular_fit_score(
        entrada(
            itens,
            estagio="não divulgado",
            datas={
                2: date(2025, 3, 1),
                3: date(2027, 1, 1),
                4: None,
            },
            suporte_classe=[1],
        )
    )
    assert pontos(resultado, "momento") == 1


@pytest.mark.parametrize(
    "estagio,esperado",
    [
        ("pre-seed", 3),
        ("seed", 5),
        ("série A", 5),
        ("série B", 2),
        ("série F", 2),
        ("não divulgado", 1),
        # A base curada aceita texto livre: a grafia inglesa precisa pontuar
        # igual à portuguesa, e não cair na faixa de série B+.
        ("Series A", 5),
        ("series a", 5),
        ("Series B", 2),
        ("Series F", 2),
    ],
)
def test_pontuacao_base_do_estagio(estagio, esperado):
    resultado = calcular_fit_score(
        entrada([afirmacao(1, "outro")], estagio=estagio)
    )
    assert pontos(resultado, "momento") == esperado


@pytest.mark.parametrize(
    "setor,esperado",
    [
        ("Saúde", 5),
        ("Mobilidade autônoma e robótica", 5),
        ("Cibersegurança e criptografia", 5),
        ("Indústria", 5),
        ("Fintech", 3),
        ("Software / automação", 3),
        ("Agronegócio", 2),
    ],
)
def test_tabela_setorial_sem_bonus_de_classe(setor, esperado):
    resultado = calcular_fit_score(
        entrada([afirmacao(1, "outro")], classe="AI-enabled", setor=setor)
    )
    assert pontos(resultado, "alinhamento_setorial") == esperado


def test_justificativa_nao_inventa_gap_quando_dimensoes_sao_desconhecidas():
    resultado = calcular_fit_score(entrada([afirmacao(1, "outro")]))
    assert "gap não confirmado na base" in resultado.justificativa_curta


def test_justificativa_escolhe_primeiro_gap_na_ordem_do_contrato():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "workflow_profundo", polaridade="ausencia_explicita"),
        afirmacao(3, "distribuicao", polaridade="ausencia_explicita"),
    ]
    resultado = calcular_fit_score(entrada(itens, suporte_classe=[1]))
    assert "gap dominante confirmado: workflow profundo" in resultado.justificativa_curta


def test_entrada_rejeita_suporte_de_classe_derrubado():
    itens = [
        afirmacao(1, "outro"),
        afirmacao(2, "stack_propria", situacao="derrubada"),
    ]
    with pytest.raises(ValidationError, match="apenas afirmações confirmadas"):
        entrada(itens, suporte_classe=[2])


def test_entrada_rejeita_metadado_de_documento_ausente():
    item = afirmacao(1, "outro")
    with pytest.raises(ValidationError, match="faltam metadados"):
        EntradaFitScore(
            classe="AI-enabled",
            ids_afirmacoes_suporte_classe=[1],
            perfil_validado=perfil([item]),
            setor="Software",
            estagio="seed",
            documentos=[
                MetadadoDocumentoFitScore(
                    id_documento=99,
                    url_fonte="https://fonte.example/outro",
                    host_normalizado="fonte.example",
                    data_publicacao=None,
                )
            ],
            data_referencia=date(2026, 9, 2),
        )


# ----------------------------------------------------------------------
# Gate de evidência (§7.3a / §9.3): pilar sem evidência não passa de 5
# ----------------------------------------------------------------------


def test_gate_de_evidencia_trava_em_cinco_pilar_sem_evidencia_contribuinte():
    """§9.3 — "pilar sem evidência confirmada não passa de 5".

    O contrato público de ``EntradaFitScore`` exige metadado para toda
    afirmação validada, o que torna este ramo inalcançável pela entrada
    válida. O gate é então exercitado direto no helper puro responsável por
    ele, sem inventar uma entrada de produção inválida.
    """
    pontos, travas = _aplicar_travas(9, [1, 2], {})

    assert pontos == 5
    assert travas == ["gate_evidencia"]
    # O gate de corroboração por host não pode ser reportado no lugar do gate
    # de evidência ausente: são proteções diferentes, com causas diferentes.
    assert "teto_corrobacao" not in travas
    assert _faixa(pontos) == "media"


def test_gate_de_evidencia_mantem_faixa_coerente_no_pilar_construido():
    pilar = _pilar("centralidade_ia", 10, [2, 1, 2], {})

    assert pilar.pontos == 5
    assert pilar.faixa == "media"
    assert pilar.travas_aplicadas == ["gate_evidencia"]
    assert pilar.ids_evidencias == [1, 2]


def test_evidencia_com_fonte_nao_aciona_o_gate_de_evidencia():
    """Contraprova: com metadado presente o gate não dispara."""
    metadados = {
        1: MetadadoDocumentoFitScore(
            id_documento=1,
            url_fonte="https://fonte-a.example/documento/1",
            host_normalizado="fonte-a.example",
            data_publicacao=None,
        ),
        2: MetadadoDocumentoFitScore(
            id_documento=2,
            url_fonte="https://fonte-b.example/documento/2",
            host_normalizado="fonte-b.example",
            data_publicacao=None,
        ),
    }
    pontos, travas = _aplicar_travas(9, [1, 2], metadados)

    assert pontos == 9
    assert travas == []


def test_todo_pilar_acima_de_cinco_tem_evidencia_confirmada_contribuinte():
    """Invariante no nível público: acima de 5 só se houver evidência citada."""
    itens = [
        afirmacao(1, "stack_propria"),
        afirmacao(2, "stack_propria", id_documento=2),
        afirmacao(3, "equipe_e_contratacao", id_documento=3),
        afirmacao(
            4, "dados_proprietarios", polaridade="ausencia_explicita", id_documento=4
        ),
        afirmacao(
            5, "workflow_profundo", polaridade="ausencia_explicita", id_documento=5
        ),
        afirmacao(6, "distribuicao", polaridade="ausencia_explicita", id_documento=6),
        afirmacao(7, "dependencia_api_externa", id_documento=7),
    ]
    resultado = calcular_fit_score(
        entrada(
            itens,
            suporte_classe=[1],
            hosts={
                indice: "fonte-a.example" if indice % 2 else "fonte-b.example"
                for indice in range(1, 8)
            },
        )
    )

    acima_de_cinco = [item for item in resultado.pilares if item.pontos > 5]
    assert acima_de_cinco, "o cenário precisa produzir ao menos um pilar acima de 5"
    for pilar in acima_de_cinco:
        assert pilar.ids_evidencias
        assert "gate_evidencia" not in pilar.travas_aplicadas
