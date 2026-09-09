"""Contratos de recomendação, fit-score e relatório, validados sem nó nenhum.

Além da forma de cada modelo, aqui mora a regra de lastro cruzado: exigir
apenas "ao menos uma citação de origem tecnologia" deixaria passar o pior caso
— recomendar CUDA sustentando com um chunk de NIM. O trecho existe, a
proveniência formal fecha, e mesmo assim a recomendação não tem lastro sobre a
tecnologia que ela oferece.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from radar.contratos import (
    CitacaoNvidia,
    EvidenciaStartup,
    FitScore,
    EntradaFitScore,
    MetadadoDocumentoFitScore,
    PilarFitScore,
    ProximaAcao,
    Recomendacao,
    RecomendacaoRascunho,
    RelatorioRecomendacoes,
)
from tests.conftest import citacao_nvidia_falsa, recomendacao_falsa

def rascunho_valido() -> dict:
    return {
        "tipo_fundamento": "gap_confirmado",
        "identificador_fundamento": "otimizacao_tecnica",
        "tecnologias": ["NVIDIA NIM"],
        "justificativa_tecnica": "NIM permite servir modelos por API.",
        "justificativa_negocio": "A adoção pode reduzir atrito operacional.",
        "proxima_acao": {
            "tipo_acao": "poc_nim",
            "detalhe": "Validar uma prova de conceito com carga representativa.",
        },
        "ids_afirmacoes": [1],
        "ids_chunks": [10],
    }


def recomendacao_valida(gap: str = "otimizacao_tecnica") -> Recomendacao:
    return Recomendacao(
        tipo_fundamento="gap_confirmado",
        identificador_fundamento=gap,
        tecnologias=["NVIDIA NIM"],
        justificativa_tecnica="NIM permite servir modelos por uma API otimizada.",
        justificativa_negocio="A adoção pode reduzir o atrito operacional.",
        prioridade="alta",
        complexidade="media",
        proxima_acao=ProximaAcao(
            tipo_acao="poc_nim",
            detalhe="Validar uma prova de conceito com carga representativa.",
        ),
        evidencias_startup=[
            EvidenciaStartup(
                id_afirmacao=1,
                id_documento=7,
                url_fonte="https://startup.example/engenharia",
                trecho_citado="A plataforma depende de uma API externa para inferência.",
            )
        ],
        citacoes_nvidia=[
            CitacaoNvidia(
                id_chunk=10,
                topico="nim",
                origem="tecnologia",
                tecnologia="NVIDIA NIM",
                fonte_url="https://nvidia.example/nim",
                breadcrumb="NIM > Visão geral",
            )
        ],
    )


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("prioridade", "alta"),
        ("complexidade", "media"),
        ("fit_score", 80),
        ("evidencias_startup", []),
        ("citacoes_nvidia", []),
    ],
)
def test_rascunho_proibe_campos_que_o_llm_nao_produz(campo, valor):
    payload = rascunho_valido()
    payload[campo] = valor
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecomendacaoRascunho.model_validate(payload)


def test_rascunho_aceita_somente_ids_e_campos_redacionais():
    rascunho = RecomendacaoRascunho.model_validate(rascunho_valido())
    assert rascunho.ids_afirmacoes == [1]
    assert rascunho.ids_chunks == [10]
    assert "prioridade" not in rascunho.model_dump()
    assert "complexidade" not in rascunho.model_dump()


def test_proxima_acao_exige_uma_unica_frase():
    with pytest.raises(ValidationError, match="exatamente uma frase"):
        ProximaAcao(
            tipo_acao="poc_nim",
            detalhe="Faça a prova de conceito. Depois compare a latência.",
        )


def test_recomendacao_valida_preserva_proveniencia_dos_dois_lados():
    recomendacao = recomendacao_valida()
    assert recomendacao.evidencias_startup[0].id_afirmacao == 1
    assert recomendacao.evidencias_startup[0].id_documento == 7
    assert str(recomendacao.evidencias_startup[0].url_fonte).startswith(
        "https://startup.example/"
    )
    assert recomendacao.citacoes_nvidia[0].id_chunk == 10
    assert recomendacao.citacoes_nvidia[0].tecnologia == "NVIDIA NIM"
    assert str(recomendacao.citacoes_nvidia[0].fonte_url).startswith(
        "https://nvidia.example/"
    )


def test_recomendacao_rejeita_ausencia_de_evidencia_da_startup():
    payload = recomendacao_valida().model_dump(mode="json")
    payload["evidencias_startup"] = []
    with pytest.raises(ValidationError):
        Recomendacao.model_validate(payload)


def test_recomendacao_rejeita_ausencia_de_citacao_nvidia():
    payload = recomendacao_valida().model_dump(mode="json")
    payload["citacoes_nvidia"] = []
    with pytest.raises(ValidationError):
        Recomendacao.model_validate(payload)


def test_recomendacao_rejeita_suporte_apenas_conceitual():
    payload = recomendacao_valida().model_dump(mode="json")
    payload["citacoes_nvidia"] = [
        {
            "id_chunk": 11,
            "topico": "ai-native-services",
            "origem": "conceitual",
            "tecnologia": None,
            "fonte_url": "https://conceito.example/ai-native",
            "breadcrumb": "AI-native services > Introdução",
        }
    ]
    with pytest.raises(ValidationError, match="chunk de tecnologia"):
        Recomendacao.model_validate(payload)


def test_recomendacao_rejeita_tecnologia_fora_das_dezesseis_do_tapi():
    payload = recomendacao_valida().model_dump(mode="json")
    payload["tecnologias"] = ["Produto inventado"]
    with pytest.raises(ValidationError):
        Recomendacao.model_validate(payload)


def test_relatorio_limita_recomendacoes_a_cinco():
    """Seis gaps distintos: isola o teto de tamanho da guarda de unicidade."""
    seis_gaps_distintos = [
        recomendacao_valida(gap)
        for gap in (
            "dados_proprietarios",
            "workflow_profundo",
            "distribuicao",
            "otimizacao_tecnica",
            "dependencia_api_externa",
            "escala_e_dor_operacional",
        )
    ]
    with pytest.raises(ValidationError):
        RelatorioRecomendacoes(recomendacoes=seis_gaps_distintos)


def test_relatorio_recusa_dois_pacotes_para_o_mesmo_gap():
    """§6.1 — um pacote coeso por gap, garantido também no contrato."""
    with pytest.raises(ValidationError) as erro:
        RelatorioRecomendacoes(
            recomendacoes=[
                recomendacao_valida("otimizacao_tecnica"),
                recomendacao_valida("distribuicao"),
                recomendacao_valida("otimizacao_tecnica"),
            ]
        )
    assert "otimizacao_tecnica" in str(erro.value)


def test_relatorio_aceita_gaps_distintos_ate_o_teto():
    relatorio = RelatorioRecomendacoes(
        recomendacoes=[
            recomendacao_valida("dados_proprietarios"),
            recomendacao_valida("workflow_profundo"),
            recomendacao_valida("distribuicao"),
            recomendacao_valida("otimizacao_tecnica"),
            recomendacao_valida("dependencia_api_externa"),
        ]
    )
    gaps = [item.identificador_fundamento for item in relatorio.recomendacoes]
    assert len(gaps) == len(set(gaps)) == 5


def test_relatorio_permite_variante_terminal_sem_recomendacao():
    assert RelatorioRecomendacoes().recomendacoes == []


def test_fit_score_rejeita_total_que_nao_corresponde_aos_pilares():
    pilares = [
        PilarFitScore(
            pilar=nome,
            pontos=1,
            faixa="baixa",
            ids_evidencias=[],
            travas_aplicadas=[],
        )
        for nome in (
            "centralidade_ia",
            "gap_enderecavel",
            "momento",
            "alinhamento_setorial",
        )
    ]
    dimensoes = [
        {"dimensao": dimensao, "estado": "desconhecido", "ids_evidencias": []}
        for dimensao in (
            "dados_proprietarios",
            "workflow_profundo",
            "distribuicao",
            "otimizacao_tecnica",
        )
    ]
    with pytest.raises(ValidationError, match="normalização"):
        FitScore(
            total=99,
            pilares=pilares,
            estado_dimensoes_gap=dimensoes,
            justificativa_curta="Pontuação de teste.",
            versao_rubrica="rubrica-v1",
        )


def test_fit_score_rejeita_gate_non_ai_com_pilar_nao_zerado():
    pilares = [
        PilarFitScore(
            pilar=nome,
            pontos=1 if nome == "centralidade_ia" else 0,
            faixa="baixa",
            ids_evidencias=[],
            travas_aplicadas=["gate_non_ai"],
        )
        for nome in (
            "centralidade_ia",
            "gap_enderecavel",
            "momento",
            "alinhamento_setorial",
        )
    ]
    dimensoes = [
        {"dimensao": dimensao, "estado": "desconhecido", "ids_evidencias": []}
        for dimensao in (
            "dados_proprietarios",
            "workflow_profundo",
            "distribuicao",
            "otimizacao_tecnica",
        )
    ]
    with pytest.raises(ValidationError, match="zerar pilares e total"):
        FitScore(
            total=0,
            pilares=pilares,
            estado_dimensoes_gap=dimensoes,
            justificativa_curta="Gate global de teste.",
            versao_rubrica="rubrica-v1",
        )


def test_data_referencia_e_metadados_sao_entradas_explicitas_do_score():
    assert EntradaFitScore.model_fields["data_referencia"].is_required()
    assert MetadadoDocumentoFitScore.model_fields["data_publicacao"].annotation == (
        date | None
    )


def test_metadado_rejeita_host_que_nao_corresponde_a_url():
    with pytest.raises(ValidationError, match="corresponder a url_fonte"):
        MetadadoDocumentoFitScore(
            id_documento=1,
            url_fonte="https://fonte-correta.example/artigo",
            host_normalizado="outra-fonte.example",
            data_publicacao=None,
        )


# ------------------------------------------------------------------------
# Toda tecnologia recomendada exige citação daquela mesma tecnologia
# ------------------------------------------------------------------------

def test_cuda_sustentado_apenas_por_chunk_de_nim_e_recusado():
    with pytest.raises(ValidationError, match="CUDA"):
        recomendacao_falsa(
            "otimizacao_tecnica",
            tecnologias=["CUDA"],
            citacao=citacao_nvidia_falsa(101, tecnologia="NVIDIA NIM"),
        )


def test_a_citacao_da_propria_tecnologia_e_aceita():
    recomendacao = recomendacao_falsa(
        "otimizacao_tecnica",
        tecnologias=["CUDA"],
        citacao=citacao_nvidia_falsa(101, tecnologia="CUDA"),
    )

    assert recomendacao.tecnologias == ["CUDA"]


def test_contexto_conceitual_adicional_continua_permitido():
    from radar.contratos import Recomendacao

    base = recomendacao_falsa(
        "otimizacao_tecnica",
        tecnologias=["CUDA"],
        citacao=citacao_nvidia_falsa(101, tecnologia="CUDA"),
    )
    completa = Recomendacao(
        **{
            **base.model_dump(),
            "citacoes_nvidia": [
                *[item.model_dump() for item in base.citacoes_nvidia],
                citacao_nvidia_falsa(106, tecnologia=None, origem="conceitual").model_dump(),
            ],
        }
    )

    assert len(completa.citacoes_nvidia) == 2


def test_toda_tecnologia_de_uma_recomendacao_multipla_precisa_de_lastro():
    from radar.contratos import Recomendacao

    base = recomendacao_falsa(
        "otimizacao_tecnica",
        tecnologias=["CUDA"],
        citacao=citacao_nvidia_falsa(101, tecnologia="CUDA"),
    )
    with pytest.raises(ValidationError, match="TensorRT-LLM"):
        Recomendacao(
            **{
                **base.model_dump(),
                "tecnologias": ["CUDA", "TensorRT-LLM"],
            }
        )
