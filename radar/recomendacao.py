"""Núcleo determinístico do motor de recomendação.

Este módulo não chama LLM, não consulta banco e não lê relógio global. O
chamador precisa fornecer inclusive a data de referência e os metadados das
fontes, tornando o fit-score reproduzível e fácil de auditar.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date

from radar.contratos import (
    AfirmacaoValidada,
    EntradaFitScore,
    FaixaFit,
    FitScore,
    MetadadoDocumentoFitScore,
    PilarFit,
    PilarFitScore,
    TravaFit,
)


VERSAO_RUBRICA = "rubrica-v1"
MAXIMO_BRUTO_FIT_SCORE = 36

_SETORES_VERTICAL_DEDICADA = (
    "saude",
    "health",
    "medic",
    "hospital",
    "clinica",
    "voz",
    "audio",
    "call center",
    "transcri",
    "robot",
    "autonom",
    "ciber",
    "cyber",
    "seguranca digital",
    "fraude",
    "identidade digital",
    "industria",
    "industrial",
    "manufatura",
    "simulacao",
    "digital twin",
)

_SETORES_STACK_GENERICA = (
    "inteligencia artificial",
    "software",
    "automacao",
    "dados",
    "fintech",
    "juridico",
    "educacao",
    "logistica",
    "mobilidade",
    "govtech",
    "rh",
    "recursos humanos",
    "contabilidade digital",
    "seguros",
    "insurtech",
    "semicondutores",
    "microeletronica",
    "acessibilidade digital",
)

_ROTULOS_GAP = {
    "dados_proprietarios": "dados proprietários",
    "workflow_profundo": "workflow profundo",
    "distribuicao": "distribuição",
    "otimizacao_tecnica": "otimização técnica",
}


def _texto_busca(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", valor.casefold())
    sem_acentos = "".join(
        caractere for caractere in normalizado if not unicodedata.combining(caractere)
    )
    return " ".join(sem_acentos.split())


def _faixa(pontos: int) -> FaixaFit:
    if pontos <= 3:
        return "baixa"
    if pontos <= 7:
        return "media"
    return "alta"


def _subtrair_meses(valor: date, meses: int) -> date:
    indice_mes = valor.year * 12 + valor.month - 1 - meses
    ano, mes_zero = divmod(indice_mes, 12)
    mes = mes_zero + 1
    dia = min(valor.day, calendar.monthrange(ano, mes)[1])
    return date(ano, mes, dia)


def _publicada_no_intervalo(
    metadado: MetadadoDocumentoFitScore,
    data_referencia: date,
    meses: int,
) -> bool:
    if metadado.data_publicacao is None:
        return False
    return (
        _subtrair_meses(data_referencia, meses)
        <= metadado.data_publicacao
        <= data_referencia
    )


def _pontos_estagio(estagio: str) -> int:
    valor = _texto_busca(estagio).replace("–", "-").replace("—", "-")
    if re.search(r"\bpre[ -]?seed\b", valor):
        return 3
    # ``series?`` aceita as duas grafias, "série A" e "Series A", depois da
    # normalização sem acento. A ordem das duas checagens é carga: a de série A
    # precisa vir antes, porque ``[b-z]`` também casaria o "s" de "series".
    if valor == "seed" or re.search(r"\bseries?\s*a\b", valor):
        return 5
    if re.search(r"\bseries?\s*[b-z]\b", valor):
        return 2
    return 1


def _pontos_setor(setor: str) -> int:
    valor = _texto_busca(setor)
    if any(termo in valor for termo in _SETORES_VERTICAL_DEDICADA):
        return 5
    if any(termo in valor for termo in _SETORES_STACK_GENERICA):
        return 3
    return 2


def _aplicar_travas(
    pontos: int,
    ids_evidencias: list[int],
    metadados_por_afirmacao: dict[int, MetadadoDocumentoFitScore],
) -> tuple[int, list[TravaFit]]:
    travas: list[TravaFit] = []
    evidencias_com_fonte = [
        id_afirmacao
        for id_afirmacao in ids_evidencias
        if id_afirmacao in metadados_por_afirmacao
    ]
    if pontos > 5 and not evidencias_com_fonte:
        pontos = 5
        travas.append("gate_evidencia")

    if pontos >= 8:
        hosts = {
            metadados_por_afirmacao[id_afirmacao].host_normalizado
            for id_afirmacao in evidencias_com_fonte
        }
        if len(hosts) < 2:
            pontos = 7
            travas.append("teto_corrobacao")
    return pontos, travas


def _pilar(
    nome: PilarFit,
    pontos: int,
    ids_evidencias: list[int],
    metadados_por_afirmacao: dict[int, MetadadoDocumentoFitScore],
) -> PilarFitScore:
    ids_ordenados = sorted(set(ids_evidencias))
    pontos_finais, travas = _aplicar_travas(
        pontos, ids_ordenados, metadados_por_afirmacao
    )
    return PilarFitScore(
        pilar=nome,
        pontos=pontos_finais,
        faixa=_faixa(pontos_finais),
        ids_evidencias=ids_ordenados,
        travas_aplicadas=travas,
    )


def _metadados_por_afirmacao(
    entrada: EntradaFitScore,
) -> dict[int, MetadadoDocumentoFitScore]:
    documentos = {item.id_documento: item for item in entrada.documentos}
    return {
        item.id_afirmacao: documentos[item.id_documento]
        for item in entrada.perfil_validado.afirmacoes_validadas
    }


def _confirmadas(entrada: EntradaFitScore) -> list[AfirmacaoValidada]:
    return [
        item
        for item in entrada.perfil_validado.afirmacoes_validadas
        if item.situacao == "confirmada"
    ]


def _pilar_centralidade(
    entrada: EntradaFitScore,
    confirmadas: list[AfirmacaoValidada],
    metadados: dict[int, MetadadoDocumentoFitScore],
) -> PilarFitScore:
    base = 5 if entrada.classe == "AI-native" else 3
    candidatas_bonus = sorted(
        item.id_afirmacao
        for item in confirmadas
        if item.categoria in {"stack_propria", "equipe_e_contratacao"}
        or (
            item.categoria == "otimizacao_tecnica"
            and item.polaridade == "presenca"
        )
    )
    ids_bonus = candidatas_bonus[:5]
    ids = sorted(set(entrada.ids_afirmacoes_suporte_classe + ids_bonus))
    return _pilar("centralidade_ia", base + len(ids_bonus), ids, metadados)


def _pilar_gap(
    entrada: EntradaFitScore,
    confirmadas: list[AfirmacaoValidada],
    metadados: dict[int, MetadadoDocumentoFitScore],
) -> PilarFitScore:
    dimensoes_gap = [
        item
        for item in entrada.perfil_validado.estado_dimensoes_gap
        if item.estado == "gap_confirmado"
    ]
    pontos = min(8, 2 * len(dimensoes_gap))
    ids = [
        id_afirmacao
        for dimensao in dimensoes_gap
        for id_afirmacao in dimensao.ids_evidencias
    ]
    ids_dor = sorted(
        item.id_afirmacao
        for item in confirmadas
        if item.categoria
        in {"dependencia_api_externa", "escala_e_dor_operacional"}
    )
    if ids_dor:
        pontos += 2
        ids.extend(ids_dor)
    return _pilar("gap_enderecavel", pontos, ids, metadados)


def _ids_recentes(
    confirmadas: list[AfirmacaoValidada],
    categoria: str,
    metadados: dict[int, MetadadoDocumentoFitScore],
    data_referencia: date,
    meses: int,
) -> list[int]:
    return sorted(
        item.id_afirmacao
        for item in confirmadas
        if item.categoria == categoria
        and _publicada_no_intervalo(
            metadados[item.id_afirmacao], data_referencia, meses
        )
    )


def _pilar_momento(
    entrada: EntradaFitScore,
    confirmadas: list[AfirmacaoValidada],
    metadados: dict[int, MetadadoDocumentoFitScore],
) -> PilarFitScore:
    pontos = _pontos_estagio(entrada.estagio)
    ids_momento = _ids_recentes(
        confirmadas,
        "momento_e_financiamento",
        metadados,
        entrada.data_referencia,
        18,
    )
    ids_equipe = _ids_recentes(
        confirmadas,
        "equipe_e_contratacao",
        metadados,
        entrada.data_referencia,
        12,
    )
    if ids_momento:
        pontos += 2
    if ids_equipe:
        pontos += 2
    return _pilar("momento", pontos, ids_momento + ids_equipe, metadados)


def _pilar_setor(
    entrada: EntradaFitScore,
    metadados: dict[int, MetadadoDocumentoFitScore],
) -> PilarFitScore:
    pontos = _pontos_setor(entrada.setor)
    if entrada.classe == "AI-native":
        pontos += 2
    return _pilar(
        "alinhamento_setorial",
        pontos,
        entrada.ids_afirmacoes_suporte_classe,
        metadados,
    )


def _pilares_gate_non_ai(entrada: EntradaFitScore) -> list[PilarFitScore]:
    ids = sorted(set(entrada.ids_afirmacoes_suporte_classe))
    return [
        PilarFitScore(
            pilar=nome,
            pontos=0,
            faixa="baixa",
            ids_evidencias=ids if nome in {"centralidade_ia", "alinhamento_setorial"} else [],
            travas_aplicadas=["gate_non_ai"],
        )
        for nome in (
            "centralidade_ia",
            "gap_enderecavel",
            "momento",
            "alinhamento_setorial",
        )
    ]


def _justificativa(entrada: EntradaFitScore) -> str:
    if entrada.classe == "non-AI":
        return (
            "Classificação non-AI validada; fit-score zerado pelo gate global da "
            "rubrica."
        )
    gap_dominante = next(
        (
            item.dimensao
            for item in entrada.perfil_validado.estado_dimensoes_gap
            if item.estado == "gap_confirmado"
        ),
        None,
    )
    contexto = (
        f"{entrada.classe} no setor {entrada.setor}, estágio {entrada.estagio}; "
    )
    if gap_dominante is None:
        return contexto + "gap não confirmado na base."
    return contexto + f"gap dominante confirmado: {_ROTULOS_GAP[gap_dominante]}."


def calcular_fit_score(entrada: EntradaFitScore) -> FitScore:
    """Calcula o fit-score v1 sem efeitos colaterais ou fontes implícitas."""

    if entrada.classe == "non-AI":
        return FitScore(
            total=0,
            pilares=_pilares_gate_non_ai(entrada),
            estado_dimensoes_gap=[
                item.model_copy(deep=True)
                for item in entrada.perfil_validado.estado_dimensoes_gap
            ],
            justificativa_curta=_justificativa(entrada),
            versao_rubrica=VERSAO_RUBRICA,
        )

    confirmadas = _confirmadas(entrada)
    metadados = _metadados_por_afirmacao(entrada)
    pilares = [
        _pilar_centralidade(entrada, confirmadas, metadados),
        _pilar_gap(entrada, confirmadas, metadados),
        _pilar_momento(entrada, confirmadas, metadados),
        _pilar_setor(entrada, metadados),
    ]
    soma_bruta = sum(item.pontos for item in pilares)
    total = round(100 * soma_bruta / MAXIMO_BRUTO_FIT_SCORE)
    return FitScore(
        total=total,
        pilares=pilares,
        estado_dimensoes_gap=[
            item.model_copy(deep=True)
            for item in entrada.perfil_validado.estado_dimensoes_gap
        ],
        justificativa_curta=_justificativa(entrada),
        versao_rubrica=VERSAO_RUBRICA,
    )
