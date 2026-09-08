"""Renderizador determinístico ``Briefing`` → Markdown (§11.4 da arquitetura).

Esta peça não decide nada: transcreve um ``Briefing`` já validado. Duas
consequências governam o arquivo inteiro — a saída é função pura da entrada
(mesma entrada, mesmos bytes) e cada variante mostra apenas o que o contrato
autoriza. O que o gerente vê na tela é o que ele leva no arquivo.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from radar.interface.rotulos import rotular_fundamento
from radar.interface.texto import destino_markdown, escapar_markdown
from radar.contratos import (
    Briefing,
    CabecalhoBriefing,
    CitacaoNvidia,
    ConclusaoAncorada,
    EvidenciaStartup,
    FonteBriefing,
    Recomendacao,
    RodapeBriefing,
)


FORMATO_DATA = "%d/%m/%Y"
LIMITE_APELIDO = 60
APELIDO_RESERVA = "startup"

RESUMO_DA_VARIANTE = {
    "normal": (
        "Startup aderente: há oportunidade NVIDIA sustentada por evidência pública."
    ),
    "nao_aderente": (
        "Empresa sem uso de IA no produto: a stack NVIDIA não se aplica a ela."
    ),
    # A §11.3 admite mais de um caminho até esta variante — zero afirmações
    # confirmadas, suporte da classe derrubado após o teto de reextração e
    # nenhuma recomendação com lastro. O cabeçalho é comum às três: quem nomeia
    # a causa é a síntese e os avisos já validados dentro do Briefing.
    "evidencia_insuficiente": (
        "Evidência insuficiente: a base disponível não sustenta uma conclusão "
        "sobre esta empresa. A causa específica está na síntese e nos avisos "
        "deste briefing; o índice de fontes fica vazio porque nenhuma conclusão "
        "desta variante cita afirmação."
    ),
}


def exportar_briefing_markdown(briefing: Briefing) -> str:
    """Devolve o briefing inteiro em Markdown UTF-8, pronto para download."""
    _exigir_briefing(briefing)
    blocos = [_titulo(briefing), _identificacao(briefing.cabecalho)]
    if briefing.variante != "evidencia_insuficiente":
        blocos.append(_veredito(briefing))
    blocos.append(_sintese(briefing.sintese_executiva))
    if briefing.pontos_de_conversa:
        blocos.append(_pontos_de_conversa(briefing.pontos_de_conversa))
    if briefing.avisos:
        blocos.append(_avisos(briefing.avisos))
    if briefing.recomendacoes:
        blocos.append(_recomendacoes(briefing.recomendacoes))
    if briefing.fontes:
        blocos.append(_fontes(briefing.fontes))
    blocos.append(_auditoria(briefing.rodape))
    return "\n\n".join(blocos) + "\n"


def nome_arquivo_briefing(briefing: Briefing) -> str:
    """``briefing_<apelido>_<data>.md`` — sem acento, sem caminho, sem surpresa."""
    _exigir_briefing(briefing)
    return (
        f"briefing_{_apelido(briefing.cabecalho.nome)}"
        f"_{briefing.cabecalho.data_geracao.isoformat()}.md"
    )


# ----------------------------------------------------------------------
# Blocos
# ----------------------------------------------------------------------


def _titulo(briefing: Briefing) -> str:
    return (
        f"# Briefing — {escapar_markdown(briefing.cabecalho.nome)}\n\n"
        f"{RESUMO_DA_VARIANTE[briefing.variante]}"
    )


def _identificacao(cabecalho: CabecalhoBriefing) -> str:
    linhas = [
        "## Identificação",
        "",
        f"- **Site oficial:** {destino_markdown(cabecalho.site)}",
        f"- **Setor:** {escapar_markdown(cabecalho.setor)}",
        f"- **Estágio:** {escapar_markdown(cabecalho.estagio)}",
        "- **Localização:** "
        + escapar_markdown(cabecalho.localizacao or "não informada"),
        f"- **Data de geração:** {_data(cabecalho.data_geracao)}",
        "- **Consulta original:** "
        + escapar_markdown(cabecalho.consulta_original),
    ]
    return "\n".join(linhas)


def _veredito(briefing: Briefing) -> str:
    veredito = briefing.veredito
    linhas = ["## Veredito", ""]
    if veredito.classe is not None:
        linhas.append(f"- **Classe validada:** {veredito.classe}")
    if veredito.fit_score_total is not None:
        linhas.append(f"- **Fit-score NVIDIA:** {veredito.fit_score_total}/100")
    linhas.append(f"- **Tese:** {escapar_markdown(veredito.tese)}")
    if veredito.ids_afirmacoes_suporte:
        linhas.append(
            "- **Afirmações que sustentam a tese:** "
            f"{_ids(veredito.ids_afirmacoes_suporte)}"
        )
    return "\n".join(linhas)


def _sintese(sintese: ConclusaoAncorada) -> str:
    linhas = ["## Síntese executiva", "", escapar_markdown(sintese.texto)]
    if sintese.ids_afirmacoes_suporte:
        linhas += [
            "",
            f"*Afirmações de suporte: {_ids(sintese.ids_afirmacoes_suporte)}*",
        ]
    return "\n".join(linhas)


def _pontos_de_conversa(pontos: list[ConclusaoAncorada]) -> str:
    linhas = ["## Pontos de conversa", ""]
    for ordem, ponto in enumerate(pontos, start=1):
        linhas.append(
            f"{ordem}. {escapar_markdown(ponto.texto)} "
            f"*(afirmações: {_ids(ponto.ids_afirmacoes_suporte)})*"
        )
    return "\n".join(linhas)


def _avisos(avisos: list[str]) -> str:
    return "\n".join(
        ["## Avisos", ""] + [f"- {escapar_markdown(aviso)}" for aviso in avisos]
    )


def _recomendacoes(recomendacoes: list[Recomendacao]) -> str:
    blocos = ["## Recomendações"]
    for ordem, recomendacao in enumerate(recomendacoes, start=1):
        blocos.append(_uma_recomendacao(ordem, recomendacao))
    return "\n\n".join(blocos)


def _uma_recomendacao(ordem: int, recomendacao: Recomendacao) -> str:
    linhas = [
        f"### {ordem}. {rotular_fundamento(recomendacao)}",
        "",
        f"- **Tecnologias NVIDIA:** {', '.join(recomendacao.tecnologias)}",
        f"- **Prioridade:** {recomendacao.prioridade}",
        f"- **Complexidade:** {recomendacao.complexidade}",
        "- **Justificativa técnica:** "
        + escapar_markdown(recomendacao.justificativa_tecnica),
        "- **Justificativa de negócio:** "
        + escapar_markdown(recomendacao.justificativa_negocio),
        (
            f"- **Próxima ação ({recomendacao.proxima_acao.tipo_acao}):** "
            + escapar_markdown(recomendacao.proxima_acao.detalhe)
        ),
        "",
        "**Evidências públicas da startup**",
        "",
    ]
    linhas += [_uma_evidencia(item) for item in recomendacao.evidencias_startup]
    linhas += ["", "**Citações da base NVIDIA**", ""]
    linhas += [_uma_citacao(item) for item in recomendacao.citacoes_nvidia]
    return "\n".join(linhas)


def _uma_evidencia(evidencia: EvidenciaStartup) -> str:
    return (
        f"- Afirmação {evidencia.id_afirmacao} "
        f"(documento {evidencia.id_documento}): "
        f'"{escapar_markdown(evidencia.trecho_citado)}" — '
        f"[fonte da afirmação {evidencia.id_afirmacao}]"
        f"({destino_markdown(evidencia.url_fonte)})"
    )


def _uma_citacao(citacao: CitacaoNvidia) -> str:
    partes = [f"- Chunk {citacao.id_chunk}"]
    if citacao.tecnologia is not None:
        partes.append(citacao.tecnologia)
    partes.append(f"origem: {citacao.origem}")
    partes.append(f"tópico: {escapar_markdown(citacao.topico)}")
    partes.append(f"trilha: {escapar_markdown(citacao.breadcrumb)}")
    partes.append(
        f"[chunk NVIDIA {citacao.id_chunk}]({destino_markdown(citacao.fonte_url)})"
    )
    return " — ".join(partes)


def _fontes(fontes: list[FonteBriefing]) -> str:
    linhas = ["## Fontes públicas", ""]
    for fonte in fontes:
        linha = (
            f"- [{escapar_markdown(fonte.titulo)}]"
            f"({destino_markdown(fonte.url_fonte)}) — "
            f"{escapar_markdown(fonte.host_normalizado)} — {fonte.tipo}"
        )
        if fonte.data_publicacao is not None:
            linha += f" — publicado em {_data(fonte.data_publicacao)}"
        linhas.append(linha)
    return "\n".join(linhas)


def _auditoria(rodape: RodapeBriefing) -> str:
    linhas = [
        "## Auditoria da execução",
        "",
        f"- **Versão da rubrica:** {rodape.versao_rubrica}",
        f"- **Data de execução:** {_data(rodape.data_execucao)}",
        f"- **Afirmações confirmadas:** {rodape.afirmacoes_confirmadas}",
        f"- **Afirmações derrubadas:** {rodape.afirmacoes_derrubadas}",
        f"- **Rota terminal (R3):** {rodape.rota_r3}",
        f"- **Trajeto do grafo:** {' → '.join(rodape.trajeto)}",
    ]
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# Apoio
# ----------------------------------------------------------------------


def _exigir_briefing(candidato: object) -> None:
    if not isinstance(candidato, Briefing):
        raise TypeError(
            "a exportação só aceita um Briefing já validado; recebeu "
            f"{type(candidato).__name__}"
        )


def _data(valor: date) -> str:
    return valor.strftime(FORMATO_DATA)


def _ids(valores: list[int]) -> str:
    return ", ".join(str(valor) for valor in valores)


def _apelido(nome: str) -> str:
    """Nome da startup reduzido a um identificador seguro de arquivo."""
    decomposto = unicodedata.normalize("NFKD", nome)
    sem_acento = "".join(
        letra for letra in decomposto if not unicodedata.combining(letra)
    )
    apelido = re.sub(r"[^a-z0-9]+", "-", sem_acento.casefold()).strip("-")
    return apelido[:LIMITE_APELIDO].strip("-") or APELIDO_RESERVA
