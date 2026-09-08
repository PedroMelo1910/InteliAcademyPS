"""Interface final do NVIDIA Startup AI Radar (Entregável 4).

A tela só orquestra e exibe: toda decisão veio pronta da fronteira da
aplicação (`executar_descoberta` e `executar_aprofundamento`). Aqui não se
ordena ranking, não se recalcula fit-score e não se remonta payload de grafo.
O que sobrevive entre reruns está em `radar.interface.estado`; o que o usuário
lê está em `radar.interface.rotulos` e `radar.interface.mensagens`; o que ele
baixa está em `radar.interface.exportacao` — o mesmo conteúdo da tela.

A direção visual tem uma regra só, e ela é semântica: **verde é aderência
NVIDIA validada, azul é relevância textual da busca**. As duas medidas nunca
compartilham cor, rótulo ou vizinhança, porque confundi-las é o erro que este
projeto não pode cometer na frente de um avaliador. O único HTML bruto da
aplicação é `CSS_TEMA`, um literal estático: nenhum texto de usuário, de fonte
pública ou de modelo chega até essa fronteira.
"""

import logging

import streamlit as st

from radar.agentes.query_planner import ErroQueryPlanner
from radar.aplicacao import criar_aplicacao
from radar.configuracao import ErroConfiguracao
from radar.interface import estado as sessao
from radar.interface import mensagens
from radar.interface.exportacao import (
    RESUMO_DA_VARIANTE,
    exportar_briefing_markdown,
    nome_arquivo_briefing,
)
from radar.interface.rotulos import (
    TOM_DA_CLASSE,
    contar_estados,
    fracao_do_fit_score,
    resumir_ranking,
    rotular_fundamento,
    tom_do_status,
)
from radar.interface.tema import CSS_TEMA
from radar.interface.texto import destino_markdown, escapar_markdown


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title=mensagens.NOME_PRODUTO,
    page_icon="📡",
    layout="wide",
)
st.html(CSS_TEMA)


@st.cache_resource
def _aplicacao_padrao():
    return criar_aplicacao()


def obter_aplicacao():
    """A aplicação real, ou a que já estiver na sessão (injeção offline)."""
    if sessao.CHAVE_APLICACAO in st.session_state:
        return st.session_state[sessao.CHAVE_APLICACAO]
    return _aplicacao_padrao()


def ids_de_suporte(valores) -> str:
    return ", ".join(str(valor) for valor in valores)


def linha_meta(*partes: str) -> str:
    """Junta setor, estágio e localização já escapados numa linha só."""
    return " · ".join(escapar_markdown(parte) for parte in partes)


def selecionar(id_startup: int) -> None:
    sessao.selecionar_startup(st.session_state, id_startup)


def voltar() -> None:
    sessao.limpar_selecao(st.session_state)


def preencher_consulta(pergunta: str) -> None:
    """Um exemplo escreve no campo e para por aí: quem busca é o usuário."""
    st.session_state["consulta"] = pergunta


# ----------------------------------------------------------------------
# Abertura e busca
# ----------------------------------------------------------------------


def renderizar_topo() -> None:
    with st.container(key="topo_radar"):
        with st.container(key="etiqueta_produto"):
            st.markdown(mensagens.ETIQUETA_PRODUTO)
        st.title(mensagens.NOME_PRODUTO)
        with st.container(key="proposito_produto"):
            st.markdown(mensagens.PROPOSITO)


def renderizar_busca() -> tuple[str, bool]:
    with st.container(key="bloco_busca"):
        with st.form("consulta_startups"):
            consulta = st.text_input(
                mensagens.ROTULO_CAMPO_CONSULTA,
                key="consulta",
                placeholder=mensagens.PLACEHOLDER_CONSULTA,
            )
            buscar = st.form_submit_button(mensagens.ROTULO_BUSCAR, type="primary")
        st.caption(mensagens.CONVITE_INICIAL)
    return consulta, buscar


def renderizar_exemplos() -> None:
    """Perguntas prontas que preenchem o campo — nenhuma delas executa a busca."""
    st.caption(mensagens.TITULO_EXEMPLOS)
    with st.container(horizontal=True, key="exemplos_consulta"):
        for indice, pergunta in enumerate(mensagens.PERGUNTAS_DE_EXEMPLO):
            st.button(
                pergunta,
                key=f"exemplo_{indice}",
                type="tertiary",
                on_click=preencher_consulta,
                args=(pergunta,),
            )
    st.caption(mensagens.LEGENDA_EXEMPLOS)


def executar_busca(consulta: str) -> None:
    """Uma submissão explícita apaga o estado antigo antes de tentar o novo."""
    sessao.iniciar_busca(st.session_state)
    try:
        with st.spinner(mensagens.MENSAGEM_CARREGANDO_BUSCA):
            saida = obter_aplicacao().executar_descoberta(consulta)
    except ErroConfiguracao as erro:
        st.error(str(erro))
    except ErroQueryPlanner as erro:
        logger.exception("O Query Planner interrompeu a consulta com segurança")
        st.error(str(erro))
    except Exception:
        logger.exception("Falha inesperada ao executar a descoberta de startups")
        st.error(mensagens.MENSAGEM_FALHA_DESCOBERTA)
    else:
        sessao.registrar_descoberta(st.session_state, saida)


# ----------------------------------------------------------------------
# Tela do ranking
# ----------------------------------------------------------------------


def renderizar_ranking(descoberta) -> None:
    if descoberta.criterios_relaxados:
        st.info(
            "A busca não encontrou candidatas com os critérios originais e "
            "relaxou: " + ", ".join(descoberta.criterios_relaxados) + "."
        )
    ranking = sessao.ranking_visivel(st.session_state)
    if not ranking:
        st.warning(mensagens.SEM_RESULTADO)
        st.caption(mensagens.SEM_RESULTADO_SAIDA)
        return

    renderizar_painel(ranking)
    st.subheader(mensagens.TITULO_RANKING)
    st.caption(mensagens.LEGENDA_RANKING)
    for resumo, item in zip(resumir_ranking(ranking), ranking, strict=True):
        renderizar_candidata(resumo, item)
    with st.expander(mensagens.TITULO_INTERPRETACAO):
        st.json(descoberta.plano.model_dump(mode="json"))
        st.caption("Fluxo executado: " + " → ".join(descoberta.trajeto))


def renderizar_painel(ranking) -> None:
    """As contagens por estado, e a explicação das duas medidas ao lado."""
    contagem = contar_estados(ranking)
    with st.container(key="painel_ranking"):
        colunas = st.columns(4)
        colunas[0].metric(mensagens.ROTULO_TOTAL_CANDIDATAS, contagem.total)
        colunas[1].metric(mensagens.ROTULO_TOTAL_ANALISADAS, contagem.concluidas)
        colunas[2].metric(mensagens.ROTULO_TOTAL_SEM_LASTRO, contagem.sem_lastro)
        colunas[3].metric(mensagens.ROTULO_TOTAL_PENDENTES, contagem.ausentes)
    with st.expander(mensagens.TITULO_COMO_LER):
        st.markdown(
            f"**{mensagens.ROTULO_FIT_SCORE}** — {mensagens.EXPLICACAO_FIT_SCORE}"
        )
        st.markdown(f"**{mensagens.ROTULO_BM25}** — {mensagens.EXPLICACAO_BM25}")


def renderizar_candidata(resumo, item) -> None:
    nome = escapar_markdown(resumo.nome)
    with st.container(border=True, key=f"cartao_{resumo.id_startup}"):
        colunas = st.columns([1, 8, 3], vertical_alignment="center")

        with colunas[0], st.container(key=f"posicao_{resumo.id_startup}"):
            st.markdown(f"{resumo.posicao:02d}")

        with colunas[1]:
            with st.container(key=f"nome_{resumo.id_startup}"):
                st.markdown(nome)
            with st.container(horizontal=True, key=f"selos_{resumo.id_startup}"):
                st.badge(resumo.rotulo_status, color=tom_do_status(resumo))
                if resumo.classe is not None:
                    st.badge(resumo.classe, color=TOM_DA_CLASSE)
            with st.container(key=f"meta_{resumo.id_startup}"):
                st.markdown(
                    linha_meta(resumo.setor, resumo.estagio, resumo.localizacao)
                )

        with colunas[2]:
            if resumo.fit_score is not None:
                st.metric(mensagens.ROTULO_FIT_SCORE, resumo.fit_score)
            else:
                st.caption(mensagens.SEM_PONTUACAO_GRAVADA)

        st.markdown(escapar_markdown(resumo.descricao))
        st.caption(resumo.explicacao_status)
        if resumo.justificativa_fit_score:
            st.markdown(
                f"**{mensagens.TITULO_JUSTIFICATIVA}:** "
                + escapar_markdown(resumo.justificativa_fit_score)
            )
        if resumo.motivo_evidencia_insuficiente:
            st.markdown(
                f"**{mensagens.TITULO_MOTIVO}:** "
                + escapar_markdown(resumo.motivo_evidencia_insuficiente)
            )

        with st.container(key=f"lexical_{resumo.id_startup}"):
            st.markdown(f"{mensagens.ROTULO_BM25}: {resumo.bm25}")

        with st.expander(f"{mensagens.TITULO_DOCUMENTOS} — {nome}"):
            st.caption(mensagens.EXPLICACAO_BM25)
            for documento in item.documentos:
                st.markdown(
                    f"- [{escapar_markdown(documento.titulo)}]"
                    f"({destino_markdown(documento.url_fonte)}) — "
                    f"{escapar_markdown(documento.dominio_fonte)}; acesso em "
                    f"{documento.data_acesso.strftime('%d/%m/%Y')}; "
                    f"BM25 {documento.score_bm25:.6f}"
                )

        st.button(
            mensagens.ROTULO_ANALISAR.format(nome),
            key=f"aprofundar_{resumo.id_startup}",
            type="primary",
            on_click=selecionar,
            args=(resumo.id_startup,),
        )


# ----------------------------------------------------------------------
# Tela da análise completa
# ----------------------------------------------------------------------


def candidata_selecionada(descoberta, id_startup):
    for item in descoberta.ranking:
        if item.empresa.id_startup == id_startup:
            return item
    return None


def renderizar_analise(descoberta) -> None:
    id_startup = sessao.startup_selecionada(st.session_state)
    item = candidata_selecionada(descoberta, id_startup)

    with st.container(key="barra_voltar"):
        st.button(
            mensagens.ROTULO_VOLTAR,
            key="voltar_para_candidatas",
            on_click=voltar,
        )
        st.caption(mensagens.LEGENDA_VOLTAR)

    if item is None:
        st.warning(mensagens.CANDIDATA_FORA_DA_BUSCA)
        return

    pendente = sessao.aprofundamento_pendente(st.session_state)
    if pendente is not None:
        with st.spinner(
            f"{escapar_markdown(item.empresa.nome)}: "
            + mensagens.MENSAGEM_CARREGANDO_ANALISE
        ):
            executar_aprofundamento(descoberta, pendente)

    falha = sessao.falha_selecionada(st.session_state)
    if falha is not None:
        st.subheader(mensagens.TITULO_SEM_ANALISE_PROFUNDA)
        st.error(falha)
        with st.expander(mensagens.TITULO_DIAGNOSTICO):
            st.caption(mensagens.DIAGNOSTICO_TECNICO)
        return

    saida = sessao.aprofundamento_selecionado(st.session_state)
    if saida is not None:
        renderizar_briefing(saida)


def executar_aprofundamento(descoberta, id_startup: int) -> None:
    try:
        saida = obter_aplicacao().executar_aprofundamento(descoberta, id_startup)
    except ErroConfiguracao as erro:
        sessao.registrar_falha_aprofundamento(st.session_state, id_startup, str(erro))
    except Exception:
        logger.exception(
            "Falha ao aprofundar a startup %s; nenhum briefing parcial foi exposto",
            id_startup,
        )
        sessao.registrar_falha_aprofundamento(
            st.session_state, id_startup, mensagens.MENSAGEM_FALHA_APROFUNDAMENTO
        )
    else:
        sessao.registrar_aprofundamento(st.session_state, saida)


# ----------------------------------------------------------------------
# Briefing validado
# ----------------------------------------------------------------------


def renderizar_briefing(saida) -> None:
    briefing = saida.briefing
    renderizar_cabecalho(briefing)

    abas = [mensagens.ABA_VISAO_GERAL]
    if briefing.fontes or briefing.recomendacoes:
        abas.append(mensagens.ABA_EVIDENCIAS)
    if briefing.recomendacoes:
        abas.append(mensagens.ABA_RECOMENDACOES)
    abas.append(mensagens.ABA_RASTRO)

    paineis = dict(zip(abas, st.tabs(abas), strict=True))

    with paineis[mensagens.ABA_VISAO_GERAL]:
        renderizar_visao_geral(briefing)
    if mensagens.ABA_EVIDENCIAS in paineis:
        with paineis[mensagens.ABA_EVIDENCIAS]:
            renderizar_evidencias(briefing)
    if mensagens.ABA_RECOMENDACOES in paineis:
        with paineis[mensagens.ABA_RECOMENDACOES]:
            renderizar_recomendacoes(briefing.recomendacoes)
    with paineis[mensagens.ABA_RASTRO]:
        renderizar_auditoria(briefing, saida)


def renderizar_cabecalho(briefing) -> None:
    """Identificação, veredito e download — a resposta executiva antes das abas."""
    cabecalho = briefing.cabecalho
    veredito = briefing.veredito
    with st.container(key="cabecalho_analise"):
        with st.container(key="etiqueta_briefing"):
            st.markdown(mensagens.ETIQUETA_BRIEFING)
        st.header(escapar_markdown(cabecalho.nome))

        with st.container(horizontal=True, key="selos_analise"):
            if veredito.classe is not None:
                st.badge(veredito.classe, color=TOM_DA_CLASSE)

        with st.container(key="meta_analise"):
            st.markdown(
                linha_meta(
                    cabecalho.setor,
                    cabecalho.estagio,
                    cabecalho.localizacao or "não informada",
                )
            )
        st.markdown(
            f"**{mensagens.ROTULO_SITE}:** "
            f"[{escapar_markdown(str(cabecalho.site))}]"
            f"({destino_markdown(cabecalho.site)})"
        )

        if veredito.fit_score_total is not None:
            colunas = st.columns([1, 3], vertical_alignment="center")
            colunas[0].metric(
                mensagens.ROTULO_FIT_SCORE, f"{veredito.fit_score_total}/100"
            )
            colunas[1].progress(fracao_do_fit_score(veredito.fit_score_total))
            colunas[1].caption(mensagens.EXPLICACAO_FIT_SCORE)

        if briefing.variante == "evidencia_insuficiente":
            st.warning(RESUMO_DA_VARIANTE[briefing.variante])
        else:
            st.caption(RESUMO_DA_VARIANTE[briefing.variante])
            st.markdown(f"**{mensagens.TITULO_TESE}**")
            with st.container(key="tese_principal"):
                st.markdown(escapar_markdown(veredito.tese))
            if veredito.ids_afirmacoes_suporte:
                st.caption(
                    "Afirmações que sustentam a tese: "
                    + ids_de_suporte(veredito.ids_afirmacoes_suporte)
                )

        st.download_button(
            mensagens.ROTULO_BAIXAR,
            data=exportar_briefing_markdown(briefing).encode("utf-8"),
            file_name=nome_arquivo_briefing(briefing),
            mime="text/markdown",
            key="baixar_briefing",
            type="primary",
        )
        st.caption(mensagens.LEGENDA_BAIXAR)


def renderizar_visao_geral(briefing) -> None:
    st.markdown(f"### {mensagens.TITULO_SINTESE}")
    st.markdown(escapar_markdown(briefing.sintese_executiva.texto))
    if briefing.sintese_executiva.ids_afirmacoes_suporte:
        st.caption(
            "Afirmações de suporte: "
            + ids_de_suporte(briefing.sintese_executiva.ids_afirmacoes_suporte)
        )
    if briefing.pontos_de_conversa:
        st.markdown(f"### {mensagens.TITULO_PONTOS}")
        for ponto in briefing.pontos_de_conversa:
            st.markdown(
                f"- {escapar_markdown(ponto.texto)} "
                f"*(afirmações: {ids_de_suporte(ponto.ids_afirmacoes_suporte)})*"
            )
    if briefing.avisos:
        st.markdown(f"### {mensagens.TITULO_AVISOS}")
        for aviso in briefing.avisos:
            st.warning(escapar_markdown(aviso))


def renderizar_evidencias(briefing) -> None:
    """Evidência da startup e evidência NVIDIA, separadas e nunca confundidas."""
    if briefing.recomendacoes:
        st.markdown(f"### {mensagens.TITULO_EVIDENCIA_STARTUP}")
        st.caption(mensagens.LEGENDA_EVIDENCIA_STARTUP)
        for evidencia in evidencias_da_startup(briefing):
            st.markdown(
                f"- Afirmação {evidencia.id_afirmacao} "
                f"(documento {evidencia.id_documento}): "
                f'"{escapar_markdown(evidencia.trecho_citado)}" — '
                f"[fonte da afirmação {evidencia.id_afirmacao}]"
                f"({destino_markdown(evidencia.url_fonte)})"
            )

        st.markdown(f"### {mensagens.TITULO_EVIDENCIA_NVIDIA}")
        st.caption(mensagens.LEGENDA_EVIDENCIA_NVIDIA)
        for citacao in citacoes_nvidia(briefing):
            st.markdown(linha_de_citacao(citacao))

    if briefing.fontes:
        st.markdown(f"### {mensagens.TITULO_FONTES}")
        for fonte in briefing.fontes:
            linha = (
                f"- [{escapar_markdown(fonte.titulo)}]"
                f"({destino_markdown(fonte.url_fonte)}) — "
                f"{escapar_markdown(fonte.host_normalizado)} — {fonte.tipo}"
            )
            if fonte.data_publicacao is not None:
                linha += (
                    f" — publicado em {fonte.data_publicacao.strftime('%d/%m/%Y')}"
                )
            st.markdown(linha)


def evidencias_da_startup(briefing):
    """Cada afirmação uma vez só, na ordem em que as recomendações a citam."""
    vistas = set()
    for recomendacao in briefing.recomendacoes:
        for evidencia in recomendacao.evidencias_startup:
            if evidencia.id_afirmacao in vistas:
                continue
            vistas.add(evidencia.id_afirmacao)
            yield evidencia


def citacoes_nvidia(briefing):
    vistas = set()
    for recomendacao in briefing.recomendacoes:
        for citacao in recomendacao.citacoes_nvidia:
            if citacao.id_chunk in vistas:
                continue
            vistas.add(citacao.id_chunk)
            yield citacao


def linha_de_citacao(citacao) -> str:
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


def renderizar_recomendacoes(recomendacoes) -> None:
    st.caption(mensagens.LEGENDA_RECOMENDACAO)
    for ordem, recomendacao in enumerate(recomendacoes, start=1):
        with st.container(border=True, key=f"bloco_recomendacao_{ordem}"):
            st.markdown(f"#### {ordem}. {rotular_fundamento(recomendacao)}")
            st.markdown(
                f"**{mensagens.ROTULO_TECNOLOGIAS}:** "
                + ", ".join(recomendacao.tecnologias)
            )
            colunas = st.columns(2)
            colunas[0].metric(
                mensagens.ROTULO_PRIORIDADE, recomendacao.prioridade
            )
            colunas[1].metric(
                mensagens.ROTULO_COMPLEXIDADE, recomendacao.complexidade
            )
            st.markdown(
                f"**{mensagens.ROTULO_JUSTIFICATIVA_NEGOCIO}:** "
                + escapar_markdown(recomendacao.justificativa_negocio)
            )
            st.markdown(
                f"**{mensagens.ROTULO_JUSTIFICATIVA_TECNICA}:** "
                + escapar_markdown(recomendacao.justificativa_tecnica)
            )
            st.markdown(
                f"**Próxima ação ({recomendacao.proxima_acao.tipo_acao}):** "
                + escapar_markdown(recomendacao.proxima_acao.detalhe)
            )
            st.caption(
                f"{mensagens.TITULO_LASTRO}: afirmações "
                + ids_de_suporte(
                    item.id_afirmacao for item in recomendacao.evidencias_startup
                )
                + " · chunks NVIDIA "
                + ids_de_suporte(
                    item.id_chunk for item in recomendacao.citacoes_nvidia
                )
            )


def renderizar_auditoria(briefing, saida) -> None:
    rodape = briefing.rodape
    st.markdown(f"### {mensagens.TITULO_AUDITORIA}")
    st.markdown(f"- **Versão da rubrica:** {rodape.versao_rubrica}")
    st.markdown(
        f"- **Data de execução:** {rodape.data_execucao.strftime('%d/%m/%Y')}"
    )
    st.markdown(
        f"- **Afirmações confirmadas:** {rodape.afirmacoes_confirmadas} · "
        f"**derrubadas:** {rodape.afirmacoes_derrubadas}"
    )
    st.markdown(f"- **Rota terminal (R3):** {rodape.rota_r3}")
    st.markdown("- **Trajeto do briefing:** " + " → ".join(rodape.trajeto))
    st.markdown("- **Trajeto desta execução:** " + " → ".join(saida.trajeto))
    for erro in saida.erros:
        st.markdown(
            "- **Ocorrência registrada pelo grafo:** " + escapar_markdown(erro)
        )
    st.caption(
        f"Gerado em {briefing.cabecalho.data_geracao.strftime('%d/%m/%Y')} para a "
        "consulta: " + escapar_markdown(briefing.cabecalho.consulta_original)
    )


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------

renderizar_topo()
consulta_digitada, submeteu = renderizar_busca()

if submeteu:
    if consulta_digitada.strip():
        executar_busca(consulta_digitada)
    else:
        st.warning(mensagens.CONSULTA_EM_BRANCO)

descoberta_corrente = sessao.descoberta_atual(st.session_state)
if descoberta_corrente is None:
    renderizar_exemplos()
elif sessao.startup_selecionada(st.session_state) is None:
    renderizar_ranking(descoberta_corrente)
else:
    renderizar_analise(descoberta_corrente)
