"""Interface final do NVIDIA Startup AI Radar (Entregável 4).

A tela só orquestra e exibe: toda decisão veio pronta da fronteira da
aplicação (`executar_descoberta` e `executar_aprofundamento`). Os controles
pedem à aplicação apenas para reordenar ou filtrar medidas já calculadas;
aqui não se recalcula fit-score e não se remonta payload de grafo.
O que sobrevive entre reruns está em `radar.interface.estado`; o que o usuário
lê está em `radar.interface.rotulos` e `radar.interface.mensagens`; o que ele
baixa está em `radar.interface.exportacao` — o mesmo conteúdo da tela.

A direção visual tem uma regra só, e ela é semântica: **verde é aderência
NVIDIA validada**. A relevância textual continua como critério interno da
busca, mas não é apresentada como outra pontuação ao usuário. O único HTML
bruto da aplicação é `CSS_TEMA`, um literal estático: nenhum texto de usuário,
de fonte pública ou de modelo chega até essa fronteira.
"""

import logging
from collections import Counter

import streamlit as st

from radar.agentes.query_planner import ErroQueryPlanner
from radar.aplicacao import criar_aplicacao, personalizar_ranking
from radar.base_startups import BaseStartups
from radar.configuracao import CAMINHO_BANCO, ErroConfiguracao
from radar.interface import estado as sessao
from radar.interface import mensagens
from radar.interface.exportacao import (
    RESUMO_DA_VARIANTE,
    exportar_briefing_markdown,
    nome_arquivo_briefing,
)
from radar.interface.rotulos import (
    TOM_DA_CLASSE,
    fracao_do_fit_score,
    maximo_do_pilar,
    resumir_ranking,
    rotulo,
    rotular_fundamento,
    texto_legivel,
    tom_do_status,
)
from radar.interface.tema import CSS_TEMA
from radar.interface.texto import destino_markdown, escapar_markdown


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title=mensagens.NOME_PRODUTO,
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.html(CSS_TEMA)


@st.cache_resource(show_spinner=False)
def _aplicacao_padrao():
    return criar_aplicacao()


def obter_aplicacao():
    """A aplicação real, ou a que já estiver na sessão (injeção offline)."""
    if sessao.CHAVE_APLICACAO in st.session_state:
        return st.session_state[sessao.CHAVE_APLICACAO]
    return _aplicacao_padrao()


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


def renderizar_navegacao() -> str:
    """Mantém as duas superfícies no mesmo app e no mesmo processo."""
    st.sidebar.markdown(f"### {mensagens.NOME_PRODUTO}")
    return st.sidebar.radio(
        mensagens.ROTULO_NAVEGACAO,
        (mensagens.PAGINA_RADAR, mensagens.PAGINA_DASHBOARD),
        label_visibility="collapsed",
        key="pagina_principal",
    )


# ----------------------------------------------------------------------
# Abertura e busca
# ----------------------------------------------------------------------


def renderizar_topo() -> None:
    with st.container(key="topo_radar"):
        st.title(mensagens.NOME_PRODUTO)


def renderizar_busca() -> tuple[str, bool]:
    with st.container(key="bloco_busca"):
        with st.form("consulta_startups"):
            consulta = st.text_input(
                mensagens.ROTULO_CAMPO_CONSULTA,
                key="consulta",
                placeholder=mensagens.PLACEHOLDER_CONSULTA,
            )
            buscar = st.form_submit_button(mensagens.ROTULO_BUSCAR, type="primary")
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
# Dashboard da base persistida
# ----------------------------------------------------------------------


def dados_do_dashboard():
    """Lê somente projeções permitidas; nunca consulta o gabarito da curadoria."""
    base = BaseStartups(CAMINHO_BANCO)
    empresas = base.listar_startups_para_lote()
    analises = base.carregar_analises(
        [empresa.id_startup for empresa in empresas]
    )
    return empresas, analises


def renderizar_distribuicao(titulo: str, valores, total: int) -> None:
    st.markdown(f"### {titulo}")
    for nome, quantidade in valores:
        colunas = st.columns([3, 6, 1], vertical_alignment="center")
        colunas[0].markdown(f"**{escapar_markdown(nome)}**")
        colunas[1].progress(quantidade / total if total else 0)
        colunas[2].markdown(str(quantidade))


def renderizar_dashboard() -> None:
    st.title(mensagens.TITULO_DASHBOARD)
    st.caption(mensagens.LEGENDA_DASHBOARD)
    try:
        empresas, analises = dados_do_dashboard()
    except Exception:
        logger.exception("Falha ao carregar o panorama local")
        st.error("Não foi possível carregar o panorama da base local.")
        return

    concluidas = [item for item in analises.values() if item.status == "concluida"]
    insuficientes = len(analises) - len(concluidas)
    ausentes = len(empresas) - len(analises)
    pontuacoes = [item.fit_score.total for item in concluidas if item.fit_score]

    with st.container(key="metricas_dashboard"):
        colunas = st.columns(4)
        colunas[0].metric("Startups", len(empresas))
        colunas[1].metric("Análises concluídas", len(concluidas))
        colunas[2].metric(
            "Fit-score médio",
            f"{sum(pontuacoes) / len(pontuacoes):.1f}/100" if pontuacoes else "—",
        )
        colunas[3].metric("Maior fit-score", f"{max(pontuacoes)}/100" if pontuacoes else "—")

    esquerda, direita = st.columns(2)
    classes = Counter(item.classe for item in concluidas if item.classe is not None)
    faixas = (
        ("Zero validado", sum(valor == 0 for valor in pontuacoes)),
        ("De 1 a 25", sum(1 <= valor <= 25 for valor in pontuacoes)),
        ("De 26 a 50", sum(26 <= valor <= 50 for valor in pontuacoes)),
        ("Acima de 50", sum(valor > 50 for valor in pontuacoes)),
    )
    with esquerda:
        renderizar_distribuicao(
            "Classificação do uso de IA",
            ((classe, classes.get(classe, 0)) for classe in (
                "AI-native", "AI-enabled", "non-AI"
            )),
            len(concluidas),
        )
    with direita:
        renderizar_distribuicao(
            "Distribuição do fit-score", faixas, len(pontuacoes)
        )

    setores = Counter(empresa.setor for empresa in empresas)
    renderizar_distribuicao(
        "Setores mais representados",
        setores.most_common(6),
        len(empresas),
    )
    if insuficientes or ausentes:
        st.info(
            f"{insuficientes} análise(s) sem evidência suficiente e "
            f"{ausentes} ainda não analisada(s)."
        )
    st.caption(
        "O fit-score mede aderência comprovada à stack NVIDIA. Ele não avalia "
        "a qualidade das empresas e não é aumentado para melhorar o gráfico."
    )


# ----------------------------------------------------------------------
# Tela do ranking
# ----------------------------------------------------------------------


def renderizar_ranking(descoberta) -> None:
    if descoberta.criterios_relaxados:
        st.info(
            "A busca não encontrou candidatas com os critérios originais e "
            "ampliou: "
            + ", ".join(texto_legivel(item) for item in descoberta.criterios_relaxados)
            + "."
        )
    ranking_completo = sessao.ranking_visivel(st.session_state)
    if not ranking_completo:
        st.warning(mensagens.SEM_RESULTADO)
        st.caption(mensagens.SEM_RESULTADO_SAIDA)
        return

    st.subheader(mensagens.TITULO_RANKING)
    st.caption(mensagens.LEGENDA_RANKING)

    coluna_ordem, coluna_classe = st.columns(2)
    with coluna_ordem:
        rotulo_ordem = st.selectbox(
            mensagens.ROTULO_ORDENACAO,
            (
                mensagens.OPCAO_ORDENAR_FIT_SCORE,
                mensagens.OPCAO_ORDENAR_RELEVANCIA,
            ),
            key="ordenacao_ranking",
        )
    with coluna_classe:
        rotulo_classe = st.selectbox(
            mensagens.ROTULO_FILTRO_CLASSE,
            (
                mensagens.OPCAO_TODAS_CLASSES,
                "AI-native",
                "AI-enabled",
                "non-AI",
            ),
            key="filtro_classe_ranking",
        )

    criterio = (
        "fit_score"
        if rotulo_ordem == mensagens.OPCAO_ORDENAR_FIT_SCORE
        else "relevancia"
    )
    classe = None if rotulo_classe == mensagens.OPCAO_TODAS_CLASSES else rotulo_classe
    ranking = personalizar_ranking(
        ranking_completo,
        criterio=criterio,
        classe=classe,
        consulta=descoberta.consulta,
    )
    if not ranking:
        st.info(mensagens.SEM_RESULTADO_NO_FILTRO)
        return

    for resumo, item in zip(resumir_ranking(ranking), ranking, strict=True):
        renderizar_candidata(resumo, item)


def renderizar_candidata(resumo, item) -> None:
    nome = escapar_markdown(resumo.nome)
    with st.container(border=True, key=f"cartao_{resumo.id_startup}"):
        colunas = st.columns([1, 8, 3], vertical_alignment="center")

        with colunas[0], st.container(key=f"posicao_{resumo.id_startup}"):
            st.markdown(str(resumo.posicao))

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
            if resumo.fit_score is not None and item.fit_score_total is not None:
                st.metric(mensagens.ROTULO_FIT_SCORE, resumo.fit_score)
                st.progress(fracao_do_fit_score(item.fit_score_total))
            else:
                st.caption(mensagens.SEM_PONTUACAO_GRAVADA)

        st.markdown(escapar_markdown(resumo.descricao))
        if resumo.motivo_evidencia_insuficiente:
            st.markdown(
                f"**{mensagens.TITULO_MOTIVO}:** "
                + escapar_markdown(resumo.motivo_evidencia_insuficiente)
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
        with st.spinner(mensagens.MENSAGEM_CARREGANDO_ANALISE):
            executar_aprofundamento(descoberta, pendente)

    falha = sessao.falha_selecionada(st.session_state)
    if falha is not None:
        st.subheader(mensagens.TITULO_SEM_ANALISE_PROFUNDA)
        st.error(falha)
        with st.expander(mensagens.TITULO_DIAGNOSTICO):
            st.caption(mensagens.ORIENTACAO_FALHA)
            st.caption(mensagens.DIAGNOSTICO_TECNICO)
        st.button(
            mensagens.ROTULO_TENTAR_NOVAMENTE,
            key=f"tentar_novamente_{id_startup}",
            type="primary",
            on_click=selecionar,
            args=(id_startup,),
        )
        return

    saida = sessao.aprofundamento_selecionado(st.session_state)
    if saida is not None:
        renderizar_briefing(saida, item, descoberta)


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


def renderizar_briefing(saida, item=None, descoberta=None) -> None:
    briefing = saida.briefing
    renderizar_cabecalho(briefing)
    renderizar_origem_da_analise(item, briefing)

    abas = [mensagens.ABA_VISAO_GERAL]
    if briefing.fontes or briefing.recomendacoes:
        abas.append(mensagens.ABA_EVIDENCIAS)
    if briefing.recomendacoes:
        abas.append(mensagens.ABA_RECOMENDACOES)

    paineis = dict(zip(abas, st.tabs(abas), strict=True))

    with paineis[mensagens.ABA_VISAO_GERAL]:
        renderizar_visao_geral(briefing, saida, item, descoberta)
    if mensagens.ABA_EVIDENCIAS in paineis:
        with paineis[mensagens.ABA_EVIDENCIAS]:
            renderizar_evidencias(briefing)
    if mensagens.ABA_RECOMENDACOES in paineis:
        with paineis[mensagens.ABA_RECOMENDACOES]:
            renderizar_recomendacoes(briefing.recomendacoes)


def renderizar_origem_da_analise(item, briefing) -> None:
    if item is None or item.status_analise != "concluida":
        return
    st.info(mensagens.AVISO_CACHE_ATUAL)
    cache_diverge = (
        item.classe != briefing.veredito.classe
        or item.fit_score_total != briefing.veredito.fit_score_total
    )
    if cache_diverge:
        st.warning(mensagens.AVISO_CACHE_DIVERGENTE)


def renderizar_cabecalho(briefing) -> None:
    """Identificação e download — o resultado vem depois do lastro na visão geral."""
    cabecalho = briefing.cabecalho
    with st.container(key="cabecalho_analise"):
        with st.container(key="etiqueta_briefing"):
            st.markdown(mensagens.ETIQUETA_BRIEFING)
        st.header(escapar_markdown(cabecalho.nome))

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

        st.download_button(
            mensagens.ROTULO_BAIXAR,
            data=exportar_briefing_markdown(briefing).encode("utf-8"),
            file_name=nome_arquivo_briefing(briefing),
            mime="text/markdown",
            key="baixar_briefing",
            type="primary",
        )
        st.caption(mensagens.LEGENDA_BAIXAR)


def renderizar_visao_geral(briefing, saida=None, item=None, descoberta=None) -> None:
    if descoberta is not None and item is not None:
        st.markdown(f"### {mensagens.TITULO_POR_QUE_APARECEU}")
        st.markdown(
            "Esta startup apareceu entre as candidatas para **"
            + escapar_markdown(descoberta.consulta)
            + "**. A busca encontrou correspondência em "
            + f"{len(item.documentos)} documento(s) público(s)."
        )

    perfil = getattr(saida, "perfil_validado", None) if saida is not None else None
    renderizar_evidencias_confirmadas(perfil)
    renderizar_incertezas(perfil)
    renderizar_resultado_analise(briefing)
    if saida is not None:
        renderizar_fit_score(saida)
    renderizar_necessidades(perfil)

    st.markdown(f"### {mensagens.TITULO_SINTESE}")
    st.markdown(escapar_markdown(briefing.sintese_executiva.texto))

    if not briefing.recomendacoes:
        if briefing.variante == "evidencia_insuficiente":
            st.info(mensagens.EXPLICACAO_SEM_RECOMENDACAO)

    if briefing.pontos_de_conversa:
        st.markdown(f"### {mensagens.TITULO_PONTOS}")
        for ponto in briefing.pontos_de_conversa:
            st.markdown(f"- {escapar_markdown(ponto.texto)}")
    if briefing.avisos:
        st.markdown(f"### {mensagens.TITULO_AVISOS}")
        for aviso in briefing.avisos:
            st.warning(escapar_markdown(texto_legivel(aviso)))


def renderizar_evidencias_confirmadas(perfil) -> None:
    if perfil is None:
        return
    confirmadas = [
        item for item in perfil.afirmacoes_validadas if item.situacao == "confirmada"
    ]
    if confirmadas:
        st.markdown(f"### {mensagens.TITULO_EVIDENCIAS_CONFIRMADAS}")
        for item in confirmadas:
            with st.container(key=f"evidencia_confirmada_{item.id_afirmacao}"):
                st.markdown(escapar_markdown(item.texto))
                st.caption(rotulo(item.categoria))



def renderizar_incertezas(perfil) -> None:
    if perfil is None:
        return
    desconhecidos = [
        item for item in perfil.estado_dimensoes_gap if item.estado == "desconhecido"
    ]
    derrubadas = [
        item for item in perfil.afirmacoes_validadas if item.situacao == "derrubada"
    ]
    if desconhecidos:
        with st.expander(mensagens.TITULO_INFORMACOES_ABERTAS):
            for item in desconhecidos:
                st.markdown(f"- {rotulo(item.dimensao)}")

    if derrubadas:
        with st.expander(mensagens.TITULO_EVIDENCIAS_DESCARTADAS):
            for item in derrubadas:
                st.markdown(escapar_markdown(item.texto))
                st.caption(
                    f"{rotulo(item.categoria)} · "
                    + escapar_markdown(
                        texto_legivel(item.motivo or "Referência não confirmada.")
                    )
                )


def renderizar_resultado_analise(briefing) -> None:
    veredito = briefing.veredito
    st.markdown(f"### {mensagens.TITULO_RESULTADO_ATUAL}")
    if veredito.classe is not None:
        st.badge(veredito.classe, color=TOM_DA_CLASSE)
    if veredito.fit_score_total is not None:
        colunas = st.columns([1, 3], vertical_alignment="center")
        colunas[0].metric(
            mensagens.ROTULO_FIT_SCORE, f"{veredito.fit_score_total}/100"
        )
        colunas[1].progress(fracao_do_fit_score(veredito.fit_score_total))
        colunas[1].caption(mensagens.EXPLICACAO_FIT_SCORE)

    if briefing.variante == "evidencia_insuficiente":
        st.warning(RESUMO_DA_VARIANTE[briefing.variante])
        return
    if briefing.variante == "nao_aderente":
        st.info(mensagens.EXPLICACAO_NON_AI)
    else:
        st.caption(RESUMO_DA_VARIANTE[briefing.variante])
    st.markdown(f"**{mensagens.TITULO_TESE}**")
    with st.container(key="tese_principal"):
        st.markdown(escapar_markdown(veredito.tese))


def renderizar_necessidades(perfil) -> None:
    if perfil is None:
        return
    gaps = [
        item for item in perfil.estado_dimensoes_gap if item.estado == "gap_confirmado"
    ]
    if gaps:
        st.markdown(f"### {mensagens.TITULO_NECESSIDADES}")
        for item in gaps:
            with st.container(key=f"necessidade_{item.dimensao}"):
                st.markdown(f"**{rotulo(item.dimensao)}**")
                st.caption(
                    "A ausência foi declarada em uma fonte e passou pela conferência."
                )


def renderizar_fit_score(saida) -> None:
    fit_score = getattr(saida, "fit_score", None)
    if fit_score is None:
        return
    st.markdown(f"### {mensagens.TITULO_SCORE}")
    st.caption(mensagens.EXPLICACAO_FIT_SCORE)
    colunas = st.columns(4)
    for coluna, pilar in zip(colunas, fit_score.pilares, strict=True):
        with coluna, st.container(key=f"pilar_{pilar.pilar}"):
            st.metric(
                rotulo(pilar.pilar),
                f"{pilar.pontos}/{maximo_do_pilar(pilar.pilar)}",
            )
            st.caption(f"Faixa {rotulo(pilar.faixa).lower()}")
            for trava in pilar.travas_aplicadas:
                st.caption(rotulo(trava))


def renderizar_evidencias(briefing) -> None:
    """Evidência da startup e evidência NVIDIA, separadas e nunca confundidas."""
    if briefing.recomendacoes:
        st.markdown(f"### {mensagens.TITULO_EVIDENCIA_STARTUP}")
        st.caption(mensagens.LEGENDA_EVIDENCIA_STARTUP)
        for evidencia in evidencias_da_startup(briefing):
            st.caption(f'“{escapar_markdown(evidencia.trecho_citado)}”')
            st.markdown(
                f"[Abrir fonte pública]"
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
                f"{escapar_markdown(fonte.host_normalizado)} — {rotulo(fonte.tipo)}"
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
    partes = []
    for valor in (citacao.tecnologia, citacao.topico, citacao.breadcrumb):
        if valor is None:
            continue
        legivel = escapar_markdown(valor)
        if legivel.casefold() not in {item.casefold() for item in partes}:
            partes.append(legivel)
    partes.append(
        f"[Abrir referência NVIDIA]({destino_markdown(citacao.fonte_url)})"
    )
    return " — ".join(partes)


def renderizar_recomendacoes(recomendacoes) -> None:
    for ordem, recomendacao in enumerate(recomendacoes, start=1):
        with st.container(border=True, key=f"bloco_recomendacao_{ordem}"):
            st.caption(f"RECOMENDAÇÃO {ordem}")
            st.markdown(f"### {' + '.join(recomendacao.tecnologias)}")
            st.caption(rotular_fundamento(recomendacao))
            st.caption(
                f"Prioridade: {rotulo(recomendacao.prioridade)} · "
                f"Complexidade: {rotulo(recomendacao.complexidade)}"
            )
            st.markdown(escapar_markdown(recomendacao.justificativa_negocio))
            st.info(
                f"**Próximo passo — "
                f"{rotulo(recomendacao.proxima_acao.tipo_acao)}:** "
                + escapar_markdown(recomendacao.proxima_acao.detalhe)
            )
            with st.expander("Entenda a solução técnica"):
                st.markdown(escapar_markdown(recomendacao.justificativa_tecnica))
            with st.expander(mensagens.TITULO_LASTRO):
                st.markdown("**Fonte pública da startup**")
                for evidencia in recomendacao.evidencias_startup:
                    st.caption(f'“{escapar_markdown(evidencia.trecho_citado)}”')
                    st.markdown(
                        f"[Abrir fonte pública]"
                        f"({destino_markdown(evidencia.url_fonte)})"
                    )
                st.markdown("**Fonte técnica NVIDIA**")
                for citacao in recomendacao.citacoes_nvidia:
                    st.markdown(linha_de_citacao(citacao))


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------

pagina = renderizar_navegacao()
if pagina == mensagens.PAGINA_DASHBOARD:
    renderizar_dashboard()
else:
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
