import logging

import streamlit as st

from radar.agentes.query_planner import ErroQueryPlanner
from radar.aplicacao import criar_aplicacao
from radar.configuracao import ErroConfiguracao


logger = logging.getLogger(__name__)


st.set_page_config(page_title="NVIDIA Startup AI Radar", page_icon="📡", layout="wide")


@st.cache_resource
def obter_aplicacao():
    return criar_aplicacao()


st.title("NVIDIA Startup AI Radar")
st.caption(
    "Busca de startups reais por filtros estruturados e relevância FTS5/BM25. "
    "Este ranking ainda não é o fit-score NVIDIA."
)

with st.form("consulta_startups"):
    consulta = st.text_input(
        "O que você procura?",
        placeholder="Ex.: empresas brasileiras com modelos de linguagem em português",
    )
    buscar = st.form_submit_button("Buscar candidatas", type="primary")

if buscar:
    if not consulta.strip():
        st.warning("Escreva uma consulta antes de buscar.")
    else:
        try:
            with st.spinner("Planejando a consulta e recuperando documentos..."):
                saida = obter_aplicacao().executar_descoberta(consulta)
        except ErroConfiguracao as erro:
            st.error(str(erro))
        except ErroQueryPlanner as erro:
            logger.exception("O Query Planner interrompeu a consulta com segurança")
            st.error(str(erro))
        except Exception as erro:
            logger.exception("Falha inesperada ao executar a descoberta de startups")
            st.error(
                "A consulta não pôde ser concluída. Nenhum resultado foi inventado; "
                "consulte o terminal para diagnosticar a execução local."
            )
            st.caption(f"Tipo técnico da falha: {type(erro).__name__}")
        else:
            if saida.criterios_relaxados:
                st.info(
                    "A busca não encontrou candidatas inicialmente e relaxou: "
                    + ", ".join(saida.criterios_relaxados)
                    + "."
                )
            if not saida.ranking:
                st.warning(
                    "Nenhuma candidata encontrada após o limite de duas tentativas de relaxamento."
                )
            else:
                st.subheader(f"{len(saida.ranking)} candidata(s) por relevância")
                st.caption(
                    "No BM25 do SQLite, valores menores indicam maior relevância lexical."
                )
                for item in saida.ranking:
                    with st.container(border=True):
                        st.markdown(f"### {item.posicao}. {item.empresa.nome}")
                        st.write(item.empresa.descricao_curta)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Setor", item.empresa.setor)
                        col2.metric("Estágio", item.empresa.estagio)
                        col3.metric("Melhor BM25", f"{item.melhor_score_bm25:.6f}")
                        st.caption(f"Localização: {item.empresa.localizacao or 'não informada'}")
                        with st.expander("Documentos recuperados e fontes"):
                            for documento in item.documentos:
                                st.markdown(
                                    f"- [{documento.titulo}]({documento.url_fonte}) — "
                                    f"{documento.dominio_fonte}; acesso em "
                                    f"{documento.data_acesso.strftime('%d/%m/%Y')}; "
                                    f"BM25 {documento.score_bm25:.6f}"
                                )
                        st.button(
                            "Aprofundar análise (em desenvolvimento)",
                            key=f"aprofundar_{item.empresa.id_startup}",
                            disabled=True,
                        )
                with st.expander("Como esta consulta foi interpretada"):
                    st.json(saida.plano.model_dump(mode="json"))
                    st.caption("Fluxo executado: " + " → ".join(saida.trajeto))
