"""Grafo completo sobre documentos sintéticos escritos para sustentar os fatos.

Diferente da fronteira da curadoria real, aqui os provedores são fakes — e por
isso os documentos são escritos de propósito para que cada trecho citado
realmente sustente a afirmação que o cita. Isto prova mecânica: roteamento,
contratos, elegibilidade por gap e por sinal, descarte e falha segura. **Não**
prova qualidade de análise de nenhuma startup real.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from radar.agentes.extractor import ErroExtractor
from radar.aplicacao import SaidaDescoberta, criar_aplicacao
from radar.base_startups import inicializar_banco
from radar.contratos import Briefing, FiltrosEstruturados, PlanoConsulta
from tests.conftest import ConsultorNvidiaFalso, trecho_nvidia_falso

DATA_FIXA = date(2026, 9, 5)

DOC_PRODUTO = (
    "A Sintética Robótica opera veículos autônomos em pátios industriais fechados. "
    "O sistema integra radar, GPS e visão computacional para navegar sem motorista. "
    "A plataforma processa 2 bilhões de amostras de telemetria por mês em oito clientes. "
    "A empresa declara que não possui canal próprio de distribuição e depende de "
    "revendas parceiras para chegar ao cliente final."
)
DOC_SETOR = (
    "A Sintética Robótica atua no setor de saúde suplementar e atende operadoras "
    "no Brasil inteiro, segundo o material institucional publicado pela empresa."
)

DOC_EQUIPE = (
    "A Sintética Robótica contrata pessoa engenheira de percepção para o time de "
    "autonomia, com experiência em otimização de inferência em GPUs e MLOps."
)

TRECHO_VISAO = "integra radar, GPS e visão computacional para navegar sem motorista"
TRECHO_ESCALA = "processa 2 bilhões de amostras de telemetria por mês em oito clientes"
TRECHO_GAP = "não possui canal próprio de distribuição e depende de revendas parceiras"
TRECHO_SETOR = "atua no setor de saúde suplementar e atende operadoras"


def _curadoria(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    registro = {
        "nome": "Sintética Robótica",
        "site": "https://sintetica.example.com",
        "setor": "Saúde",
        "estagio": "série A",
        "localizacao": "Campinas, SP",
        "descricao_curta": "Veículos autônomos para pátios industriais.",
        "ano_fundacao": 2020,
        "tamanho_time": "51-200",
        "classe_referencia": "AI-native",
        "documentos": [
            {
                "tipo": "site institucional",
                "titulo": "Produto",
                "conteudo_texto": DOC_PRODUTO,
                "url_fonte": "https://sintetica.example.com/produto",
                "dominio_fonte": "sintetica.example.com",
                "data_publicacao": "2026-03-10",
                "data_acesso": "2026-09-01",
            },
            {
                "tipo": "vaga",
                "titulo": "Engenharia de percepção",
                "conteudo_texto": DOC_EQUIPE,
                "url_fonte": "https://vagas.example.org/sintetica-percepcao",
                "dominio_fonte": "vagas.example.org",
                "data_publicacao": "2026-05-01",
                "data_acesso": "2026-09-01",
            },
            {
                "tipo": "notícia",
                "titulo": "Setor",
                "conteudo_texto": DOC_SETOR,
                "url_fonte": "https://noticia.example.org/sintetica",
                "dominio_fonte": "noticia.example.org",
                "data_publicacao": "2026-04-01",
                "data_acesso": "2026-09-01",
            },
        ],
    }
    (destino / "01_sintetica.json").write_text(
        json.dumps(registro, ensure_ascii=False), encoding="utf-8"
    )


class ProvedorConstante:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0

    def invocar(self, mensagens):
        self.chamadas += 1
        if isinstance(self.resposta, Exception):
            raise self.resposta
        return self.resposta


def afirmacao(id_afirmacao, categoria, polaridade, id_documento, trecho, sinais=()):
    return {
        "id_afirmacao": id_afirmacao,
        "texto": f"Fato sintético número {id_afirmacao} descrito na fonte.",
        "categoria": categoria,
        "polaridade": polaridade,
        "id_documento": id_documento,
        "trecho_citado": trecho,
        "sinais_tecnicos": list(sinais),
    }


def perfil(afirmacoes, id_startup):
    return {
        "id_startup": id_startup,
        "resumo_produto": (
            "A empresa opera veículos autônomos em pátio industrial. "
            "A oferta chega ao cliente por revendas parceiras."
        ),
        "afirmacoes": afirmacoes,
    }


CLASSIFICACAO = {
    "classe": "AI-enabled",
    "justificativa": (
        "O material público descreve o produto vendido pela empresa. "
        "A evidência citada sustenta a classe atribuída."
    ),
    "ids_afirmacoes_suporte": [1],
}

BRIEFING = {
    "tese": {
        "texto": "A empresa tem oportunidade NVIDIA sustentada na base pública.",
        "ids_afirmacoes_suporte": [1],
    },
    "sintese_executiva": {
        "texto": "O material público sustenta a operação descrita na fonte.",
        "ids_afirmacoes_suporte": [1],
    },
    "pontos_de_conversa": [
        {"texto": "Perguntar sobre o custo por inferência.", "ids_afirmacoes_suporte": [1]},
        {"texto": "Explorar o plano de expansão comercial.", "ids_afirmacoes_suporte": [1]},
    ],
}


def contexto_com_cuda():
    """Contexto recuperado que realmente contém um chunk de CUDA.

    Sem ele, recomendar CUDA sustentando com um chunk de NIM passaria pela
    proveniência formal sem ter lastro sobre a tecnologia oferecida.
    """
    from radar.contratos import ContextoNvidia

    return ContextoNvidia(
        consulta_gerada="visao computacional em pátio industrial",
        trechos=[
            trecho_nvidia_falso(109, tecnologia="CUDA", score_rerank=0.95),
            trecho_nvidia_falso(101, tecnologia="NVIDIA NIM", score_rerank=0.9),
            trecho_nvidia_falso(
                102, tecnologia="NVIDIA Triton Inference Server", score_rerank=0.85
            ),
            trecho_nvidia_falso(107, tecnologia="NVIDIA Inception", score_rerank=0.8),
            trecho_nvidia_falso(
                106, tecnologia=None, topico="ai-native-services", score_rerank=0.7
            ),
        ],
    )


def rascunho(tipo, identificador, tecnologias, ids_afirmacoes=(1,), ids_chunks=(107,)):
    return {
        "rascunhos": [
            {
                "tipo_fundamento": tipo,
                "identificador_fundamento": identificador,
                "tecnologias": list(tecnologias),
                "justificativa_tecnica": "O trecho NVIDIA descreve a aceleração citada.",
                "justificativa_negocio": "O ganho aparece no custo operacional do cliente.",
                "proxima_acao": {
                    "tipo_acao": "call_tecnica_descoberta",
                    "detalhe": "Agendar a call técnica de descoberta nesta semana.",
                },
                "ids_afirmacoes": list(ids_afirmacoes),
                "ids_chunks": list(ids_chunks),
            }
        ]
    }


@pytest.fixture
def montar(tmp_path):
    """Devolve uma fábrica de aplicação sintética com provedores injetados."""

    def fabricar(
        *, extracao, classificacao=None, recomendacao=None, briefing=None, consultor=None
    ):
        curadoria = tmp_path / "curadoria"
        _curadoria(curadoria)
        banco = tmp_path / "sintetico.db"
        inicializar_banco(banco, curadoria)
        with sqlite3.connect(banco) as conexao:
            id_startup = conexao.execute("SELECT id FROM startups").fetchone()[0]
        aplicacao = criar_aplicacao(
            ProvedorConstante(
                {
                    "filtros": {},
                    "termos_busca": ["sintetica"],
                    "sinais_ia": [],
                    "foco_analise": "verificação sintética do grafo",
                }
            ),
            banco,
            tmp_path / "checkpoints_sintetico.db",
            ProvedorConstante(extracao(id_startup)),
            ProvedorConstante(classificacao or CLASSIFICACAO),
            consultor or ConsultorNvidiaFalso(),
            ProvedorConstante(
                recomendacao
                if recomendacao is not None
                else rascunho(
                    "gap_confirmado", "distribuicao", ["NVIDIA Inception"], (2,)
                )
            ),
            ProvedorConstante(briefing or BRIEFING),
            relogio=lambda: DATA_FIXA,
        )
        plano = PlanoConsulta(
            filtros=FiltrosEstruturados(),
            termos_busca=["sintetica"],
            sinais_ia=[],
            foco_analise="verificação sintética do grafo",
        )
        descoberta = SaidaDescoberta(
            consulta="verificação sintética",
            rota="candidatas_prontas",
            plano=plano,
            resultado=aplicacao.base.recuperar(plano, id_startup),
            ranking=(),
            tentativas_relaxamento=0,
            criterios_relaxados=(),
            trajeto=("query_planner", "retriever"),
        )
        return aplicacao, descoberta, id_startup

    return fabricar


# ----------------------------------------------------------------------
# Perfis sintéticos, cada um com trecho que sustenta de fato a afirmação
# ----------------------------------------------------------------------


def perfil_gap(id_startup):
    return perfil(
        [
            afirmacao(1, "stack_propria", "neutro", 1, TRECHO_VISAO),
            afirmacao(2, "distribuicao", "ausencia_explicita", 1, TRECHO_GAP),
        ],
        id_startup,
    )


def perfil_oportunidade(id_startup):
    return perfil(
        [
            afirmacao(
                1, "stack_propria", "neutro", 1, TRECHO_VISAO, ["visao_computacional"]
            )
        ],
        id_startup,
    )


def perfil_so_setor(id_startup):
    return perfil(
        # o documento de setor é o terceiro da curadoria sintética
        [afirmacao(1, "outro", "neutro", 3, TRECHO_SETOR)],
        id_startup,
    )


def perfil_sinal_derrubado(id_startup):
    """O trecho não existe no documento: o Evidence Validator derruba tudo."""
    return perfil(
        [
            afirmacao(
                1,
                "stack_propria",
                "neutro",
                1,
                "esta frase inventada nao aparece em nenhum documento da base",
                ["visao_computacional"],
            )
        ],
        id_startup,
    )


def perfil_polaridade_redundante(id_startup):
    return perfil(
        [
            afirmacao(1, "stack_propria", "presenca", 1, TRECHO_VISAO),
            afirmacao(2, "distribuicao", "ausencia_explicita", 1, TRECHO_GAP),
        ],
        id_startup,
    )


def executar(aplicacao, descoberta, id_startup):
    return aplicacao.executar_aprofundamento(descoberta, id_startup)


# ----------------------------------------------------------------------
# Cenários
# ----------------------------------------------------------------------


def test_briefing_normal_fundamentado_em_gap_confirmado(montar):
    saida = executar(*montar(extracao=perfil_gap))

    assert saida.briefing.variante == "normal"
    recomendacao = saida.briefing.recomendacoes[0]
    assert recomendacao.tipo_fundamento == "gap_confirmado"
    assert recomendacao.identificador_fundamento == "distribuicao"


def test_briefing_normal_fundamentado_em_oportunidade_confirmada(montar):
    saida = executar(
        *montar(
            extracao=perfil_oportunidade,
            recomendacao=rascunho(
                "oportunidade_confirmada",
                "visao_computacional",
                ["CUDA"],
                ids_chunks=(109,),
            ),
            consultor=ConsultorNvidiaFalso(contexto=contexto_com_cuda()),
        )
    )

    assert saida.briefing.variante == "normal"
    recomendacao = saida.briefing.recomendacoes[0]
    assert recomendacao.tipo_fundamento == "oportunidade_confirmada"
    citacoes = recomendacao.citacoes_nvidia
    assert any(
        item.origem == "tecnologia" and item.tecnologia == "CUDA" for item in citacoes
    )
    assert recomendacao.identificador_fundamento == "visao_computacional"
    assert recomendacao.tecnologias == ["CUDA"]


def test_briefing_nao_aderente_para_classe_non_ai(montar):
    saida = executar(
        *montar(
            extracao=perfil_oportunidade,
            classificacao={**CLASSIFICACAO, "classe": "non-AI"},
        )
    )

    assert saida.briefing.variante == "nao_aderente"
    assert saida.briefing.veredito.fit_score_total == 0
    assert saida.briefing.recomendacoes == []


def test_briefing_de_evidencia_insuficiente_quando_so_ha_setor(montar):
    saida = executar(*montar(extracao=perfil_so_setor))

    assert saida.briefing.variante == "evidencia_insuficiente"
    assert saida.briefing.veredito.classe is None
    assert saida.briefing.recomendacoes == []


def test_setor_sozinho_nao_cria_fundamento_nenhum(montar):
    saida = executar(*montar(extracao=perfil_so_setor))

    assert any("nenhum fundamento" in erro for erro in saida.erros)


def test_evidencia_de_sinal_derrubada_nao_sustenta_oportunidade(montar):
    saida = executar(
        *montar(
            extracao=perfil_sinal_derrubado,
            recomendacao=rascunho(
                "oportunidade_confirmada", "visao_computacional", ["CUDA"]
            ),
        )
    )

    assert saida.briefing.variante == "evidencia_insuficiente"


def test_oportunidade_sem_lastro_no_perfil_e_descartada(montar):
    saida = executar(
        *montar(
            extracao=perfil_gap,  # nenhum sinal técnico neste perfil
            recomendacao=rascunho(
                "oportunidade_confirmada", "robotica_ou_simulacao", ["NVIDIA Isaac"]
            ),
        )
    )

    assert saida.briefing.variante == "evidencia_insuficiente"
    assert any("não está sustentad" in erro for erro in saida.erros)


def test_tecnologia_fora_do_conjunto_do_sinal_e_descartada(montar):
    saida = executar(
        *montar(
            extracao=perfil_oportunidade,
            recomendacao=rascunho(
                "oportunidade_confirmada", "visao_computacional", ["NVIDIA Riva"]
            ),
        )
    )

    assert saida.briefing.variante == "evidencia_insuficiente"
    assert any("não são candidatas" in erro for erro in saida.erros)


def test_falha_de_provedor_nao_produz_briefing_parcial(montar):
    aplicacao, descoberta, id_startup = montar(extracao=perfil_gap)
    aplicacao.grafo  # o grafo existe; o provedor do Extractor é que falha
    quebrado, _, _ = montar(extracao=lambda _: RuntimeError("provedor fora do ar"))

    with pytest.raises(Exception):
        executar(quebrado, descoberta, id_startup)


def test_polaridade_redundante_e_normalizada_e_o_fluxo_segue(montar):
    saida = executar(*montar(extracao=perfil_polaridade_redundante))

    assert saida.briefing.variante == "normal"


def test_o_briefing_sobrevive_a_ida_e_volta_de_serializacao(montar):
    saida = executar(*montar(extracao=perfil_gap))

    assert Briefing.model_validate(saida.briefing.model_dump()) == saida.briefing


def test_o_checkpoint_registra_a_execucao(montar, tmp_path):
    executar(*montar(extracao=perfil_gap))

    with sqlite3.connect(tmp_path / "checkpoints_sintetico.db") as conexao:
        linhas = conexao.execute("SELECT count(*) FROM checkpoints").fetchone()[0]

    assert linhas > 0
