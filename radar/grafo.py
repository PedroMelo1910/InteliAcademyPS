from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from radar.agentes.classifier import Classifier
from radar.agentes.extractor import Extractor
from radar.agentes.query_planner import QueryPlanner
from radar.agentes.retriever import Retriever
from radar.agentes.roteadores import rotear_r1
from radar.base_startups import BaseStartups
from radar.contratos import EstadoRadar
from radar.provedores import (
    ProvedorClassificacao,
    ProvedorPerfilExtraido,
    ProvedorPlanoConsulta,
)


def montar_grafo(
    base: BaseStartups,
    provedor_plano: ProvedorPlanoConsulta,
    provedor_extracao: ProvedorPerfilExtraido,
    provedor_classificacao: ProvedorClassificacao,
    caminho_checkpoints: Path,
):
    caminho_checkpoints.parent.mkdir(parents=True, exist_ok=True)
    conexao_checkpoints = sqlite3.connect(caminho_checkpoints, check_same_thread=False)
    checkpointer = SqliteSaver(conexao_checkpoints)

    construtor = StateGraph(EstadoRadar)
    construtor.add_node("query_planner", QueryPlanner(base, provedor_plano))
    construtor.add_node("retriever", Retriever(base))
    construtor.add_node("extractor", Extractor(base, provedor_extracao))
    construtor.add_node("classifier", Classifier(provedor_classificacao))
    construtor.add_edge(START, "query_planner")
    construtor.add_edge("query_planner", "retriever")
    construtor.add_conditional_edges(
        "retriever",
        rotear_r1,
        {
            "analisar": "extractor",
            "candidatas_prontas": END,
            "relaxar": "query_planner",
            "sem_resultado": END,
        },
    )
    construtor.add_edge("extractor", "classifier")
    # O grafo termina aqui por ora: o Evidence Validator é o próximo marco.
    construtor.add_edge("classifier", END)
    return construtor.compile(checkpointer=checkpointer), conexao_checkpoints
