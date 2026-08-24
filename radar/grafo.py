from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from radar.agentes.query_planner import QueryPlanner
from radar.agentes.retriever import Retriever
from radar.agentes.roteadores import rotear_r1
from radar.base_startups import BaseStartups
from radar.contratos import EstadoRadar
from radar.provedores import ProvedorPlanoConsulta


def montar_grafo(
    base: BaseStartups,
    provedor: ProvedorPlanoConsulta,
    caminho_checkpoints: Path,
):
    caminho_checkpoints.parent.mkdir(parents=True, exist_ok=True)
    conexao_checkpoints = sqlite3.connect(caminho_checkpoints, check_same_thread=False)
    checkpointer = SqliteSaver(conexao_checkpoints)

    construtor = StateGraph(EstadoRadar)
    construtor.add_node("query_planner", QueryPlanner(base, provedor))
    construtor.add_node("retriever", Retriever(base))
    construtor.add_edge(START, "query_planner")
    construtor.add_edge("query_planner", "retriever")
    construtor.add_conditional_edges(
        "retriever",
        rotear_r1,
        {
            "analisar": END,
            "candidatas_prontas": END,
            "relaxar": "query_planner",
            "sem_resultado": END,
        },
    )
    return construtor.compile(checkpointer=checkpointer), conexao_checkpoints

