from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

from dotenv import load_dotenv

from radar.agentes.roteadores import rotear_r1
from radar.base_startups import BaseStartups, inicializar_banco
from radar.configuracao import (
    CAMINHO_BANCO,
    CAMINHO_CHECKPOINTS,
    CAMINHO_DADOS_CURADOS,
    ErroConfiguracao,
    RAIZ_PROJETO,
)
from radar.contratos import (
    DocumentoRecuperado,
    EmpresaCandidata,
    EstadoRadar,
    PlanoConsulta,
    ResultadoR1,
    ResultadoRecuperacao,
)
from radar.grafo import montar_grafo
from radar.provedores import (
    ProvedorGeminiPerfilExtraido,
    ProvedorGeminiPlanoConsulta,
    ProvedorPerfilExtraido,
    ProvedorPlanoConsulta,
)


@dataclass(frozen=True)
class ItemRanking:
    posicao: int
    empresa: EmpresaCandidata
    melhor_score_bm25: float
    documentos: tuple[DocumentoRecuperado, ...]


@dataclass(frozen=True)
class SaidaDescoberta:
    rota: ResultadoR1
    plano: PlanoConsulta
    resultado: ResultadoRecuperacao
    ranking: tuple[ItemRanking, ...]
    tentativas_relaxamento: int
    criterios_relaxados: tuple[str, ...]
    trajeto: tuple[str, ...]


def construir_ranking(resultado: ResultadoRecuperacao) -> tuple[ItemRanking, ...]:
    documentos_por_empresa: dict[int, list[DocumentoRecuperado]] = {}
    for documento in resultado.documentos:
        documentos_por_empresa.setdefault(documento.id_startup, []).append(documento)
    ordenadas = sorted(
        resultado.empresas,
        key=lambda empresa: min(
            (
                documento.score_bm25
                for documento in documentos_por_empresa.get(empresa.id_startup, [])
            ),
            default=float("inf"),
        ),
    )
    itens: list[ItemRanking] = []
    for posicao, empresa in enumerate(ordenadas, start=1):
        documentos = tuple(documentos_por_empresa.get(empresa.id_startup, []))
        itens.append(
            ItemRanking(
                posicao=posicao,
                empresa=empresa,
                melhor_score_bm25=min(
                    (documento.score_bm25 for documento in documentos),
                    default=0.0,
                ),
                documentos=documentos,
            )
        )
    return tuple(itens)


class AplicacaoRadar:
    def __init__(self, grafo, conexao_checkpoints):
        self.grafo = grafo
        self._conexao_checkpoints = conexao_checkpoints

    def executar_descoberta(self, consulta: str) -> SaidaDescoberta:
        estado_inicial: EstadoRadar = {
            "consulta_usuario": consulta,
            "startup_selecionada": None,
            "tentativas_relaxamento": 0,
            "tentativas_extracao": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        }
        estado_final = self.grafo.invoke(
            estado_inicial,
            config={"configurable": {"thread_id": str(uuid4())}},
        )
        resultado = ResultadoRecuperacao.model_validate(
            estado_final["resultado_recuperacao"]
        )
        plano = PlanoConsulta.model_validate(estado_final["plano_consulta"])
        rota = rotear_r1(estado_final)
        return SaidaDescoberta(
            rota=rota,
            plano=plano,
            resultado=resultado,
            ranking=construir_ranking(resultado),
            tentativas_relaxamento=int(estado_final.get("tentativas_relaxamento", 0)),
            criterios_relaxados=tuple(estado_final.get("criterios_relaxados", [])),
            trajeto=tuple(estado_final.get("trajeto", [])),
        )


def criar_aplicacao(
    provedor: ProvedorPlanoConsulta | None = None,
    caminho_banco=CAMINHO_BANCO,
    caminho_checkpoints=CAMINHO_CHECKPOINTS,
    provedor_extracao: ProvedorPerfilExtraido | None = None,
) -> AplicacaoRadar:
    inicializar_banco(caminho_banco, CAMINHO_DADOS_CURADOS)
    if (provedor is None) != (provedor_extracao is None):
        raise ErroConfiguracao(
            "Para injeção offline, informe juntos os provedores do Query Planner "
            "e do Extractor."
        )
    if provedor is None and provedor_extracao is None:
        load_dotenv(RAIZ_PROJETO / ".env")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ErroConfiguracao(
                "GOOGLE_API_KEY não está configurada no .env local. "
                "Adicione a chave e reinicie a aplicação."
            )
        provedor = ProvedorGeminiPlanoConsulta(api_key)
        provedor_extracao = ProvedorGeminiPerfilExtraido(api_key)
    assert provedor is not None and provedor_extracao is not None
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor,
        provedor_extracao,
        caminho_checkpoints,
    )
    return AplicacaoRadar(grafo, conexao)
