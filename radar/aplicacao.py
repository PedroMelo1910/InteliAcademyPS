from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

from dotenv import load_dotenv

from radar.agentes.roteadores import rotear_r1
from radar.base_startups import BaseStartups, inicializar_banco
from radar.conhecimento_nvidia import ConhecimentoNvidia
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
    ProvedorClassificacao,
    ProvedorContextoNvidia,
    ProvedorEmbeddingNvidia,
    ProvedorGeminiClassificacao,
    ProvedorGeminiPerfilExtraido,
    ProvedorGeminiPlanoConsulta,
    ProvedorGeminiRecomendacaoRascunho,
    ProvedorPerfilExtraido,
    ProvedorPlanoConsulta,
    ProvedorRecomendacaoRascunho,
    ProvedorRerankListwiseGemini,
    ProvedorRerankNvidia,
    RerankComFallback,
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
    provedor_classificacao: ProvedorClassificacao | None = None,
    consultor_nvidia: ProvedorContextoNvidia | None = None,
    provedor_recomendacao: ProvedorRecomendacaoRascunho | None = None,
) -> AplicacaoRadar:
    inicializar_banco(caminho_banco, CAMINHO_DADOS_CURADOS)
    injetados = (
        provedor,
        provedor_extracao,
        provedor_classificacao,
        consultor_nvidia,
        provedor_recomendacao,
    )
    if any(item is not None for item in injetados) and any(
        item is None for item in injetados
    ):
        raise ErroConfiguracao(
            "Para injeção offline, informe juntos os provedores do Query Planner, "
            "do Extractor, do Classifier, do NVIDIA RAG e do Recommendation."
        )
    if all(item is None for item in injetados):
        load_dotenv(RAIZ_PROJETO / ".env")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ErroConfiguracao(
                "GOOGLE_API_KEY não está configurada no .env local. "
                "Adicione a chave e reinicie a aplicação."
            )
        chave_nvidia = os.getenv("NVIDIA_API_KEY", "").strip()
        if not chave_nvidia:
            raise ErroConfiguracao(
                "NVIDIA_API_KEY não está configurada no .env local. "
                "O caminho aderente consulta a base de conhecimento NVIDIA; "
                "adicione a chave e reinicie a aplicação."
            )
        provedor = ProvedorGeminiPlanoConsulta(api_key)
        provedor_extracao = ProvedorGeminiPerfilExtraido(api_key)
        provedor_classificacao = ProvedorGeminiClassificacao(api_key)
        # A composição de reranking aprovada no Entregável 2 é reusada como
        # está: NVIDIA primário e fallback listwise no backbone LLM.
        consultor_nvidia = ConhecimentoNvidia(
            caminho_banco,
            ProvedorEmbeddingNvidia(chave_nvidia),
            RerankComFallback(
                ProvedorRerankNvidia(chave_nvidia),
                ProvedorRerankListwiseGemini(api_key),
            ),
        )
        provedor_recomendacao = ProvedorGeminiRecomendacaoRascunho(api_key)
    assert (
        provedor is not None
        and provedor_extracao is not None
        and provedor_classificacao is not None
        and consultor_nvidia is not None
        and provedor_recomendacao is not None
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor,
        provedor_extracao,
        provedor_classificacao,
        caminho_checkpoints,
        consultor_nvidia,
        provedor_recomendacao,
    )
    return AplicacaoRadar(grafo, conexao)
