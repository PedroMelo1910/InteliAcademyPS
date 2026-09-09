"""Orquestra a descoberta, o ranking e o aprofundamento usados pela interface."""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv

from radar.agentes.roteadores import rotear_r1
from radar.base_startups import BaseStartups, preparar_cache_analises
from radar.conhecimento_nvidia import ConhecimentoNvidia
from radar.configuracao import (
    CAMINHO_BANCO,
    CAMINHO_CHECKPOINTS,
    ErroConfiguracao,
    RAIZ_PROJETO,
)
from radar.contratos import (
    AnalisePersistida,
    Briefing,
    ClasseStartup,
    DocumentoRecuperado,
    EmpresaCandidata,
    EstadoRadar,
    FitScore,
    PlanoConsulta,
    PerfilValidado,
    ResultadoR1,
    ResultadoRecuperacao,
    StatusAnaliseRanking,
)
from radar.grafo import montar_grafo
from radar.provedores import (
    ProvedorBriefingRascunho,
    ProvedorClassificacao,
    ProvedorContextoNvidia,
    ProvedorEmbeddingNvidia,
    ProvedorGeminiBriefingRascunho,
    ProvedorGeminiClassificacao,
    ProvedorGeminiPerfilExtraido,
    ProvedorGeminiPlanoConsulta,
    ProvedorGeminiRecomendacaoRascunho,
    ProvedorGroqBriefingRascunho,
    ProvedorGroqClassificacao,
    ProvedorGroqPerfilExtraido,
    ProvedorGroqPlanoConsulta,
    ProvedorGroqRecomendacaoRascunho,
    compor_com_reserva,
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
    status_analise: StatusAnaliseRanking
    classe: ClasseStartup | None
    fit_score_total: int | None
    justificativa_fit_score: str | None
    motivo_evidencia_insuficiente: str | None


@dataclass(frozen=True)
class SaidaDescoberta:
    consulta: str
    rota: ResultadoR1
    plano: PlanoConsulta
    resultado: ResultadoRecuperacao
    ranking: tuple[ItemRanking, ...]
    tentativas_relaxamento: int
    criterios_relaxados: tuple[str, ...]
    trajeto: tuple[str, ...]


CriterioOrdenacaoRanking = Literal["fit_score", "relevancia"]


_PALAVRAS_ESTRUTURAIS_IGNORADAS = frozenset(
    {
        "a",
        "as",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "para",
        "por",
    }
)


def _tokens_comparaveis(texto: str | None) -> frozenset[str]:
    """Normaliza texto para comparar atributos sem depender de acento ou caixa."""
    if not texto:
        return frozenset()
    sem_acentos = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", texto.casefold())
        if not unicodedata.combining(caractere)
    )
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", sem_acentos)
        if token not in _PALAVRAS_ESTRUTURAIS_IGNORADAS
    )


def _aderencia_estruturada(item: ItemRanking, consulta: str) -> float:
    """Mede quanto dos atributos da empresa foi pedido literalmente na consulta.

    A recuperação pode relaxar setor, estágio ou localização para não devolver
    uma lista curta demais. Na visualização por relação, esses atributos ainda
    devem funcionar como preferência: uma empresa de Saúde deve preceder uma
    varejista quando a pessoa escreveu ``saúde``, mesmo que a varejista tenha
    um documento com uma expressão técnica rara da mesma pergunta.
    """
    tokens_consulta = _tokens_comparaveis(consulta)
    if not tokens_consulta:
        return 0.0

    afinidades: list[float] = []
    for valor in (
        item.empresa.nome,
        item.empresa.setor,
        item.empresa.estagio,
        item.empresa.localizacao,
    ):
        tokens_valor = _tokens_comparaveis(valor)
        if tokens_valor:
            afinidades.append(len(tokens_valor & tokens_consulta) / len(tokens_valor))
    # A soma recompensa quem preserva mais de uma restrição da pergunta
    # (por exemplo, setor e localização), em vez de tratar um único acerto
    # isolado como equivalente ao conjunto completo da intenção.
    return sum(afinidades)


def personalizar_ranking(
    ranking: tuple[ItemRanking, ...],
    *,
    criterio: CriterioOrdenacaoRanking,
    classe: ClasseStartup | None = None,
    consulta: str = "",
) -> tuple[ItemRanking, ...]:
    """Filtra e reordena candidatas sem recalcular nenhuma das duas medidas.

    O fit-score e o BM25 já chegam prontos da aplicação. Esta função apenas
    escolhe qual deles governa a ordem visível e renumera o recorte escolhido
    pelo usuário. Assim, a interface não precisa conhecer regras de domínio.
    """
    filtrados = tuple(
        item for item in ranking if classe is None or item.classe == classe
    )

    if criterio == "fit_score":
        for item in filtrados:
            if item.status_analise == "concluida" and item.fit_score_total is None:
                raise ErroAplicacao(
                    "análise concluída sem FitScore para a startup "
                    f"{item.empresa.id_startup}: a ordenação não inventa pontuação"
                )
        ordenados = sorted(
            filtrados,
            key=lambda item: (
                0 if item.status_analise == "concluida" else 1,
                -(item.fit_score_total or 0),
                item.melhor_score_bm25,
                item.empresa.nome.casefold(),
                item.empresa.id_startup,
            ),
        )
    elif criterio == "relevancia":
        # Primeiro preserva atributos explicitamente pedidos que podem ter sido
        # relaxados na recuperação. Depois usa o BM25 do SQLite, no qual valores
        # menores representam maior proximidade lexical.
        ordenados = sorted(
            filtrados,
            key=lambda item: (
                -_aderencia_estruturada(item, consulta),
                item.melhor_score_bm25,
                item.empresa.nome.casefold(),
                item.empresa.id_startup,
            ),
        )
    else:
        raise ValueError(f"critério de ordenação desconhecido: {criterio}")

    return tuple(
        replace(item, posicao=posicao)
        for posicao, item in enumerate(ordenados, start=1)
    )


def construir_ranking(
    resultado: ResultadoRecuperacao,
    analises: Mapping[int, AnalisePersistida] | None = None,
) -> tuple[ItemRanking, ...]:
    """Cruza relevância lexical com o cache, sem misturar as duas medidas."""
    analises = analises or {}
    documentos_por_empresa: dict[int, list[DocumentoRecuperado]] = {}
    for documento in resultado.documentos:
        documentos_por_empresa.setdefault(documento.id_startup, []).append(documento)

    def melhor_bm25(empresa: EmpresaCandidata) -> float:
        return min(
            (
                documento.score_bm25
                for documento in documentos_por_empresa.get(empresa.id_startup, [])
            ),
            default=float("inf"),
        )

    def chave(empresa: EmpresaCandidata):
        analise = analises.get(empresa.id_startup)
        if analise is None:
            grupo, total = 1, 0
        elif analise.status == "evidencia_insuficiente":
            grupo, total = 1, 0
        else:
            if analise.fit_score is None:
                # `assert` sumiria sob python -O e o ranking cairia num
                # AttributeError obscuro. Uma concluída sem FitScore é um
                # cache corrompido: falhar alto é melhor do que rebaixá-la em
                # silêncio ou arbitrar uma pontuação que ninguém calculou.
                raise ErroAplicacao(
                    "análise concluída sem FitScore para a startup "
                    f"{empresa.id_startup}: o ranking não inventa pontuação"
                )
            grupo = 0
            total = analise.fit_score.total
        return (
            grupo,
            -total,
            melhor_bm25(empresa),
            empresa.nome.casefold(),
            empresa.id_startup,
        )

    ordenadas = sorted(resultado.empresas, key=chave)
    itens: list[ItemRanking] = []
    for posicao, empresa in enumerate(ordenadas, start=1):
        documentos = tuple(documentos_por_empresa.get(empresa.id_startup, []))
        analise = analises.get(empresa.id_startup)
        status: StatusAnaliseRanking = (
            analise.status if analise is not None else "ausente"
        )
        itens.append(
            ItemRanking(
                posicao=posicao,
                empresa=empresa,
                melhor_score_bm25=min(
                    (documento.score_bm25 for documento in documentos),
                    default=0.0,
                ),
                documentos=documentos,
                status_analise=status,
                classe=analise.classe if analise is not None else None,
                fit_score_total=(
                    analise.fit_score.total
                    if analise is not None and analise.fit_score is not None
                    else None
                ),
                justificativa_fit_score=(
                    analise.fit_score.justificativa_curta
                    if analise is not None and analise.fit_score is not None
                    else None
                ),
                motivo_evidencia_insuficiente=(
                    analise.motivo_evidencia_insuficiente
                    if analise is not None
                    else None
                ),
            )
        )
    return tuple(itens)


@dataclass(frozen=True)
class SaidaAprofundamento:
    """O que a interface recebe ao clicar numa candidata da descoberta."""

    briefing: Briefing
    plano: PlanoConsulta
    id_startup: int
    trajeto: tuple[str, ...]
    erros: tuple[str, ...]
    perfil_validado: PerfilValidado | None = None
    fit_score: FitScore | None = None


class ErroAplicacao(RuntimeError):
    """Uso inválido da fronteira da aplicação, antes de tocar o grafo."""


class AplicacaoRadar:
    def __init__(self, grafo, conexao_checkpoints, base: BaseStartups):
        self.grafo = grafo
        self._conexao_checkpoints = conexao_checkpoints
        self.base = base

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
        analises = self.base.carregar_analises(
            [empresa.id_startup for empresa in resultado.empresas]
        )
        return SaidaDescoberta(
            consulta=consulta,
            rota=rota,
            plano=plano,
            resultado=resultado,
            ranking=construir_ranking(resultado, analises),
            tentativas_relaxamento=int(estado_final.get("tentativas_relaxamento", 0)),
            criterios_relaxados=tuple(estado_final.get("criterios_relaxados", [])),
            trajeto=tuple(estado_final.get("trajeto", [])),
        )


    def executar_aprofundamento(
        self, descoberta: SaidaDescoberta, id_startup: int
    ) -> SaidaAprofundamento:
        """Segunda invocação do mesmo grafo, com a startup escolhida (§1.3).

        Não há cirurgia de thread nem ``interrupt``: o estado inicial reaproveita
        a consulta e o ``PlanoConsulta`` já validados da descoberta, o Query
        Planner pula (§2.3), o Retriever faz a busca pinada e o fluxo corre até
        o Briefing.
        """
        candidatas = {
            empresa.id_startup for empresa in descoberta.resultado.empresas
        }
        if id_startup not in candidatas:
            raise ErroAplicacao(
                f"a startup {id_startup} não está entre as candidatas desta "
                f"descoberta ({sorted(candidatas)}); a interface só aprofunda o "
                "que o ranking devolveu"
            )
        estado_inicial: EstadoRadar = {
            "consulta_usuario": descoberta.consulta,
            "startup_selecionada": id_startup,
            "plano_consulta": descoberta.plano,
            "tentativas_relaxamento": descoberta.tentativas_relaxamento,
            "tentativas_extracao": 0,
            "criterios_relaxados": list(descoberta.criterios_relaxados),
            "erros": [],
            "trajeto": [],
        }
        estado_final = self.grafo.invoke(
            estado_inicial,
            config={"configurable": {"thread_id": str(uuid4())}},
        )
        bruto = estado_final.get("briefing")
        if bruto is None:
            raise ErroAplicacao(
                "o grafo terminou sem briefing para a startup "
                f"{id_startup}; nenhum resultado parcial é exposto"
            )
        briefing = Briefing.model_validate(bruto)
        perfil_bruto = estado_final.get("perfil_validado")
        fit_bruto = estado_final.get("fit_score")
        perfil = (
            PerfilValidado.model_validate(perfil_bruto)
            if perfil_bruto is not None
            else None
        )
        fit_score = (
            FitScore.model_validate(fit_bruto)
            if fit_bruto is not None
            else None
        )
        return SaidaAprofundamento(
            briefing=briefing,
            plano=PlanoConsulta.model_validate(estado_final["plano_consulta"]),
            id_startup=id_startup,
            trajeto=tuple(estado_final.get("trajeto", [])),
            erros=tuple(estado_final.get("erros", [])),
            perfil_validado=perfil,
            fit_score=fit_score,
        )


def criar_aplicacao(
    provedor: ProvedorPlanoConsulta | None = None,
    caminho_banco=CAMINHO_BANCO,
    caminho_checkpoints=CAMINHO_CHECKPOINTS,
    provedor_extracao: ProvedorPerfilExtraido | None = None,
    provedor_classificacao: ProvedorClassificacao | None = None,
    consultor_nvidia: ProvedorContextoNvidia | None = None,
    provedor_recomendacao: ProvedorRecomendacaoRascunho | None = None,
    provedor_briefing: ProvedorBriefingRascunho | None = None,
    relogio: Callable[[], date] | None = None,
) -> AplicacaoRadar:
    # A base curada é estática em execução (§4.2) e ``aplicacao`` só orquestra
    # e exibe (§9.1): abrir a tela lê o banco, nunca o reconstrói. Semear aqui
    # refazia o upsert das startups e dos documentos e reconstruía o índice FTS
    # a cada boot do Streamlit — escrita no caminho de leitura, capaz de
    # colidir com um lote em andamento. O seed pertence ao comando explícito.
    if not caminho_banco.exists():
        raise ErroConfiguracao(
            "dados/radar.db não existe; execute primeiro "
            "python -m scripts.inicializar_base"
        )
    try:
        preparar_cache_analises(caminho_banco)
    except (sqlite3.DatabaseError, ValueError) as erro:
        raise ErroConfiguracao(str(erro)) from erro
    injetados = (
        provedor,
        provedor_extracao,
        provedor_classificacao,
        consultor_nvidia,
        provedor_recomendacao,
        provedor_briefing,
    )
    if any(item is not None for item in injetados) and any(
        item is None for item in injetados
    ):
        raise ErroConfiguracao(
            "Para injeção offline, informe juntos os provedores do Query Planner, "
            "do Extractor, do Classifier, do NVIDIA RAG, do Recommendation e "
            "do Briefing."
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
        # Gemini é o primário. A reserva Groq só entra quando GROQ_API_KEY
        # existe; sem ela, `compor_com_reserva` devolve o próprio provedor
        # Gemini e o comportamento fica idêntico ao de hoje.
        chave_reserva = os.getenv("GROQ_API_KEY", "").strip()

        def com_reserva(primario, fabrica, fronteira):
            return compor_com_reserva(
                primario,
                fabrica,
                fronteira=fronteira,
                chave_reserva=chave_reserva,
            )

        provedor = com_reserva(
            ProvedorGeminiPlanoConsulta(api_key),
            ProvedorGroqPlanoConsulta,
            "query_planner",
        )
        provedor_extracao = com_reserva(
            ProvedorGeminiPerfilExtraido(api_key),
            ProvedorGroqPerfilExtraido,
            "extractor",
        )
        provedor_classificacao = com_reserva(
            ProvedorGeminiClassificacao(api_key),
            ProvedorGroqClassificacao,
            "classifier",
        )
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
        provedor_recomendacao = com_reserva(
            ProvedorGeminiRecomendacaoRascunho(api_key),
            ProvedorGroqRecomendacaoRascunho,
            "recommendation",
        )
        provedor_briefing = com_reserva(
            ProvedorGeminiBriefingRascunho(api_key),
            ProvedorGroqBriefingRascunho,
            "briefing",
        )
    assert (
        provedor is not None
        and provedor_extracao is not None
        and provedor_classificacao is not None
        and consultor_nvidia is not None
        and provedor_recomendacao is not None
        and provedor_briefing is not None
    )
    base = BaseStartups(caminho_banco)
    grafo, conexao = montar_grafo(
        base,
        provedor,
        provedor_extracao,
        provedor_classificacao,
        caminho_checkpoints,
        consultor_nvidia,
        provedor_recomendacao,
        provedor_briefing,
        relogio,
    )
    return AplicacaoRadar(grafo, conexao, base)
