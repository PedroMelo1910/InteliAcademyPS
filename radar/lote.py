"""Orquestra a pré-análise persistida usada pelo ranking da interface."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from radar.agentes.roteadores import rotear_r3
from radar.base_startups import BaseStartups, preparar_cache_analises
from radar.configuracao import (
    CAMINHO_BANCO,
    CAMINHO_CHECKPOINTS,
    ErroConfiguracao,
    RAIZ_PROJETO,
)
from radar.contratos import (
    AnalisePersistida,
    Classificacao,
    EntradaFitScore,
    EstadoRadar,
    PerfilValidado,
    PlanoConsulta,
)
from radar.grafo import montar_grafo_lote
from radar.provedores import (
    ProvedorClassificacao,
    ProvedorGeminiClassificacao,
    ProvedorGeminiPerfilExtraido,
    ProvedorGroqClassificacao,
    ProvedorGroqPerfilExtraido,
    compor_com_reserva,
    ProvedorPerfilExtraido,
    falha_operacional,
)
from radar.recomendacao import VERSAO_RUBRICA, calcular_fit_score


CHAMADAS_LLM_MINIMAS_POR_STARTUP = 2
CHAMADAS_LLM_MAXIMAS_POR_STARTUP = 8
REQUISICOES_GEMINI_POR_SEGUNDO = 0.2
LIMITE_MENSAGEM_FALHA = 300

# A causa-raiz de uma falha vem de biblioteca de terceiros: pode carregar um
# corpo de resposta inteiro e, no pior caso, credencial ecoada da requisição.
# Estes padrões cobrem as formas concretas que este projeto pode produzir --
# chave do Google, chave da NVIDIA e rótulo genérico seguido de um segredo.
_PADROES_CREDENCIAL = (
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    re.compile(r"nvapi-[0-9A-Za-z_-]{10,}"),
    re.compile(
        r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|authorization|bearer|secret)\b"
        r"\s*[:=]?\s*[\"']?[A-Za-z0-9\-._~+/]{8,}=*"
    ),
)
MARCADOR_CREDENCIAL = "[credencial omitida]"
MARCADOR_TRUNCAGEM = "… [truncado]"


class ErroAnaliseLote(RuntimeError):
    """A pré-análise não produziu um artefato seguro para persistência."""


@dataclass(frozen=True)
class FalhaExecucaoLote:
    startup_id: int
    tipo: str
    mensagem: str


@dataclass(frozen=True)
class ResultadoExecucaoLote:
    concluidas: tuple[int, ...]
    evidencias_insuficientes: tuple[int, ...]
    falhas: tuple[FalhaExecucaoLote, ...]


def estimar_chamadas_externas(quantidade_startups: int) -> tuple[int, int]:
    """Faixa de chamadas lógicas, incluindo retries e uma reextração por R2."""
    if quantidade_startups < 0:
        raise ValueError("quantidade_startups não pode ser negativa")
    return (
        quantidade_startups * CHAMADAS_LLM_MINIMAS_POR_STARTUP,
        quantidade_startups * CHAMADAS_LLM_MAXIMAS_POR_STARTUP,
    )


class AnalisadorLote:
    """Executa uma startup por vez e grava somente resultados validados."""

    def __init__(
        self,
        base: BaseStartups,
        grafo,
        conexao_checkpoints,
        relogio: Callable[[], date] | None = None,
    ):
        self.base = base
        self.grafo = grafo
        self._conexao_checkpoints = conexao_checkpoints
        self._relogio = relogio or date.today

    def fechar(self) -> None:
        self._conexao_checkpoints.close()

    def executar_startup(self, startup_id: int) -> AnalisePersistida:
        recuperacao = self.base.recuperar_para_lote(startup_id)
        if len(recuperacao.empresas) != 1 or not recuperacao.documentos:
            raise ErroAnaliseLote(
                f"a startup {startup_id} não existe ou não possui documentos"
            )
        empresa = recuperacao.empresas[0]
        plano = PlanoConsulta(
            termos_busca=[empresa.nome],
            sinais_ia=[],
            foco_analise=(
                "Identificar o produto, a centralidade de IA, capacidades, gaps, "
                "momento da empresa e evidências técnicas explicitamente citadas."
            ),
        )
        estado_inicial: EstadoRadar = {
            "consulta_usuario": f"pré-análise persistida de {empresa.nome}",
            "startup_selecionada": startup_id,
            "plano_consulta": plano,
            "resultado_recuperacao": recuperacao,
            "tentativas_relaxamento": 0,
            "tentativas_extracao": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        }
        estado_final = self.grafo.invoke(
            estado_inicial,
            config={
                "configurable": {
                    "thread_id": f"lote-{startup_id}-{uuid4()}"
                }
            },
        )
        perfil = PerfilValidado.model_validate(estado_final.get("perfil_validado"))
        classificacao = Classificacao.model_validate(
            estado_final.get("classificacao")
        )
        rota = rotear_r3(estado_final)
        if rota == "evidencia_insuficiente":
            analise = AnalisePersistida(
                startup_id=startup_id,
                status="evidencia_insuficiente",
                classe=None,
                fit_score=None,
                perfil_validado=perfil,
                motivo_evidencia_insuficiente=_motivo_insuficiencia(
                    perfil, classificacao
                ),
                data_execucao=self._relogio(),
                versao_rubrica=VERSAO_RUBRICA,
            )
            self.base.salvar_analise(analise)
            return analise

        ids_documentos_recuperados = {
            item.id_documento for item in recuperacao.documentos
        }
        ids_documentos_perfil = {
            item.id_documento for item in perfil.afirmacoes_validadas
        }
        if not ids_documentos_perfil.issubset(ids_documentos_recuperados):
            raise ErroAnaliseLote(
                "o perfil validado referencia documentos fora da startup analisada"
            )
        metadados = self.base.carregar_metadados_fit_score(
            sorted(ids_documentos_perfil)
        )
        entrada = EntradaFitScore(
            classe=classificacao.classe,
            ids_afirmacoes_suporte_classe=sorted(
                classificacao.ids_afirmacoes_suporte
            ),
            perfil_validado=perfil,
            setor=empresa.setor,
            estagio=empresa.estagio,
            documentos=[metadados[item] for item in sorted(metadados)],
            data_referencia=max(item.data_acesso for item in recuperacao.documentos),
        )
        fit_score = calcular_fit_score(entrada)
        analise = AnalisePersistida(
            startup_id=startup_id,
            status="concluida",
            classe=classificacao.classe,
            fit_score=fit_score,
            perfil_validado=perfil,
            motivo_evidencia_insuficiente=None,
            data_execucao=self._relogio(),
            versao_rubrica=VERSAO_RUBRICA,
        )
        self.base.salvar_analise(analise)
        return analise

    def executar_todas(
        self, ids_startups: Sequence[int] | None = None
    ) -> ResultadoExecucaoLote:
        ids = (
            list(ids_startups)
            if ids_startups is not None
            else [item.id_startup for item in self.base.listar_startups_para_lote()]
        )
        concluidas: list[int] = []
        insuficientes: list[int] = []
        falhas: list[FalhaExecucaoLote] = []
        for startup_id in ids:
            try:
                analise = self.executar_startup(startup_id)
            except Exception as erro:  # a falha fica fora do cache e segue visível
                falhas.append(
                    FalhaExecucaoLote(
                        startup_id=startup_id,
                        tipo=_tipo_falha(erro),
                        mensagem=_mensagem_falha(erro),
                    )
                )
                continue
            if analise.status == "concluida":
                concluidas.append(startup_id)
            else:
                insuficientes.append(startup_id)
        return ResultadoExecucaoLote(
            concluidas=tuple(concluidas),
            evidencias_insuficientes=tuple(insuficientes),
            falhas=tuple(falhas),
        )


def _motivo_insuficiencia(
    perfil: PerfilValidado, classificacao: Classificacao
) -> str:
    confirmados = {
        item.id_afirmacao
        for item in perfil.afirmacoes_validadas
        if item.situacao == "confirmada"
    }
    if not confirmados:
        return "Nenhuma afirmação permaneceu confirmada após a validação."
    suporte_derrubado = sorted(
        set(classificacao.ids_afirmacoes_suporte) - confirmados
    )
    return (
        "A classificação depende de afirmações que não sobreviveram à validação: "
        f"{suporte_derrubado}."
    )


def _tipo_falha(erro: Exception) -> str:
    return (
        "falha_operacional"
        if falha_operacional(erro)
        else "falha_de_processamento"
    )


def _censurar(texto: str) -> str:
    """Troca material com forma de credencial por um marcador visível."""
    for padrao in _PADROES_CREDENCIAL:
        texto = padrao.sub(MARCADOR_CREDENCIAL, texto)
    return texto


def _resumir(texto: str) -> str:
    """Deixa a mensagem em uma linha, censurada e com tamanho previsível.

    O começo é o que informa — código de status, motivo, campo inválido —, então
    a truncagem preserva o prefixo e sinaliza o corte em vez de calar.
    """
    limpo = " ".join(_censurar(texto).split())
    if len(limpo) <= LIMITE_MENSAGEM_FALHA:
        return limpo
    return limpo[:LIMITE_MENSAGEM_FALHA].rstrip() + MARCADOR_TRUNCAGEM


def _mensagem_falha(erro: Exception) -> str:
    """Preserva o erro de domínio e revela a causa-raiz útil ao operador.

    A mensagem da causa-raiz é texto de terceiro que vai parar no console do
    operador: ela passa por censura de credencial e truncagem antes de sair.
    O tipo da causa é mantido porque é o que orienta o diagnóstico.
    """
    atual: BaseException = erro
    vistos: set[int] = set()
    proxima = atual.__cause__ or atual.__context__
    while proxima is not None and id(atual) not in vistos:
        vistos.add(id(atual))
        atual = proxima
        proxima = atual.__cause__ or atual.__context__
    principal = _resumir(str(erro)) or type(erro).__name__
    if atual is erro:
        return principal
    detalhe = _resumir(str(atual)) or "sem mensagem adicional"
    return f"{principal} Causa: {type(atual).__name__}: {detalhe}"


def criar_analisador_lote(
    provedor_extracao: ProvedorPerfilExtraido | None = None,
    provedor_classificacao: ProvedorClassificacao | None = None,
    *,
    caminho_banco: Path = CAMINHO_BANCO,
    caminho_checkpoints: Path = CAMINHO_CHECKPOINTS,
    relogio: Callable[[], date] | None = None,
) -> AnalisadorLote:
    """Composição live ou inteiramente injetável para testes offline."""
    if (provedor_extracao is None) != (provedor_classificacao is None):
        raise ErroConfiguracao(
            "informe juntos os provedores do Extractor e do Classifier"
        )
    if not caminho_banco.exists():
        raise ErroConfiguracao(
            "dados/radar.db não existe; execute primeiro "
            "python -m scripts.inicializar_base"
        )
    try:
        preparar_cache_analises(caminho_banco)
    except (sqlite3.DatabaseError, ValueError) as erro:
        raise ErroConfiguracao(str(erro)) from erro
    if provedor_extracao is None:
        load_dotenv(RAIZ_PROJETO / ".env")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ErroConfiguracao(
                "GOOGLE_API_KEY não está configurada no .env local"
            )
        limitador = InMemoryRateLimiter(
            requests_per_second=REQUISICOES_GEMINI_POR_SEGUNDO,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        chave_reserva = os.getenv("GROQ_API_KEY", "").strip()
        provedor_extracao = compor_com_reserva(
            ProvedorGeminiPerfilExtraido(api_key, limitador=limitador),
            ProvedorGroqPerfilExtraido,
            fronteira="extractor",
            chave_reserva=chave_reserva,
        )
        provedor_classificacao = compor_com_reserva(
            ProvedorGeminiClassificacao(api_key, limitador=limitador),
            ProvedorGroqClassificacao,
            fronteira="classifier",
            chave_reserva=chave_reserva,
        )
    assert provedor_classificacao is not None
    base = BaseStartups(caminho_banco)
    grafo, conexao = montar_grafo_lote(
        base,
        provedor_extracao,
        provedor_classificacao,
        caminho_checkpoints,
    )
    return AnalisadorLote(base, grafo, conexao, relogio)
