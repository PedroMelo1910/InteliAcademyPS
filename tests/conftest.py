"""Fornece fixtures globais e impede chamadas de rede na suíte offline."""

import hashlib
import asyncio
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

from radar.base_startups import BaseStartups, inicializar_banco
from radar.configuracao import CAMINHO_DADOS_CURADOS


@pytest.fixture(autouse=True)
def bloquear_rede(monkeypatch):
    """Nenhum teste automatizado pode acessar a rede; falha alta se tentar."""

    def negar(*_args, **_kwargs):
        raise RuntimeError(
            "teste tentou abrir uma conexão de rede; a suíte é estritamente offline"
        )

    monkeypatch.setattr(socket.socket, "connect", negar)
    monkeypatch.setattr(socket.socket, "connect_ex", negar)
    monkeypatch.setattr(socket, "create_connection", negar)
    monkeypatch.setattr(socket, "getaddrinfo", negar)
    monkeypatch.setattr(socket, "gethostbyname", negar)
    monkeypatch.setattr(socket, "gethostbyname_ex", negar)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", negar)
    monkeypatch.setattr(subprocess, "Popen", negar)
    monkeypatch.setattr(os, "system", negar)


@pytest.fixture
def caminho_banco(tmp_path: Path) -> Path:
    caminho = tmp_path / "radar_teste.db"
    inicializar_banco(caminho, CAMINHO_DADOS_CURADOS)
    return caminho


@pytest.fixture
def base(caminho_banco: Path) -> BaseStartups:
    return BaseStartups(caminho_banco)


def _tokens(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.casefold())


class EmbeddingFalso:
    """EmbeddingProvider determinístico: saco de tokens por hashing, sem rede.

    Textos que compartilham tokens compartilham componentes do vetor, o que
    dá similaridade de cosseno controlável nos testes.
    """

    def __init__(self, dimensao: int = 32, modelo: str | None = None):
        self._dimensao = dimensao
        self._modelo = modelo or f"embedding-falso-{dimensao}"
        self.chamadas_passagens = 0
        self.chamadas_consulta = 0
        self.textos_embedados: list[str] = []

    @property
    def dimensao(self) -> int:
        return self._dimensao

    @property
    def modelo(self) -> str:
        return self._modelo

    def _vetor(self, texto: str) -> list[float]:
        vetor = [0.0] * self._dimensao
        for token in _tokens(texto):
            digerido = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vetor[int(digerido, 16) % self._dimensao] += 1.0
        if not any(vetor):
            vetor[0] = 1.0  # evita vetor nulo, cujo cosseno é indefinido
        return vetor

    def embutir_passagens(self, textos: list[str]) -> list[list[float]]:
        self.chamadas_passagens += 1
        self.textos_embedados.extend(textos)
        return [self._vetor(texto) for texto in textos]

    def embutir_consulta(self, texto: str) -> list[float]:
        self.chamadas_consulta += 1
        return self._vetor(texto)


class RerankFalso:
    """RerankProvider determinístico: sobreposição de tokens com a consulta."""

    def __init__(self):
        self.chamadas = 0
        self.ultimo_lote: list[str] | None = None

    def reordenar(self, consulta: str, textos: list[str]) -> list[float]:
        self.chamadas += 1
        self.ultimo_lote = list(textos)
        termos = set(_tokens(consulta))
        return [
            len(termos & set(_tokens(texto))) / (len(termos) or 1)
            for texto in textos
        ]


class ConsultorNvidiaFalso:
    """ProvedorContextoNvidia determinístico: nenhuma rede, consulta registrada."""

    def __init__(self, contexto=None, erro: Exception | None = None):
        self._contexto = contexto if contexto is not None else contexto_nvidia_falso()
        self._erro = erro
        self.chamadas = 0
        self.consultas: list[str] = []

    def consultar(self, consulta: str):
        self.chamadas += 1
        self.consultas.append(consulta)
        if self._erro is not None:
            raise self._erro
        return self._contexto


def trecho_nvidia_falso(
    id_chunk: int,
    *,
    tecnologia: str | None = "NVIDIA NIM",
    topico: str | None = None,
    score_rerank: float = 0.9,
):
    """Um ``TrechoNvidia`` coerente: origem e tecnologia sempre combinam."""
    from radar.contratos import TrechoNvidia

    origem = "tecnologia" if tecnologia is not None else "conceitual"
    rotulo = topico or (tecnologia or "ai-native-services")
    return TrechoNvidia(
        id_chunk=id_chunk,
        topico=rotulo,
        origem=origem,
        tecnologia=tecnologia,
        breadcrumb=f"{rotulo} > seção {id_chunk}",
        texto=(
            f"Trecho {id_chunk} da base NVIDIA sobre {rotulo}, com inferência, "
            "latência e custo em produção."
        ),
        fonte_url=f"https://nvidia.example/{id_chunk}",
        score_rerank=score_rerank,
    )


def contexto_nvidia_falso(consulta: str = "consulta NVIDIA de teste"):
    """Contexto rico: 7 chunks de tecnologia e 1 conceitual, no teto da faixa 5–8.

    É deliberadamente farto para que as fixtures do caminho feliz tenham
    lastro real de tecnologia. Quando a **escassez** é o comportamento sob
    teste, use ``contexto_nvidia_escasso`` — este aqui cobre os seis gaps do
    contrato e não consegue exercitar interseção vazia.
    """
    from radar.contratos import ContextoNvidia

    trechos = [
        trecho_nvidia_falso(101, tecnologia="NVIDIA NIM", score_rerank=0.95),
        trecho_nvidia_falso(102, tecnologia="NVIDIA Triton Inference Server", score_rerank=0.9),
        trecho_nvidia_falso(103, tecnologia="TensorRT-LLM", score_rerank=0.85),
        trecho_nvidia_falso(104, tecnologia="NeMo Guardrails", score_rerank=0.8),
        trecho_nvidia_falso(105, tecnologia="NVIDIA RAPIDS", score_rerank=0.75),
        trecho_nvidia_falso(107, tecnologia="NVIDIA Inception", score_rerank=0.72),
        trecho_nvidia_falso(108, tecnologia="NVIDIA Riva", score_rerank=0.71),
        trecho_nvidia_falso(
            106, tecnologia=None, topico="ai-native-services", score_rerank=0.7
        ),
    ]
    return ContextoNvidia(consulta_gerada=consulta, trechos=trechos)


def contexto_nvidia_escasso(consulta: str = "consulta NVIDIA escassa"):
    """Contexto pobre de propósito: sem NVIDIA Morpheus e sem AI Enterprise.

    Existe para exercitar o ramo em que um fundamento sustentado por evidência
    não encontra **nenhuma** tecnologia candidata entre os trechos recuperados.
    O sinal ``ciberseguranca_em_escala`` tem exatamente essas duas candidatas,
    então a interseção com este contexto é vazia por construção.
    """
    from radar.contratos import ContextoNvidia

    trechos = [
        trecho_nvidia_falso(101, tecnologia="NVIDIA NIM", score_rerank=0.95),
        trecho_nvidia_falso(
            102, tecnologia="NVIDIA Triton Inference Server", score_rerank=0.9
        ),
        trecho_nvidia_falso(103, tecnologia="TensorRT-LLM", score_rerank=0.85),
        trecho_nvidia_falso(105, tecnologia="NVIDIA RAPIDS", score_rerank=0.8),
        trecho_nvidia_falso(
            106, tecnologia=None, topico="ai-native-services", score_rerank=0.7
        ),
    ]
    return ContextoNvidia(consulta_gerada=consulta, trechos=trechos)


DIMENSOES_ESTRUTURAIS = (
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
)


def trecho_citado_falso(id_afirmacao: int) -> str:
    """O trecho canônico da afirmação: recomendação e perfil citam o mesmo."""
    return f"Trecho público verificável para a evidência {id_afirmacao} citada."


def afirmacao_validada_falsa(
    id_afirmacao: int,
    categoria: str,
    *,
    polaridade: str | None = None,
    situacao: str = "confirmada",
    id_documento: int | None = None,
    texto: str | None = None,
    motivo: str | None = None,
    sinais_tecnicos=(),
):
    """``AfirmacaoValidada`` mínima e coerente com as regras de polaridade."""
    from radar.contratos import AfirmacaoValidada

    if polaridade is None:
        polaridade = "neutro" if categoria not in DIMENSOES_ESTRUTURAIS else "presenca"
    return AfirmacaoValidada(
        id_afirmacao=id_afirmacao,
        texto=texto or f"A evidência número {id_afirmacao} está documentada.",
        categoria=categoria,
        polaridade=polaridade,
        id_documento=id_documento if id_documento is not None else id_afirmacao,
        trecho_citado=trecho_citado_falso(id_afirmacao),
        sinais_tecnicos=tuple(sinais_tecnicos),
        situacao=situacao,
        motivo=(
            None
            if situacao == "confirmada"
            else (motivo or "Trecho não ocorre na fonte.")
        ),
    )


def perfil_validado_falso(itens, hosts: list[str] | None = None):
    """Monta o ``PerfilValidado`` derivando dimensões e taxa das afirmações."""
    from radar.contratos import EstadoDimensaoGap, PerfilValidado

    estados = []
    for dimensao in DIMENSOES_ESTRUTURAIS:
        presencas = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "presenca"
        )
        ausencias = sorted(
            item.id_afirmacao
            for item in itens
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == "ausencia_explicita"
        )
        if presencas and ausencias:
            estado, ids = "desconhecido", sorted(presencas + ausencias)
        elif presencas:
            estado, ids = "capacidade_confirmada", presencas
        elif ausencias:
            estado, ids = "gap_confirmado", ausencias
        else:
            estado, ids = "desconhecido", []
        estados.append(
            EstadoDimensaoGap(dimensao=dimensao, estado=estado, ids_evidencias=ids)
        )
    derrubadas = sum(1 for item in itens if item.situacao == "derrubada")
    return PerfilValidado(
        afirmacoes_validadas=itens,
        taxa_derrubada=derrubadas / len(itens),
        hosts_distintos=sorted(hosts if hosts is not None else ["fonte-a.example"]),
        estado_dimensoes_gap=estados,
    )


# ----------------------------------------------------------------------
# Dublês de provedor de structured output
#
# Três dublês, três formas de asserção — deliberadamente distintos:
#
# * ``ProvedorFixo``            — uma resposta só, repetida em toda chamada.
# * ``ProvedorSequencial``      — ``chamadas`` é a **lista de mensagens**, e
#                                 ``ultimo_prompt`` junta o texto da última.
# * ``ProvedorSequencialFalso`` — ``chamadas`` é um **contador**, as mensagens
#                                 ficam em ``mensagens`` e o esgotamento vira
#                                 ``AssertionError`` em vez de ``IndexError``.
#
# Quem assere ``len(provedor.chamadas) == 2`` usa o segundo; quem assere
# ``provedor.chamadas == 2`` usa o terceiro.
# ----------------------------------------------------------------------


class ProvedorFixo:
    """Sempre a mesma resposta; um item ``Exception`` é levantado.

    A guarda de ``Exception`` existe para que um provedor injetado como "este
    nó não pode ser chamado" exploda no ponto da chamada, em vez de devolver a
    exceção como valor e falhar depois, com diagnóstico pior.
    """

    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        if isinstance(self.resposta, Exception):
            raise self.resposta
        return self.resposta


class ProvedorSequencial:
    """Uma resposta programada por chamada; ``chamadas`` guarda as mensagens."""

    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas.append(mensagens)
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    @property
    def ultimo_prompt(self) -> str:
        return "\n".join(texto for _, texto in self.chamadas[-1])


class ProvedorSequencialFalso:
    """Provedor de structured output com uma resposta programada por chamada.

    Um item ``Exception`` é levantado em vez de devolvido, o que permite testar
    falha de provedor e falha de contrato com o mesmo fake.
    """

    def __init__(self, *respostas):
        self._respostas = list(respostas)
        self.chamadas = 0
        self.mensagens: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        if not self._respostas:
            raise AssertionError(
                f"o provedor foi chamado {self.chamadas} vezes, além das respostas "
                "programadas"
            )
        resposta = self._respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


# ----------------------------------------------------------------------
# Apoio do marco Briefing
# ----------------------------------------------------------------------


def citacao_nvidia_falsa(id_chunk: int = 101, **ajustes):
    """``CitacaoNvidia`` idêntica ao chunk correspondente do contexto falso."""
    from radar.contratos import CitacaoNvidia

    trecho = trecho_nvidia_falso(id_chunk)
    campos = {
        "id_chunk": trecho.id_chunk,
        "topico": trecho.topico,
        "origem": trecho.origem,
        "tecnologia": trecho.tecnologia,
        "fonte_url": trecho.fonte_url,
        "breadcrumb": trecho.breadcrumb,
    }
    campos.update(ajustes)
    return CitacaoNvidia(**campos)


def recomendacao_falsa(
    gap: str = "distribuicao",
    *,
    tecnologias: list[str] | None = None,
    id_afirmacao: int = 1,
    id_documento: int = 1,
    id_chunk: int = 101,
    url_fonte: str = "https://fonte-a.example/materia",
    trecho_citado: str | None = None,
    citacao=None,
):
    """``Recomendacao`` completa e coerente com o perfil e o contexto falsos.

    O trecho citado é o da afirmação referenciada e a citação NVIDIA reproduz o
    chunk referenciado — é assim que o nó Recommendation real a constrói.
    """
    from radar.contratos import EvidenciaStartup, ProximaAcao, Recomendacao

    from radar.contratos import GAPS_ENDERECAVEIS

    return Recomendacao(
        tipo_fundamento=(
            "gap_confirmado" if gap in GAPS_ENDERECAVEIS else "oportunidade_confirmada"
        ),
        identificador_fundamento=gap,
        # o padrão acompanha o chunk 101 para a citação ter lastro real
        tecnologias=tecnologias or ["NVIDIA NIM"],
        justificativa_tecnica="O programa abre acesso a suporte técnico dedicado.",
        justificativa_negocio="A validação encurta o ciclo de venda enterprise.",
        prioridade="media",
        complexidade="baixa",
        proxima_acao=ProximaAcao(
            tipo_acao="convite_inception",
            detalhe="Enviar o convite de admissão ao programa nesta semana.",
        ),
        evidencias_startup=[
            EvidenciaStartup(
                id_afirmacao=id_afirmacao,
                id_documento=id_documento,
                url_fonte=url_fonte,
                trecho_citado=trecho_citado or trecho_citado_falso(id_afirmacao),
            )
        ],
        citacoes_nvidia=[
            citacao if citacao is not None else citacao_nvidia_falsa(id_chunk)
        ],
    )


def cabecalho_falso(**ajustes):
    from datetime import date

    from radar.contratos import CabecalhoBriefing

    campos = {
        "nome": "Caju",
        "site": "https://www.caju.com.br/",
        "setor": "Fintech / RH",
        "estagio": "série B",
        "localizacao": "São Paulo, SP",
        "data_geracao": date(2026, 9, 3),
        "consulta_original": "fintech brasileira de benefícios com cartão",
    }
    campos.update(ajustes)
    return CabecalhoBriefing(**campos)


def rodape_falso(**ajustes):
    from datetime import date

    from radar.contratos import RodapeBriefing

    campos = {
        "versao_rubrica": "rubrica-v1",
        "data_execucao": date(2026, 9, 3),
        "afirmacoes_confirmadas": 2,
        "afirmacoes_derrubadas": 0,
        "trajeto": ["extractor", "classifier", "evidence_validator", "briefing"],
        "rota_r3": "prosseguir",
    }
    campos.update(ajustes)
    return RodapeBriefing(**campos)


def fonte_falsa(
    url: str = "https://fonte-a.example/materia",
    *,
    tipo: str = "notícia",
    titulo: str = "Matéria pública sobre a startup",
    data_publicacao=None,
):
    from radar.contratos import FonteBriefing, normalizar_dominio
    from urllib.parse import urlparse

    return FonteBriefing(
        url_fonte=url,
        host_normalizado=normalizar_dominio(urlparse(url).hostname or ""),
        tipo=tipo,
        titulo=titulo,
        data_publicacao=data_publicacao,
    )


def briefing_normal_falso(**ajustes):
    """``Briefing`` normal mínimo e válido, para exercitar o contrato."""
    from radar.contratos import Briefing, ConclusaoAncorada, VereditoBriefing

    campos = {
        "variante": "normal",
        "cabecalho": cabecalho_falso(),
        "veredito": VereditoBriefing(
            classe="AI-enabled",
            fit_score_total=61,
            tese="A empresa usa IA no produto e tem lacuna de distribuição.",
            ids_afirmacoes_suporte=[1],
        ),
        "sintese_executiva": ConclusaoAncorada(
            texto="A startup opera benefícios corporativos com apoio de modelos de terceiros.",
            ids_afirmacoes_suporte=[1, 2],
        ),
        "pontos_de_conversa": [
            ConclusaoAncorada(
                texto="Perguntar como o time mede custo por inferência.",
                ids_afirmacoes_suporte=[2],
            ),
            ConclusaoAncorada(
                texto="Explorar o canal de distribuição atual.",
                ids_afirmacoes_suporte=[1],
            ),
        ],
        "recomendacoes": [recomendacao_falsa()],
        "fontes": [fonte_falsa()],
        "avisos": [],
        "rodape": rodape_falso(),
    }
    campos.update(ajustes)
    return Briefing(**campos)


def briefing_nao_aderente_falso(**ajustes):
    """``Briefing`` da variante non-AI: score zero real e nenhuma recomendação."""
    from radar.contratos import ConclusaoAncorada, VereditoBriefing

    campos = {
        "variante": "nao_aderente",
        "veredito": VereditoBriefing(
            classe="non-AI",
            fit_score_total=0,
            tese="A base pública não descreve uso de IA no produto vendido.",
            ids_afirmacoes_suporte=[1],
        ),
        "pontos_de_conversa": [
            ConclusaoAncorada(
                texto="Confirmar se há projeto de IA fora do material público.",
                ids_afirmacoes_suporte=[1],
            )
        ],
        "recomendacoes": [],
        "avisos": [
            "Veredito non-AI: a stack NVIDIA não é recomendada para esta empresa."
        ],
        "rodape": rodape_falso(rota_r3="nao_aderente"),
    }
    campos.update(ajustes)
    return briefing_normal_falso(**campos)


def briefing_insuficiente_falso(**ajustes):
    """``Briefing`` sem classe, sem score, sem pontos e sem fontes (§11.3)."""
    from radar.contratos import ConclusaoAncorada, VereditoBriefing

    campos = {
        "variante": "evidencia_insuficiente",
        "veredito": VereditoBriefing(
            classe=None,
            fit_score_total=None,
            tese="A base disponível não sustenta uma conclusão sobre a empresa.",
            ids_afirmacoes_suporte=[],
        ),
        "sintese_executiva": ConclusaoAncorada(
            texto="Nenhuma afirmação sobreviveu à conferência de proveniência.",
            ids_afirmacoes_suporte=[],
        ),
        "pontos_de_conversa": [],
        "recomendacoes": [],
        "fontes": [],
        "avisos": ["Evidência insuficiente: nenhuma afirmação confirmada."],
        "rodape": rodape_falso(
            afirmacoes_confirmadas=0,
            afirmacoes_derrubadas=2,
            rota_r3="evidencia_insuficiente",
        ),
    }
    campos.update(ajustes)
    return briefing_normal_falso(**campos)


# --------------------------------------------------------------------------
# Falha segura na fronteira do agente: mensagem sem nome de provedor
# --------------------------------------------------------------------------

# Corpo cru de terceiro, com chave embutida, injetado nas falhas operacionais
# dos testes de mensagem neutra. Nada daqui pode alcançar o texto final.
CORPO_BRUTO_DO_PROVEDOR = (
    '{"error":{"status":"UNAVAILABLE","message":"backend sobrecarregado",'
    '"apiKey":"AIzaSyD-CHAVE-FALSA-QUE-NAO-PODE-VAZAR"}}'
)

# Chave, corpo cru e instrução de sistema: o que a falha segura nunca repete.
TERMOS_QUE_A_FALHA_SEGURA_NAO_REPETE = (
    "AIzaSyD-CHAVE-FALSA-QUE-NAO-PODE-VAZAR",
    "apiKey",
    "UNAVAILABLE",
    "backend sobrecarregado",
    "Você é o",
)


def exigir_falha_neutra_de_provedor(mensagem: str) -> None:
    """O nó não observa qual provedor respondeu, logo não pode nomear nenhum.

    Com a reserva Groq ativa, a resposta final pode ter vindo do primário ou da
    reserva. Dizer "O Gemini" na fronteira do agente afirmaria um fato que o nó
    não tem como verificar, e ficaria factualmente errado toda vez que a
    reserva atendesse. Nomear a reserva teria o mesmo defeito. A composição,
    os adaptadores e o log da fronteira de fallback seguem nomeando os
    provedores, porque lá a identidade é observada de fato.
    """
    assert "O provedor de IA" in mensagem, mensagem
    for nome in ("Gemini", "Groq"):
        assert nome not in mensagem, (
            f"a fronteira do agente nomeou {nome} sem poder saber disso: {mensagem}"
        )
    for termo in TERMOS_QUE_A_FALHA_SEGURA_NAO_REPETE:
        assert termo not in mensagem, (
            f"a falha segura repetiu conteúdo que não pode sair do nó: {termo}"
        )
