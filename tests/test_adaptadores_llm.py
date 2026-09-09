"""Adaptadores de LLM construídos offline, sem tocar em endpoint.

O que estes testes provam é a fronteira de cada provedor: método de saída
estruturada, schema do contrato certo, limitador compartilhado e devolução de
dicionário para o mesmo ``model_validate`` do nó. **Não** provam
compatibilidade real do endpoint com o schema — isso só o smoke ao vivo diz.

A primeira parte cobre os cinco adaptadores Groq (a reserva). A segunda cobre o
primário Gemini, cujo schema precisa perder as restrições que a API rejeita sem
afrouxar o contrato do lado de cá.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

import radar.provedores as provedores
from radar.configuracao import MODELO_GROQ, REQUISICOES_GROQ_POR_SEGUNDO
from radar.contratos import (
    BriefingRascunho,
    Classificacao,
    PerfilExtraido,
    PlanoConsulta,
)
from radar.provedores import (
    ProvedorGroqBriefingRascunho,
    ProvedorGroqClassificacao,
    ProvedorGroqPerfilExtraido,
    ProvedorGroqPlanoConsulta,
    ProvedorGroqRecomendacaoRascunho,
    RascunhosRecomendacao,
)
from tests.conftest import trecho_citado_falso


# ------------------------------------------------------------------------
# Os cinco adaptadores Groq (a reserva)
# ------------------------------------------------------------------------

CHAVE_FALSA = "groq-chave-de-teste-que-nunca-sai-daqui"

PLANO = {
    "filtros": {"setor": "Fintech"},
    "termos_busca": ["fintech"],
    "sinais_ia": [],
    "foco_analise": "uso de IA no produto vendido",
}
PERFIL = {
    "id_startup": 1,
    "resumo_produto": "A empresa vende um produto. O produto chega ao cliente.",
    "afirmacoes": [
        {
            "id_afirmacao": 1,
            "texto": "A empresa opera um modelo de linguagem próprio.",
            "categoria": "stack_propria",
            "polaridade": "neutro",
            "id_documento": 1,
            "trecho_citado": trecho_citado_falso(1),
            "sinais_tecnicos": [],
        }
    ],
}
CLASSIFICACAO = {
    "classe": "AI-enabled",
    "justificativa": (
        "O material público descreve o produto vendido. "
        "A evidência citada sustenta a classe atribuída."
    ),
    "ids_afirmacoes_suporte": [1],
}
RASCUNHOS = {
    "rascunhos": [
        {
            "tipo_fundamento": "gap_confirmado",
            "identificador_fundamento": "distribuicao",
            "tecnologias": ["NVIDIA Inception"],
            "justificativa_tecnica": "O programa abre acesso a suporte técnico.",
            "justificativa_negocio": "A validação encurta o ciclo de venda.",
            "proxima_acao": {
                "tipo_acao": "convite_inception",
                "detalhe": "Enviar o convite de admissão nesta semana.",
            },
            "ids_afirmacoes": [1],
            "ids_chunks": [101],
        }
    ]
}
BRIEFING = {
    "tese": {"texto": "A empresa tem lacuna de distribuição.", "ids_afirmacoes_suporte": [1]},
    "sintese_executiva": {
        "texto": "O material público sustenta a operação descrita.",
        "ids_afirmacoes_suporte": [1],
    },
    "pontos_de_conversa": [
        {"texto": "Perguntar sobre o canal atual.", "ids_afirmacoes_suporte": [1]},
        {"texto": "Explorar a expansão comercial.", "ids_afirmacoes_suporte": [1]},
    ],
}

FRONTEIRAS = [
    (ProvedorGroqPlanoConsulta, PlanoConsulta, PLANO),
    (ProvedorGroqPerfilExtraido, PerfilExtraido, PERFIL),
    (ProvedorGroqClassificacao, Classificacao, CLASSIFICACAO),
    (ProvedorGroqRecomendacaoRascunho, RascunhosRecomendacao, RASCUNHOS),
    (ProvedorGroqBriefingRascunho, BriefingRascunho, BRIEFING),
]


class RunnableFalso:
    def __init__(self, resposta):
        self.resposta = resposta

    def invoke(self, mensagens):
        return self.resposta


class ModeloFalso:
    """Substitui o ChatGroq: registra a configuração e nunca abre conexão."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        self.method = method
        return RunnableFalso(RESPOSTA_POR_TITULO[schema["title"]])


RESPOSTA_POR_TITULO = {
    contrato.__name__: exemplo for _, contrato, exemplo in FRONTEIRAS
}


@pytest.fixture
def groq_falso(monkeypatch):
    construidos: list[ModeloFalso] = []

    def fabrica(**kwargs):
        modelo = ModeloFalso(**kwargs)
        construidos.append(modelo)
        return modelo

    monkeypatch.setattr("langchain_groq.ChatGroq", fabrica)
    return construidos


@pytest.mark.parametrize(
    "adaptador,contrato,exemplo", FRONTEIRAS, ids=lambda item: getattr(item, "__name__", "")
)
def test_cada_adaptador_e_construido_sem_rede_e_devolve_dicionario(
    groq_falso, adaptador, contrato, exemplo
):
    provedor = adaptador(CHAVE_FALSA)
    modelo = groq_falso[0]

    assert modelo.method == "json_schema"
    assert modelo.schema["title"] == contrato.__name__
    assert modelo.kwargs["model"] == MODELO_GROQ

    devolvido = provedor.invocar([("system", "instrução")])

    assert isinstance(devolvido, dict)
    assert contrato.model_validate(devolvido)


@pytest.mark.parametrize(
    "adaptador,contrato,_", FRONTEIRAS, ids=lambda item: getattr(item, "__name__", "")
)
def test_o_schema_oferecido_nao_afrouxa_extra_forbid(groq_falso, adaptador, contrato, _):
    adaptador(CHAVE_FALSA)

    assert groq_falso[0].schema["additionalProperties"] is False


def test_o_schema_do_lote_de_recomendacao_preserva_aninhado_enum_e_uniao(groq_falso):
    ProvedorGroqRecomendacaoRascunho(CHAVE_FALSA)
    schema = groq_falso[0].schema
    texto = repr(schema)

    assert "$defs" in schema
    assert "enum" in texto
    assert "anyOf" in texto or "oneOf" in texto


def test_os_cinco_adaptadores_compartilham_o_mesmo_limitador(groq_falso):
    for adaptador, _, _exemplo in FRONTEIRAS:
        adaptador(CHAVE_FALSA)

    limitadores = {id(modelo.kwargs["rate_limiter"]) for modelo in groq_falso}

    assert len(groq_falso) == 5
    assert len(limitadores) == 1


def test_a_taxa_da_reserva_e_conservadora():
    assert REQUISICOES_GROQ_POR_SEGUNDO == 0.05


def test_a_chave_nunca_aparece_em_log(groq_falso, caplog):
    with caplog.at_level(logging.DEBUG):
        for adaptador, _, _exemplo in FRONTEIRAS:
            adaptador(CHAVE_FALSA)

    assert CHAVE_FALSA not in caplog.text


# ------------------------------------------------------------------------
# O primário Gemini e o schema que a API aceita
# ------------------------------------------------------------------------

def _chaves_de_schema(valor: object) -> set[str]:
    chaves: set[str] = set()
    if isinstance(valor, dict):
        for chave, item in valor.items():
            chaves.add(chave)
            if chave in {"properties", "$defs"} and isinstance(item, dict):
                for subschema in item.values():
                    chaves.update(_chaves_de_schema(subschema))
            else:
                chaves.update(_chaves_de_schema(item))
    elif isinstance(valor, list):
        for item in valor:
            chaves.update(_chaves_de_schema(item))
    return chaves


def test_schema_do_extractor_remove_restricoes_rejeitadas_pelo_gemini():
    original = PerfilExtraido.model_json_schema()
    compativel = provedores._schema_json_compativel_gemini(PerfilExtraido)

    assert {"minLength", "maxLength"} <= _chaves_de_schema(original)
    assert _chaves_de_schema(compativel) <= provedores._CHAVES_JSON_SCHEMA_GEMINI
    assert compativel["required"] == [
        "id_startup",
        "resumo_produto",
        "afirmacoes",
    ]
    assert set(compativel["properties"]) == {
        "id_startup",
        "resumo_produto",
        "afirmacoes",
    }
    assert compativel["properties"]["afirmacoes"]["type"] == "array"
    assert "minLength" not in json.dumps(compativel)
    assert "maxLength" not in json.dumps(compativel)
    assert "additionalProperties" not in json.dumps(compativel)
    assert "minimum" not in json.dumps(compativel)
    assert "minItems" not in json.dumps(compativel)
    assert "maxItems" not in json.dumps(compativel)
    afirmacao = compativel["properties"]["afirmacoes"]["items"]
    assert afirmacao["properties"]["categoria"]["enum"][-1] == "outro"
    assert "$defs" not in json.dumps(compativel)
    assert "$ref" not in json.dumps(compativel)
    afirmacao = compativel["properties"]["afirmacoes"]["items"]
    assert "id_afirmacao" in afirmacao["properties"]


def test_provedor_envia_schema_compativel_sem_fazer_chamada_de_rede(monkeypatch):
    observado: dict[str, Any] = {}

    class ExecutavelStub:
        def invoke(self, mensagens: object) -> object:
            observado["mensagens"] = mensagens
            return {"ok": True}

    class ChatStub:
        def __init__(self, **kwargs: object):
            observado["configuracao"] = kwargs

        def with_structured_output(
            self, schema: dict[str, Any], *, method: str
        ) -> ExecutavelStub:
            observado["schema"] = schema
            observado["method"] = method
            return ExecutavelStub()

    monkeypatch.setattr(provedores, "ChatGoogleGenerativeAI", ChatStub)

    limitador = object()
    provedor = provedores.ProvedorGeminiPerfilExtraido(
        "chave-de-teste", limitador=limitador
    )
    resposta = provedor.invocar([("human", "conteudo de teste")])

    assert resposta == {"ok": True}
    assert observado["method"] == "json_schema"
    assert _chaves_de_schema(observado["schema"]) <= (
        provedores._CHAVES_JSON_SCHEMA_GEMINI
    )
    assert observado["mensagens"] == [("human", "conteudo de teste")]
    assert observado["configuracao"]["rate_limiter"] is limitador
