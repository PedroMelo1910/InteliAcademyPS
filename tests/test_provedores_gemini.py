from __future__ import annotations

import json
from typing import Any

import radar.provedores as provedores
from radar.contratos import PerfilExtraido


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
