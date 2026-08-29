from __future__ import annotations

from typing import Protocol

from langchain_google_genai import ChatGoogleGenerativeAI

from radar.configuracao import MODELO_GEMINI
from radar.contratos import PlanoConsulta


class ProvedorPlanoConsulta(Protocol):
    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        """Produz uma resposta estruturada ainda sujeita à validação da fronteira."""


class ProvedorGeminiPlanoConsulta:
    """Único contato de rede do fluxo atual."""

    def __init__(self, api_key: str):
        modelo = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            api_key=api_key,
            temperature=None,
            retries=1,
            request_timeout=30,
        )
        self._estruturado = modelo.with_structured_output(
            PlanoConsulta, method="json_schema"
        )

    def invocar(self, mensagens: list[tuple[str, str]]) -> object:
        return self._estruturado.invoke(mensagens)
