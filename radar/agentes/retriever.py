from __future__ import annotations

from typing import Any

from radar.base_startups import BaseStartups
from radar.contratos import EstadoRadar, PlanoConsulta


# Toda a análise descende do conjunto recuperado. Quando o Retriever substitui
# esse conjunto, o que foi derivado do anterior deixa de ter lastro — inclusive
# num thread retomado, em que o checkpoint traria a análise de outra busca.
CAMPOS_DERIVADOS_DA_RECUPERACAO: tuple[str, ...] = (
    "perfil_extraido",
    "classificacao",
    "perfil_validado",
    "confianca_perfil",
    "contexto_nvidia",
    "recomendacoes",
    "fit_score",
)


class Retriever:
    def __init__(self, base: BaseStartups):
        self.base = base

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        plano = PlanoConsulta.model_validate(estado["plano_consulta"])
        resultado = self.base.recuperar(plano, estado.get("startup_selecionada"))
        saida: dict[str, Any] = {
            "resultado_recuperacao": resultado,
            "tentativas_extracao": 0,
            "trajeto": ["retriever"],
        }
        # ``trajeto``, ``erros`` e ``criterios_relaxados`` são histórico
        # acumulado e permanecem intactos.
        for campo in CAMPOS_DERIVADOS_DA_RECUPERACAO:
            saida[campo] = None
        return saida
