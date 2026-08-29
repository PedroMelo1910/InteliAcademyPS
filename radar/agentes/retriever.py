from __future__ import annotations

from typing import Any

from radar.base_startups import BaseStartups
from radar.contratos import EstadoRadar, PlanoConsulta


class Retriever:
    def __init__(self, base: BaseStartups):
        self.base = base

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        plano = PlanoConsulta.model_validate(estado["plano_consulta"])
        resultado = self.base.recuperar(plano, estado.get("startup_selecionada"))
        return {"resultado_recuperacao": resultado, "trajeto": ["retriever"]}

