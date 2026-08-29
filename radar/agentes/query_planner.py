from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from radar.base_startups import BaseStartups
from radar.contratos import EstadoRadar, PlanoConsulta, ResultadoRecuperacao
from radar.provedores import ProvedorPlanoConsulta


class ErroQueryPlanner(RuntimeError):
    """Falha segura: não inclui credencial nem inventa um plano alternativo."""


class QueryPlanner:
    def __init__(self, base: BaseStartups, provedor: ProvedorPlanoConsulta):
        self.base = base
        self.provedor = provedor

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        plano_existente = estado.get("plano_consulta")
        resultado_existente = estado.get("resultado_recuperacao")
        if plano_existente is not None:
            plano = PlanoConsulta.model_validate(plano_existente)
            if resultado_existente is not None:
                resultado = ResultadoRecuperacao.model_validate(resultado_existente)
                if not resultado.empresas:
                    return self._relaxar(plano, estado)
            # No aprofundamento o plano é reutilizado, sem nova chamada ao LLM.
            return {"trajeto": ["query_planner"]}

        consulta = estado.get("consulta_usuario", "").strip()
        if not consulta:
            raise ErroQueryPlanner("A consulta do usuário está vazia.")
        plano = self._gerar_com_validacao(consulta)
        return {"plano_consulta": plano, "trajeto": ["query_planner"]}

    def _gerar_com_validacao(self, consulta: str) -> PlanoConsulta:
        vocabularios = self.base.vocabularios()
        instrucao = (
            "Você é o Query Planner do NVIDIA Startup AI Radar. Converta a consulta "
            "em um PlanoConsulta estritamente estruturado. Use somente valores exatos "
            "dos vocabulários fornecidos; quando não houver correspondência exata, use null. "
            "Não infira classe analisada a partir de classe_referencia. termos_busca deve ter "
            "de 1 a 8 expressões úteis para busca lexical; sinais_ia registra separadamente "
            "sinais técnicos de IA explicitamente pedidos. A resposta deve estar em português.\n"
            f"Vocabulários disponíveis: {json.dumps(vocabularios, ensure_ascii=False)}"
        )
        erro_anterior: str | None = None
        for tentativa in range(2):
            mensagens = [("system", instrucao), ("human", consulta)]
            if erro_anterior:
                mensagens.append(
                    (
                        "system",
                        "A resposta anterior violou o contrato. Corrija sem texto livre. "
                        f"Falha de validação: {erro_anterior}",
                    )
                )
            try:
                bruto = self.provedor.invocar(mensagens)
            except Exception as exc:
                raise ErroQueryPlanner(
                    "O Gemini não respondeu ao Query Planner; nenhum resultado foi fabricado."
                ) from exc
            try:
                plano = PlanoConsulta.model_validate(bruto)
                self._validar_vocabulario(plano, vocabularios)
                return plano
            except (ValidationError, ValueError) as exc:
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroQueryPlanner(
                        "O Gemini respondeu duas vezes fora do contrato estruturado; "
                        "a execução foi interrompida sem resultados."
                    ) from exc
        raise AssertionError("laço de validação terminou em estado impossível")

    @staticmethod
    def _resumir_erro(erro: Exception) -> str:
        if isinstance(erro, ValidationError):
            campos = [".".join(str(item) for item in falha["loc"]) for falha in erro.errors()]
            return "campos inválidos: " + ", ".join(campos)
        return str(erro)

    @staticmethod
    def _validar_vocabulario(
        plano: PlanoConsulta, vocabularios: dict[str, list[str]]
    ) -> None:
        filtros = plano.filtros
        escalares = {
            "setor": filtros.setor,
            "localizacao": filtros.localizacao,
        }
        listas = {
            "estagio": filtros.estagio,
            "tamanho_time": filtros.tamanho_time,
            "classe_analisada": filtros.classe_analisada,
        }
        for campo, valor in escalares.items():
            if valor is not None and valor not in vocabularios[campo]:
                raise ValueError(f"{campo} fora do vocabulário controlado")
        for campo, valores in listas.items():
            if valores and any(valor not in vocabularios[campo] for valor in valores):
                raise ValueError(f"{campo} fora do vocabulário controlado")

    @staticmethod
    def _relaxar(plano: PlanoConsulta, estado: EstadoRadar) -> dict[str, Any]:
        tentativa = int(estado.get("tentativas_relaxamento", 0))
        copia = plano.model_copy(deep=True)
        criterios: list[str] = []
        if tentativa == 0:
            for campo in ("estagio", "localizacao", "tamanho_time"):
                if getattr(copia.filtros, campo) is not None:
                    criterios.append(campo)
                setattr(copia.filtros, campo, None)
        elif tentativa == 1:
            if copia.filtros.setor is not None:
                criterios.append("setor")
            copia.filtros.setor = None
        else:
            # R1 impede uma terceira reentrada; esta proteção evita relaxamento silencioso.
            return {"trajeto": ["query_planner"]}
        return {
            "plano_consulta": copia,
            "tentativas_relaxamento": tentativa + 1,
            "criterios_relaxados": criterios,
            "trajeto": ["query_planner"],
        }
