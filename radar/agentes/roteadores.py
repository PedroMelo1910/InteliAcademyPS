from radar.configuracao import TETO_RELAXAMENTO
from radar.contratos import EstadoRadar, ResultadoR1, ResultadoRecuperacao


def rotear_r1(estado: EstadoRadar) -> ResultadoR1:
    """R1 puro; a ordem dos quatro predicados é parte do contrato."""
    if estado.get("startup_selecionada") is not None:
        return "analisar"
    resultado = ResultadoRecuperacao.model_validate(estado["resultado_recuperacao"])
    if resultado.empresas:
        return "candidatas_prontas"
    if int(estado.get("tentativas_relaxamento", 0)) < TETO_RELAXAMENTO:
        return "relaxar"
    return "sem_resultado"

