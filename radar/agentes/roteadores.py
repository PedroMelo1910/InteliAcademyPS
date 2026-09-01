from radar.configuracao import LIMIAR_DERRUBADA, MAX_EXTRACOES, TETO_RELAXAMENTO
from radar.contratos import (
    Classificacao,
    EstadoRadar,
    PerfilValidado,
    ResultadoR1,
    ResultadoR2,
    ResultadoR3,
    ResultadoRecuperacao,
)


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


def _evidencia_e_classificacao(
    estado: EstadoRadar,
) -> tuple[PerfilValidado, Classificacao]:
    perfil = PerfilValidado.model_validate(estado["perfil_validado"])
    classificacao = Classificacao.model_validate(estado["classificacao"])
    return perfil, classificacao


def _ids_confirmados(perfil: PerfilValidado) -> set[int]:
    return {
        afirmacao.id_afirmacao
        for afirmacao in perfil.afirmacoes_validadas
        if afirmacao.situacao == "confirmada"
    }


def precisa_reextrair(
    perfil: PerfilValidado, classificacao: Classificacao
) -> bool:
    """Predicado único de reextração, sem teto e sem estado.

    R2 usa este predicado para decidir a rota; o Extractor usa o mesmo para
    decidir o modo estrito. Manter duas cópias sutilmente diferentes faria o
    Extractor reextrair com o prompt idêntico, gastando uma chamada de LLM
    para reproduzir a evidência que já havia sido rejeitada.
    """
    if perfil.taxa_derrubada >= LIMIAR_DERRUBADA:
        return True
    return not set(classificacao.ids_afirmacoes_suporte).issubset(
        _ids_confirmados(perfil)
    )


def rotear_r2(estado: EstadoRadar) -> ResultadoR2:
    """R2 puro: reextrai evidência ruim uma única vez, respeitando o teto."""
    perfil, classificacao = _evidencia_e_classificacao(estado)
    if precisa_reextrair(perfil, classificacao) and (
        int(estado.get("tentativas_extracao", 0)) < MAX_EXTRACOES
    ):
        return "reextrair"
    return "evidencia_pronta"


def rotear_r3(estado: EstadoRadar) -> ResultadoR3:
    """R3 puro; insuficiência precede o gate non-AI e o caminho aderente."""
    perfil, classificacao = _evidencia_e_classificacao(estado)
    confirmados = _ids_confirmados(perfil)
    suporte_confirmado = set(classificacao.ids_afirmacoes_suporte).issubset(
        confirmados
    )
    if not confirmados or not suporte_confirmado:
        return "evidencia_insuficiente"
    if classificacao.classe == "non-AI":
        return "nao_aderente"
    return "prosseguir"
