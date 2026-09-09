"""Memória da sessão: o que sobrevive a um rerun e o que uma busca invalida.

O Streamlit reexecuta o script inteiro a cada interação. Sem um lugar
explícito para guardar a descoberta e o aprofundamento já concluídos, cada
clique reabriria uma execução de grafo. Estas funções operam sobre qualquer
mapa (o ``session_state`` real ou um dicionário no teste) usando apenas
``in``, indexação e ``del`` — o subconjunto que os dois suportam.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - só para tipagem
    from radar.aplicacao import SaidaAprofundamento, SaidaDescoberta


CHAVE_APLICACAO = "aplicacao_radar"
CHAVE_DESCOBERTA = "descoberta"
CHAVE_SELECIONADA = "startup_selecionada"
CHAVE_APROFUNDAMENTOS = "aprofundamentos"
CHAVE_FALHAS = "falhas_aprofundamento"
CHAVE_CARTOES = "cartoes_atualizados"

Memoria = MutableMapping[str, Any]


# ----------------------------------------------------------------------
# Descoberta
# ----------------------------------------------------------------------


def iniciar_busca(memoria: Memoria) -> None:
    """Zera a sessão no instante da submissão: nada antigo sobrevive à consulta nova."""
    for chave in (
        CHAVE_DESCOBERTA,
        CHAVE_SELECIONADA,
        CHAVE_APROFUNDAMENTOS,
        CHAVE_FALHAS,
        CHAVE_CARTOES,
    ):
        _descartar(memoria, chave)


def registrar_descoberta(memoria: Memoria, descoberta: SaidaDescoberta) -> None:
    """Guarda a descoberta corrente e invalida seleção e briefing anteriores."""
    iniciar_busca(memoria)
    memoria[CHAVE_DESCOBERTA] = descoberta


def descoberta_atual(memoria: Memoria) -> SaidaDescoberta | None:
    return _ler(memoria, CHAVE_DESCOBERTA)


# ----------------------------------------------------------------------
# Seleção de candidata
# ----------------------------------------------------------------------


def selecionar_startup(memoria: Memoria, id_startup: int) -> None:
    """Um clique explícito: escolhe a candidata e libera nova tentativa após falha."""
    memoria[CHAVE_SELECIONADA] = id_startup
    _falhas(memoria).pop(id_startup, None)


def limpar_selecao(memoria: Memoria) -> None:
    """Volta para a lista sem descartar o que já foi analisado."""
    _descartar(memoria, CHAVE_SELECIONADA)


def startup_selecionada(memoria: Memoria) -> int | None:
    return _ler(memoria, CHAVE_SELECIONADA)


# ----------------------------------------------------------------------
# Aprofundamento
# ----------------------------------------------------------------------


def aprofundamento_pendente(memoria: Memoria) -> int | None:
    """Id que ainda precisa rodar; ``None`` quando já há resultado ou falha."""
    id_startup = startup_selecionada(memoria)
    if id_startup is None:
        return None
    if id_startup in _aprofundamentos(memoria) or id_startup in _falhas(memoria):
        return None
    return id_startup


def registrar_aprofundamento(memoria: Memoria, saida: SaidaAprofundamento) -> None:
    """Guarda o resultado e atualiza o cartão da candidata nesta sessão.

    A descoberta é uma fotografia do cache no instante da busca; depois de uma
    análise bem-sucedida ela fica desatualizada para aquela startup. A correção
    vive só aqui: o SQLite não é tocado e o ranking não é reordenado.
    """
    from radar.interface.rotulos import atualizar_item_com_briefing

    _aprofundamentos(memoria)[saida.id_startup] = saida
    descoberta = descoberta_atual(memoria)
    if descoberta is None:
        return
    for item in descoberta.ranking:
        if item.empresa.id_startup == saida.id_startup:
            _cartoes(memoria)[saida.id_startup] = atualizar_item_com_briefing(
                item, saida.briefing
            )
            return


def registrar_falha_aprofundamento(
    memoria: Memoria, id_startup: int, mensagem: str
) -> None:
    """Falha é resultado registrado, não briefing: nenhum rerun a repete sozinho."""
    _falhas(memoria)[id_startup] = mensagem


def aprofundamento_selecionado(memoria: Memoria) -> SaidaAprofundamento | None:
    id_startup = startup_selecionada(memoria)
    if id_startup is None:
        return None
    return _aprofundamentos(memoria).get(id_startup)


def falha_selecionada(memoria: Memoria) -> str | None:
    id_startup = startup_selecionada(memoria)
    if id_startup is None:
        return None
    return _falhas(memoria).get(id_startup)


# ----------------------------------------------------------------------
# Apoio
# ----------------------------------------------------------------------


def ranking_visivel(memoria: Memoria) -> tuple:
    """O ranking da descoberta, com os cartões que esta sessão já atualizou.

    A ordem é a que a aplicação entregou: só o conteúdo de um cartão pode
    mudar, nunca a posição dele.
    """
    descoberta = descoberta_atual(memoria)
    if descoberta is None:
        return ()
    cartoes = _cartoes(memoria)
    return tuple(
        cartoes.get(item.empresa.id_startup, item) for item in descoberta.ranking
    )


def _cartoes(memoria: Memoria) -> dict:
    return _mapa(memoria, CHAVE_CARTOES)


def _aprofundamentos(memoria: Memoria) -> dict[int, SaidaAprofundamento]:
    return _mapa(memoria, CHAVE_APROFUNDAMENTOS)


def _falhas(memoria: Memoria) -> dict[int, str]:
    return _mapa(memoria, CHAVE_FALHAS)


def _mapa(memoria: Memoria, chave: str) -> dict:
    if chave not in memoria:
        memoria[chave] = {}
    return memoria[chave]


def _ler(memoria: Memoria, chave: str):
    return memoria[chave] if chave in memoria else None


def _descartar(memoria: Memoria, chave: str) -> None:
    if chave in memoria:
        del memoria[chave]
