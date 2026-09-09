"""Gera ou atualiza o cache de análises das startups curadas."""

from __future__ import annotations

import argparse
import sqlite3

from radar.base_startups import BaseStartups
from radar.configuracao import CAMINHO_BANCO
from radar.lote import criar_analisador_lote, estimar_chamadas_externas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pré-analisa as startups e atualiza o ranking persistido."
    )
    parser.add_argument(
        "--executar",
        action="store_true",
        help="confirma as chamadas externas e a atualização do cache",
    )
    parser.add_argument(
        "--startup-id",
        action="append",
        type=int,
        help="limita a execução a um id; pode ser informado mais de uma vez",
    )
    parser.add_argument(
        "--somente-ausentes",
        action="store_true",
        help="processa apenas startups que ainda não têm análise válida no cache",
    )
    argumentos = parser.parse_args()

    if argumentos.startup_id and argumentos.somente_ausentes:
        parser.error("use --startup-id ou --somente-ausentes, não os dois juntos")

    if not CAMINHO_BANCO.exists():
        parser.error(
            "dados/radar.db não existe; execute python -m scripts.inicializar_base"
        )
    base = BaseStartups(CAMINHO_BANCO)
    try:
        ids_disponiveis = [
            item.id_startup for item in base.listar_startups_para_lote()
        ]
    except sqlite3.DatabaseError:
        parser.error(
            "dados/radar.db não está inicializado; execute "
            "python -m scripts.inicializar_base"
        )
    if argumentos.startup_id:
        ids = list(dict.fromkeys(argumentos.startup_id))
        desconhecidos = sorted(set(ids) - set(ids_disponiveis))
        if desconhecidos:
            parser.error(f"startup_id inexistente na base: {desconhecidos}")
    elif argumentos.somente_ausentes:
        try:
            analisadas = base.carregar_analises(ids_disponiveis)
        except sqlite3.DatabaseError:
            parser.error(
                "cache de análises não está preparado; execute "
                "python -m scripts.inicializar_base"
            )
        ids = [item for item in ids_disponiveis if item not in analisadas]
    else:
        ids = ids_disponiveis
    quantidade = len(ids)
    minimo, maximo = estimar_chamadas_externas(quantidade)
    print(f"Startups encontradas: {quantidade}")
    print(
        "Invocações lógicas aos modelos: "
        f"{minimo} no caminho feliz; até {maximo} com correções e reextrações."
    )
    if not argumentos.executar:
        print("Prévia apenas. Para atualizar, use --executar.")
        return 0

    analisador = criar_analisador_lote()
    try:
        resultado = analisador.executar_todas(ids)
        cobertura = analisador.base.cobertura_analises()
    finally:
        analisador.fechar()

    print(f"Concluídas nesta execução: {len(resultado.concluidas)}")
    print(
        "Evidências insuficientes nesta execução: "
        f"{len(resultado.evidencias_insuficientes)}"
    )
    for falha in resultado.falhas:
        print(f"Falha {falha.startup_id} ({falha.tipo}): {falha.mensagem}")
    print(
        "Cobertura do cache: "
        f"{cobertura.concluidas} concluídas, "
        f"{cobertura.evidencias_insuficientes} insuficientes e "
        f"{cobertura.ausentes} ausentes."
    )
    return 1 if resultado.falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
