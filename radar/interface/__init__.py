"""Apoio da interface: rótulos, estado de sessão, tema e exportação.

Nenhum módulo daqui importa Streamlit. A tela (``app.py``) orquestra estas
funções puras; assim a jornada inteira é testável sem widget e sem rede, e a
regra de negócio continua onde sempre esteve — nos agentes e nos contratos.
"""

from radar.interface.exportacao import (
    exportar_briefing_markdown,
    nome_arquivo_briefing,
)
from radar.interface.rotulos import (
    ResumoCandidata,
    resumir_candidata,
    resumir_ranking,
)

__all__ = [
    "ResumoCandidata",
    "exportar_briefing_markdown",
    "nome_arquivo_briefing",
    "resumir_candidata",
    "resumir_ranking",
]
