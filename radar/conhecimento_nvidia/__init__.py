"""Base de conhecimento NVIDIA: ingestão, índices e consulta híbrida com reranking.

Fronteira do Entregável 2. Este pacote não importa ``base_startups`` e
``base_startups`` não importa este pacote: os dois corpora permanecem
fisicamente e logicamente separados.
"""

from radar.conhecimento_nvidia.consulta import ConhecimentoNvidia, ErroConsultaNvidia
from radar.conhecimento_nvidia.ingestao import (
    ErroIngestaoNvidia,
    ResumoIngestao,
    ingerir,
)

__all__ = [
    "ConhecimentoNvidia",
    "ErroConsultaNvidia",
    "ErroIngestaoNvidia",
    "ResumoIngestao",
    "ingerir",
]
