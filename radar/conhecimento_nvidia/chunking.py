"""Chunking semântico da base de conhecimento NVIDIA.

Corta na fronteira de headings (H1 a H6) preservando o breadcrumb da
hierarquia. Seções acima do teto quebram primeiro em parágrafos, depois em
sentenças; o corte duro só acontece num segmento indivisível e nunca parte
um code point, porque o fatiamento opera sobre ``str`` (code points), não
sobre bytes. Embeda-se ``breadcrumb + "\\n\\n" + texto_limpo``; a citação
usa somente ``texto_limpo``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

from radar.configuracao import TETO_CARACTERES_CHUNK
from radar.contratos import ChunkNvidia
from radar.conhecimento_nvidia.fontes import FonteCarregada


PADRAO_HEADING = re.compile(r"^(#{1,6})(?!#)\s+(.+?)\s*$")
PADRAO_PARAGRAFO = re.compile(r"\n\s*\n")
PADRAO_SENTENCA = re.compile(r"(?<=[.!?…])\s+")


class ErroChunkingNvidia(ValueError):
    """Estrutura de headings incompatível com identidade estável de chunk."""


def calcular_hash(texto_embedado: str) -> str:
    return hashlib.sha256(texto_embedado.encode("utf-8")).hexdigest()


def texto_para_embedding(chunk: ChunkNvidia) -> str:
    return f"{chunk.breadcrumb}\n\n{chunk.texto_limpo}"


def _montar_breadcrumb(titulo: str, pilha: list[tuple[int, str]]) -> str:
    componentes = [titulo] + [texto for _, texto in pilha]
    sem_repeticao: list[str] = []
    for componente in componentes:
        if not sem_repeticao or sem_repeticao[-1] != componente:
            sem_repeticao.append(componente)
    return " > ".join(sem_repeticao)


def _secoes(titulo: str, corpo: str) -> Iterator[tuple[str, str]]:
    """Percorre o corpo emitindo (breadcrumb, texto) por fronteira de heading."""
    pilha: list[tuple[int, str]] = []
    linhas_atuais: list[str] = []
    breadcrumb_atual = _montar_breadcrumb(titulo, pilha)
    for linha in corpo.split("\n"):
        heading = PADRAO_HEADING.match(linha)
        if heading is None:
            linhas_atuais.append(linha)
            continue
        yield breadcrumb_atual, "\n".join(linhas_atuais).strip()
        nivel = len(heading.group(1))
        while pilha and pilha[-1][0] >= nivel:
            pilha.pop()
        pilha.append((nivel, heading.group(2)))
        breadcrumb_atual = _montar_breadcrumb(titulo, pilha)
        linhas_atuais = []
    yield breadcrumb_atual, "\n".join(linhas_atuais).strip()


def _quebrar_paragrafo(paragrafo: str, teto: int) -> list[str]:
    partes: list[str] = []
    atual: list[str] = []

    def fechar() -> None:
        if atual:
            partes.append(" ".join(atual))
            atual.clear()

    for sentenca in PADRAO_SENTENCA.split(paragrafo):
        if len(sentenca) > teto:
            fechar()
            partes.extend(
                sentenca[inicio : inicio + teto]
                for inicio in range(0, len(sentenca), teto)
            )
            continue
        tamanho_candidato = (
            len(sentenca) if not atual else len(" ".join(atual)) + 1 + len(sentenca)
        )
        if tamanho_candidato > teto:
            fechar()
        atual.append(sentenca)
    fechar()
    return partes


def _quebrar_secao(texto: str, teto: int) -> list[str]:
    if len(texto) <= teto:
        return [texto]
    partes: list[str] = []
    atual: list[str] = []

    def fechar() -> None:
        if atual:
            partes.append("\n\n".join(atual))
            atual.clear()

    for paragrafo in PADRAO_PARAGRAFO.split(texto):
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue
        if len(paragrafo) > teto:
            fechar()
            partes.extend(_quebrar_paragrafo(paragrafo, teto))
            continue
        tamanho_candidato = (
            len(paragrafo)
            if not atual
            else len("\n\n".join(atual)) + 2 + len(paragrafo)
        )
        if tamanho_candidato > teto:
            fechar()
        atual.append(paragrafo)
    fechar()
    return partes


def gerar_chunks(
    fonte: FonteCarregada, teto: int = TETO_CARACTERES_CHUNK
) -> list[ChunkNvidia]:
    chunks: list[ChunkNvidia] = []
    breadcrumbs_emitidos: set[str] = set()
    for breadcrumb, texto in _secoes(fonte.fonte.titulo, fonte.corpo):
        if not texto:
            continue
        if breadcrumb in breadcrumbs_emitidos:
            raise ErroChunkingNvidia(
                f"{fonte.caminho.name}: breadcrumb repetido na mesma fonte "
                f"({breadcrumb!r}); renomeie o heading para manter a identidade "
                "dos chunks estável"
            )
        breadcrumbs_emitidos.add(breadcrumb)
        for indice_parte, parte in enumerate(_quebrar_secao(texto, teto), start=1):
            chunks.append(
                ChunkNvidia(
                    topico=fonte.fonte.topico,
                    origem=fonte.fonte.origem,
                    tecnologia=fonte.fonte.tecnologia,
                    fonte_url=fonte.fonte.fonte_url,
                    breadcrumb=breadcrumb,
                    texto_limpo=parte,
                    indice_parte=indice_parte,
                    hash_texto=calcular_hash(f"{breadcrumb}\n\n{parte}"),
                )
            )
    return chunks
