"""Leitura das fontes curadas da base de conhecimento NVIDIA.

O front matter é um formato plano deliberadamente limitado: linhas
``chave: valor`` entre dois delimitadores ``---``. Não é YAML e não deve
ganhar um parser genérico; toda validação estrutural é explícita.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from radar.contratos import FonteNvidia, TECNOLOGIAS_NVIDIA


DELIMITADOR = "---"
CHAVES_OBRIGATORIAS = ("topico", "origem", "fonte_url", "titulo", "data_acesso")
CHAVES_PERMITIDAS = frozenset(CHAVES_OBRIGATORIAS) | {"tecnologia"}


class ErroFonteNvidia(ValueError):
    """Fonte curada inválida; a mensagem nomeia o arquivo e o problema."""


@dataclass(frozen=True)
class FonteCarregada:
    fonte: FonteNvidia
    corpo: str
    caminho: Path


def interpretar_arquivo_fonte(texto: str, nome: str) -> tuple[FonteNvidia, str]:
    linhas = texto.split("\n")
    if not linhas or linhas[0].strip() != DELIMITADOR:
        raise ErroFonteNvidia(
            f"{nome}: o arquivo deve começar com o delimitador '---' do front matter"
        )
    try:
        fechamento = next(
            indice
            for indice, linha in enumerate(linhas[1:], start=1)
            if linha.strip() == DELIMITADOR
        )
    except StopIteration:
        raise ErroFonteNvidia(
            f"{nome}: front matter sem delimitador de fechamento '---'"
        ) from None

    dados: dict[str, str] = {}
    for numero, linha in enumerate(linhas[1:fechamento], start=2):
        chave, separador, valor = linha.partition(":")
        chave = chave.strip()
        valor = valor.strip()
        if not separador or not chave or not valor:
            raise ErroFonteNvidia(
                f"{nome}: linha {numero} do front matter não segue o formato "
                f"'chave: valor': {linha!r}"
            )
        if chave in dados:
            raise ErroFonteNvidia(f"{nome}: chave duplicada no front matter: {chave}")
        if chave not in CHAVES_PERMITIDAS:
            raise ErroFonteNvidia(f"{nome}: chave desconhecida no front matter: {chave}")
        dados[chave] = valor

    ausentes = [chave for chave in CHAVES_OBRIGATORIAS if chave not in dados]
    if ausentes:
        raise ErroFonteNvidia(
            f"{nome}: chaves obrigatórias ausentes no front matter: {', '.join(ausentes)}"
        )

    fonte = FonteNvidia(**dados)
    corpo = "\n".join(linhas[fechamento + 1 :]).lstrip("\n")
    if not corpo.strip():
        raise ErroFonteNvidia(
            f"{nome}: corpo vazio; uma fonte sem conteúdo não pode contar para cobertura"
        )
    return fonte, corpo


def carregar_fontes(diretorio: Path) -> list[FonteCarregada]:
    arquivos = sorted(diretorio.glob("*.md"))
    if not arquivos:
        raise ErroFonteNvidia(f"nenhum arquivo de fonte encontrado em {diretorio}")
    fontes: list[FonteCarregada] = []
    urls_vistas: dict[str, str] = {}
    for arquivo in arquivos:
        fonte, corpo = interpretar_arquivo_fonte(
            arquivo.read_text(encoding="utf-8"), arquivo.name
        )
        url = str(fonte.fonte_url)
        if url in urls_vistas:
            raise ErroFonteNvidia(
                f"{arquivo.name}: fonte_url repetida (já usada em {urls_vistas[url]}): {url}"
            )
        urls_vistas[url] = arquivo.name
        fontes.append(FonteCarregada(fonte=fonte, corpo=corpo, caminho=arquivo))
    return fontes


def validar_cobertura(fontes: Sequence[FonteCarregada]) -> None:
    """Invariante do manifesto: as 16 tecnologias do TAPI + material conceitual."""
    vazias = [item.caminho.name for item in fontes if not item.corpo.strip()]
    if vazias:
        raise ErroFonteNvidia(
            "fontes sem corpo não podem contar para cobertura: " + ", ".join(vazias)
        )
    cobertas = {item.fonte.tecnologia for item in fontes if item.fonte.tecnologia}
    ausentes = [tecnologia for tecnologia in TECNOLOGIAS_NVIDIA if tecnologia not in cobertas]
    if ausentes:
        raise ErroFonteNvidia(
            "cobertura incompleta da base de conhecimento; tecnologias sem fonte: "
            + ", ".join(ausentes)
        )
    if not any(item.fonte.origem == "conceitual" for item in fontes):
        raise ErroFonteNvidia(
            "a base de conhecimento exige ao menos uma fonte conceitual "
            "(AI-native services e materiais de contexto)"
        )
