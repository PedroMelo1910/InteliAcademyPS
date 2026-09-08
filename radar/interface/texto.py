"""Fronteira única de escape: texto dinâmico entra no Markdown como texto.

Nada do que a base curada, o documento público ou o LLM escrevem pode virar
estrutura de documento. O contrato de evidência garante que o
``trecho_citado`` é substring literal da fonte — não que a fonte seja
inofensiva em Markdown. Um trecho copiado de uma página real pode conter
``[clique aqui](javascript:alert(1))``, e ele viaja igual para a tela e para o
arquivo baixado, que é aberto em visualizadores fora do nosso controle.

O escape é determinístico e reversível na leitura: o renderizador remove as
barras invertidas, então a redação exibida é exatamente a original. Só a
sintaxe morre; o conteúdo e os identificadores de evidência ficam intactos.
"""

from __future__ import annotations

import re
from urllib.parse import quote, urlsplit


# Pontuação que constrói link, imagem, HTML, código, ênfase ou tabela em
# qualquer posição da linha.
PONTUACAO_ESTRUTURAL = "\\`*_[]()<>!|~"

_TRADUCAO = str.maketrans({caractere: "\\" + caractere for caractere in PONTUACAO_ESTRUTURAL})

# Marcadores que só criam bloco quando abrem a linha: lista, título ATX,
# título setext. O blockquote ``>`` já morre na tradução acima.
_ABERTURA_DE_BLOCO = re.compile(r"^(\s*)([-+#=])")
_LISTA_ORDENADA = re.compile(r"^(\s*\d+)\.")


def escapar_markdown(texto: str) -> str:
    """Devolve ``texto`` inerte em Markdown, preservando a redação renderizada.

    Aplicado a valores livres — descrição, tese, síntese, pontos, avisos,
    justificativas, trechos citados, títulos de fonte, tópico e breadcrumb.
    **Não** se aplica a URL validada usada como destino de link: escapá-la
    quebraria o próprio link que dá rastreabilidade à evidência.
    """
    return "\n".join(_escapar_linha(linha) for linha in texto.split("\n"))


def _escapar_linha(linha: str) -> str:
    escapada = linha.translate(_TRADUCAO)
    escapada = _ABERTURA_DE_BLOCO.sub(r"\1\\\2", escapada, count=1)
    return _LISTA_ORDENADA.sub(r"\1\\.", escapada, count=1)


ESQUEMAS_DE_LINK = ("http", "https")

# Tudo que é legítimo numa URL sobrevive; o resto é percent-codificado. Ficam
# de fora do conjunto seguro os caracteres que fecham ou reestruturam o
# destino: parêntese, espaço, quebra de linha, angular, barra invertida,
# colchete, aspas e crase. O ``%`` é seguro justamente para não recodificar
# escapes que já existem — ``%20`` não pode virar ``%2520``.
CARACTERES_SEGUROS_NO_DESTINO = "%:/?#@!$&\'*+,;=~_.-"


def destino_markdown(url: object) -> str:
    """Converte uma URL http(s) já validada em destino de link inerte.

    Escapar o rótulo não basta: uma URL válida pode conter ``)`` e encerrar o
    destino mais cedo, deixando o resto da linha virar um segundo link ou uma
    imagem remota. A saída continua clicável — é percent-encoding, não escape
    de texto visível.
    """
    endereco = str(url)
    esquema = urlsplit(endereco).scheme.casefold()
    if esquema not in ESQUEMAS_DE_LINK:
        raise ValueError(
            f"destino de link só aceita http ou https; recebeu {esquema or 'nenhum'!r}"
        )
    return quote(endereco, safe=CARACTERES_SEGUROS_NO_DESTINO)
