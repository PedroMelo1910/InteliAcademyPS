"""Chunking semântico por headings, breadcrumb e quebra segura de seções longas."""

import hashlib
from datetime import date
from pathlib import Path

import pytest

from radar.contratos import FonteNvidia
from radar.conhecimento_nvidia.chunking import (
    ErroChunkingNvidia,
    calcular_hash,
    gerar_chunks,
    texto_para_embedding,
)
from radar.conhecimento_nvidia.fontes import FonteCarregada


def fonte_com_corpo(corpo: str, origem: str = "tecnologia") -> FonteCarregada:
    fonte = FonteNvidia(
        topico="nim",
        origem=origem,
        tecnologia="NVIDIA NIM" if origem == "tecnologia" else None,
        fonte_url="https://exemplo.nvidia.com/nim",
        titulo="NVIDIA NIM",
        data_acesso=date(2026, 8, 25),
    )
    return FonteCarregada(fonte=fonte, corpo=corpo, caminho=Path("nim.md"))


def test_breadcrumb_segue_a_hierarquia_de_headings():
    corpo = (
        "Introdução solta antes de qualquer heading.\n\n"
        "# Visão geral\n\nTexto da visão.\n\n"
        "## Deploy\n\n"
        "### Requisitos\n\nTexto dos requisitos.\n\n"
        "## Casos de uso\n\nTexto dos casos.\n"
    )
    chunks = gerar_chunks(fonte_com_corpo(corpo))
    breadcrumbs = [chunk.breadcrumb for chunk in chunks]
    assert breadcrumbs == [
        "NVIDIA NIM",
        "NVIDIA NIM > Visão geral",
        "NVIDIA NIM > Visão geral > Deploy > Requisitos",
        "NVIDIA NIM > Visão geral > Casos de uso",
    ]
    assert chunks[2].texto_limpo == "Texto dos requisitos."


def test_secao_vazia_e_ignorada_e_secao_curta_e_preservada():
    corpo = (
        "# Vazia\n\n"
        "# Curta\n\nSó uma linha.\n\n"
        "# Cheia\n\nConteúdo maior da seção cheia.\n"
    )
    chunks = gerar_chunks(fonte_com_corpo(corpo))
    assert [chunk.breadcrumb for chunk in chunks] == [
        "NVIDIA NIM > Curta",
        "NVIDIA NIM > Cheia",
    ]
    assert chunks[0].texto_limpo == "Só uma linha."


def test_h1_igual_ao_titulo_nao_duplica_a_raiz_do_breadcrumb():
    corpo = "# NVIDIA NIM\n\nDescrição geral.\n\n## Deploy\n\nComo publicar.\n"
    chunks = gerar_chunks(fonte_com_corpo(corpo))
    assert [chunk.breadcrumb for chunk in chunks] == [
        "NVIDIA NIM",
        "NVIDIA NIM > Deploy",
    ]


def test_headings_de_h1_a_h6_sao_reconhecidos():
    corpo = (
        "###### Nível seis\n\nTexto profundo.\n\n"
        "####### Sete hashes não é heading.\n"
    )
    chunks = gerar_chunks(fonte_com_corpo(corpo))
    assert chunks[0].breadcrumb == "NVIDIA NIM > Nível seis"
    assert "#######" in chunks[0].texto_limpo


def test_secao_longa_quebra_em_paragrafos_dentro_do_teto():
    paragrafos = [f"Parágrafo {i} com conteúdo razoável." for i in range(8)]
    corpo = "# Longa\n\n" + "\n\n".join(paragrafos) + "\n"
    chunks = gerar_chunks(fonte_com_corpo(corpo), teto=80)
    assert len(chunks) > 1
    assert all(len(chunk.texto_limpo) <= 80 for chunk in chunks)
    assert [chunk.indice_parte for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.breadcrumb == "NVIDIA NIM > Longa" for chunk in chunks)
    texto_junto = "\n\n".join(chunk.texto_limpo for chunk in chunks)
    assert texto_junto == "\n\n".join(paragrafos)


def test_paragrafo_maior_que_o_teto_quebra_por_sentencas():
    sentencas = [f"Sentença número {i} deste parágrafo." for i in range(6)]
    corpo = "# Denso\n\n" + " ".join(sentencas) + "\n"
    chunks = gerar_chunks(fonte_com_corpo(corpo), teto=90)
    assert len(chunks) > 1
    assert all(len(chunk.texto_limpo) <= 90 for chunk in chunks)
    reconstruido = " ".join(chunk.texto_limpo for chunk in chunks)
    assert reconstruido == " ".join(sentencas)


def test_sentenca_indivisivel_recebe_corte_duro_sem_partir_code_points():
    bloco = "informação🚀ção" * 40  # sem pontuação: indivisível
    corpo = "# Bloco\n\n" + bloco + "\n"
    teto = 100
    chunks = gerar_chunks(fonte_com_corpo(corpo), teto=teto)
    assert len(chunks) > 1
    assert all(len(chunk.texto_limpo) <= teto for chunk in chunks)
    assert "".join(chunk.texto_limpo for chunk in chunks) == bloco
    for chunk in chunks:
        chunk.texto_limpo.encode("utf-8").decode("utf-8")
        assert "�" not in chunk.texto_limpo


def test_hash_e_texto_de_embedding_derivam_do_breadcrumb_e_do_corpo():
    corpo = "# Visão geral\n\nTexto da visão.\n"
    chunk = gerar_chunks(fonte_com_corpo(corpo))[0]
    esperado = "NVIDIA NIM > Visão geral\n\nTexto da visão."
    assert texto_para_embedding(chunk) == esperado
    assert chunk.hash_texto == hashlib.sha256(esperado.encode("utf-8")).hexdigest()
    assert calcular_hash(esperado) == chunk.hash_texto


def test_chunking_e_deterministico_e_propaga_metadados():
    corpo = "# Visão geral\n\nTexto.\n\n## Deploy\n\nMais texto.\n"
    fonte = fonte_com_corpo(corpo)
    primeira = gerar_chunks(fonte)
    segunda = gerar_chunks(fonte)
    assert primeira == segunda
    assert all(chunk.topico == "nim" for chunk in primeira)
    assert all(chunk.tecnologia == "NVIDIA NIM" for chunk in primeira)
    assert all(str(chunk.fonte_url) == "https://exemplo.nvidia.com/nim" for chunk in primeira)

    conceitual = gerar_chunks(
        fonte_com_corpo("# Serviços\n\nTexto conceitual.\n", origem="conceitual")
    )
    assert conceitual[0].origem == "conceitual"
    assert conceitual[0].tecnologia is None


def test_breadcrumb_duplicado_na_mesma_fonte_e_rejeitado():
    corpo = "# Recursos\n\nPrimeiro.\n\n# Recursos\n\nSegundo.\n"
    with pytest.raises(ErroChunkingNvidia, match="Recursos"):
        gerar_chunks(fonte_com_corpo(corpo))
