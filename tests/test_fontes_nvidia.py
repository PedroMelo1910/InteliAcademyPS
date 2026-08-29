"""Parser limitado de front matter das fontes NVIDIA e invariantes do manifesto."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from radar.contratos import FonteNvidia, TECNOLOGIAS_NVIDIA
from radar.conhecimento_nvidia.fontes import (
    ErroFonteNvidia,
    FonteCarregada,
    carregar_fontes,
    interpretar_arquivo_fonte,
    validar_cobertura,
)


FRONT_MATTER_VALIDO = """---
topico: nim
origem: tecnologia
tecnologia: NVIDIA NIM
fonte_url: https://exemplo.nvidia.com/nim
titulo: NVIDIA NIM
data_acesso: 2026-08-25
---

# Visão geral

NIM empacota modelos como microservices otimizados.
"""


def escrever(caminho: Path, texto: str) -> Path:
    caminho.write_text(texto, encoding="utf-8")
    return caminho


def fonte_carregada(tecnologia: str | None, topico: str, url: str) -> FonteCarregada:
    origem = "tecnologia" if tecnologia else "conceitual"
    fonte = FonteNvidia(
        topico=topico,
        origem=origem,
        tecnologia=tecnologia,
        fonte_url=url,
        titulo=topico.replace("_", " ").title(),
        data_acesso=date(2026, 8, 25),
    )
    return FonteCarregada(fonte=fonte, corpo="## Secao\n\nConteudo.", caminho=Path(f"{topico}.md"))


def test_arquivo_valido_produz_fonte_e_corpo():
    fonte, corpo = interpretar_arquivo_fonte(FRONT_MATTER_VALIDO, "nim.md")
    assert fonte.tecnologia == "NVIDIA NIM"
    assert fonte.data_acesso == date(2026, 8, 25)
    assert corpo.startswith("# Visão geral")


def test_arquivo_com_corpo_vazio_e_rejeitado():
    sem_corpo = FRONT_MATTER_VALIDO[: FRONT_MATTER_VALIDO.rfind("---") + 3]
    with pytest.raises(ErroFonteNvidia, match="corpo vazio"):
        interpretar_arquivo_fonte(sem_corpo, "nim_vazio.md")


def test_erros_estruturais_do_front_matter():
    sem_abertura = FRONT_MATTER_VALIDO.replace("---\n", "", 1)
    with pytest.raises(ErroFonteNvidia, match="---"):
        interpretar_arquivo_fonte(sem_abertura, "a.md")

    sem_fechamento = "---\ntopico: nim\n"
    with pytest.raises(ErroFonteNvidia, match="fechamento"):
        interpretar_arquivo_fonte(sem_fechamento, "b.md")

    linha_invalida = FRONT_MATTER_VALIDO.replace("topico: nim", "topico nim")
    with pytest.raises(ErroFonteNvidia, match="chave: valor"):
        interpretar_arquivo_fonte(linha_invalida, "c.md")


def test_chave_obrigatoria_ausente_e_nomeada():
    sem_titulo = FRONT_MATTER_VALIDO.replace("titulo: NVIDIA NIM\n", "")
    with pytest.raises(ErroFonteNvidia, match="titulo"):
        interpretar_arquivo_fonte(sem_titulo, "nim.md")


def test_chave_duplicada_e_rejeitada():
    duplicado = FRONT_MATTER_VALIDO.replace(
        "topico: nim", "topico: nim\ntopico: outro"
    )
    with pytest.raises(ErroFonteNvidia, match="duplicada"):
        interpretar_arquivo_fonte(duplicado, "nim.md")


def test_chave_desconhecida_e_rejeitada():
    desconhecida = FRONT_MATTER_VALIDO.replace(
        "topico: nim", "topico: nim\nautor: alguem"
    )
    with pytest.raises(ErroFonteNvidia, match="desconhecida"):
        interpretar_arquivo_fonte(desconhecida, "nim.md")


def test_valores_invalidos_sao_rejeitados_pelo_contrato():
    tecnologia_falsa = FRONT_MATTER_VALIDO.replace("NVIDIA NIM", "NVIDIA Inventada")
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(tecnologia_falsa, "nim.md")

    origem_falsa = FRONT_MATTER_VALIDO.replace("origem: tecnologia", "origem: marketing")
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(origem_falsa, "nim.md")

    url_falsa = FRONT_MATTER_VALIDO.replace("https://exemplo.nvidia.com/nim", "nada")
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(url_falsa, "nim.md")

    data_falsa = FRONT_MATTER_VALIDO.replace("2026-08-25", "25/08/2026")
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(data_falsa, "nim.md")


def test_combinacao_incompativel_de_origem_e_tecnologia():
    conceitual_com_tecnologia = FRONT_MATTER_VALIDO.replace(
        "origem: tecnologia", "origem: conceitual"
    )
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(conceitual_com_tecnologia, "nim.md")

    tecnologia_sem_valor = FRONT_MATTER_VALIDO.replace(
        "tecnologia: NVIDIA NIM\n", ""
    )
    with pytest.raises(ValidationError):
        interpretar_arquivo_fonte(tecnologia_sem_valor, "nim.md")


def test_carregar_fontes_le_diretorio_em_ordem_estavel(tmp_path):
    escrever(tmp_path / "02_b.md", FRONT_MATTER_VALIDO.replace("nim", "riva").replace(
        "NVIDIA NIM", "NVIDIA Riva"
    ))
    escrever(tmp_path / "01_a.md", FRONT_MATTER_VALIDO)
    fontes = carregar_fontes(tmp_path)
    assert [item.caminho.name for item in fontes] == ["01_a.md", "02_b.md"]
    assert fontes[0].fonte.tecnologia == "NVIDIA NIM"
    assert fontes[0].corpo.startswith("# Visão geral")


def test_carregar_fontes_rejeita_diretorio_vazio_e_url_duplicada(tmp_path):
    with pytest.raises(ErroFonteNvidia, match="nenhum"):
        carregar_fontes(tmp_path)

    escrever(tmp_path / "01_a.md", FRONT_MATTER_VALIDO)
    escrever(
        tmp_path / "02_b.md",
        FRONT_MATTER_VALIDO.replace("topico: nim", "topico: nim2").replace(
            "titulo: NVIDIA NIM", "titulo: NVIDIA NIM 2"
        ),
    )
    with pytest.raises(ErroFonteNvidia, match="fonte_url"):
        carregar_fontes(tmp_path)


def test_cobertura_completa_das_dezesseis_tecnologias_e_do_conceitual():
    fontes = [
        fonte_carregada(tecnologia, f"tec_{indice:02d}", f"https://exemplo.com/{indice}")
        for indice, tecnologia in enumerate(TECNOLOGIAS_NVIDIA)
    ]
    fontes.append(
        fonte_carregada(None, "ai_native_services", "https://exemplo.com/conceito")
    )
    validar_cobertura(fontes)  # não deve levantar

    sem_riva = [
        item for item in fontes if item.fonte.tecnologia != "NVIDIA Riva"
    ]
    with pytest.raises(ErroFonteNvidia, match="NVIDIA Riva"):
        validar_cobertura(sem_riva)

    sem_conceitual = fontes[:-1]
    with pytest.raises(ErroFonteNvidia, match="conceitual"):
        validar_cobertura(sem_conceitual)
