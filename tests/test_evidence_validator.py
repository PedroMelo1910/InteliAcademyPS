import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from radar.agentes.evidence_validator import ErroValidadorEvidencias, EvidenceValidator
from radar.base_startups import BaseStartups, inicializar_banco
from radar.contratos import (
    DIMENSOES_GAP,
    DocumentoRecuperado,
    DocumentoVerificavel,
    EmpresaCandidata,
    EstadoDimensaoGap,
    FiltrosEstruturados,
    PerfilValidado,
    PlanoConsulta,
    ResultadoRecuperacao,
)


# --------------------------------------------------------------------------
# Base controlada com duas startups: só assim dá para provar que um documento
# de outra empresa derruba a afirmação em vez de passar despercebido.
# --------------------------------------------------------------------------

TEXTO_SITE_A = (
    "A Acme Robotics vende inspeção visual automatizada para linhas de montagem. "
    "O banco de imagens rotuladas da Acme Robotics é proprietário e foi construído "
    "junto dos clientes ao longo de seis anos. A plataforma é distribuída como "
    "integração nativa nos sistemas MES de doze fábricas clientes no Brasil."
)
TEXTO_VAGA_A = (
    "A Acme Robotics contrata pessoa engenheira de machine learning para percepção. "
    "A vaga pede experiência em compilação de kernels e otimização própria de "
    "inferência em GPUs, além de rotina de monitoramento contínuo em produção."
)
TEXTO_NOTICIA_A = (
    "Reportagem sobre visão industrial no Brasil menciona a Acme Robotics. "
    "Segundo a matéria, a empresa não possui qualquer otimização própria de "
    "inferência e depende de uma API externa. O texto afirma ainda que a empresa "
    "não mantém workflow proprietário de operação em suas fábricas parceiras."
)
TEXTO_SITE_B = (
    "A Boreal Benefícios opera uma plataforma de cartão de benefícios corporativos. "
    "O cartão é o produto contratado pelas empresas clientes em todo o país. "
    "A companhia mantém uma base própria de transações de uso do benefício."
)
TEXTO_BLOG_B = (
    "No blog, a Boreal Benefícios descreve o processo de adesão das empresas. "
    "A equipe explica como o cartão chega às pessoas colaboradoras em poucos dias. "
    "O texto detalha ainda a rotina de conciliação financeira mensal do produto."
)
TEXTO_NOTICIA_B = (
    "Matéria de mercado cita a Boreal Benefícios entre as fintechs de benefícios. "
    "A reportagem descreve o crescimento da categoria no último ano no Brasil. "
    "O texto menciona a disputa por distribuição junto a grandes empregadores."
)

TRECHO_DADOS = "banco de imagens rotuladas da Acme Robotics é proprietário"
TRECHO_DISTRIBUICAO = "distribuída como integração nativa nos sistemas MES"
TRECHO_OTIMIZACAO = "otimização própria de inferência em GPUs"
TRECHO_AUSENCIA_OTIMIZACAO = "não possui qualquer otimização própria de inferência"
TRECHO_AUSENCIA_WORKFLOW = "não mantém workflow proprietário de operação"
TRECHO_ESCALA = "doze fábricas clientes no Brasil"
TRECHO_OUTRA_STARTUP = "base própria de transações de uso do benefício"


def _startup_a() -> dict:
    return {
        "nome": "Acme Robotics",
        "site": "https://acmerobotics.example.com",
        "setor": "Indústria",
        "estagio": "série A",
        "localizacao": "Campinas, SP",
        "descricao_curta": "Inspeção visual automatizada para linhas de montagem.",
        "ano_fundacao": 2019,
        "tamanho_time": "51-200",
        "classe_referencia": "AI-native",
        "documentos": [
            {
                "tipo": "site institucional",
                "titulo": "Plataforma de inspeção visual",
                "conteudo_texto": TEXTO_SITE_A,
                "url_fonte": "https://acmerobotics.example.com/produto",
                "dominio_fonte": "acmerobotics.example.com",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "vaga",
                "titulo": "Engenharia de machine learning",
                "conteudo_texto": TEXTO_VAGA_A,
                "url_fonte": "https://vagas.example.org/acme-robotics-ml",
                "dominio_fonte": "vagas.example.org",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "notícia",
                "titulo": "Visão industrial cresce no país",
                "conteudo_texto": TEXTO_NOTICIA_A,
                "url_fonte": "https://jornal.example.net/visao-industrial",
                "dominio_fonte": "jornal.example.net",
                "data_acesso": "2026-08-30",
            },
        ],
    }


def _startup_b() -> dict:
    return {
        "nome": "Boreal Benefícios",
        "site": "https://boreal.example.com",
        "setor": "Financeiro",
        "estagio": "série B",
        "localizacao": "São Paulo, SP",
        "descricao_curta": "Cartão de benefícios corporativos.",
        "ano_fundacao": 2018,
        "tamanho_time": "201-500",
        "classe_referencia": "AI-enabled",
        "documentos": [
            {
                "tipo": "site institucional",
                "titulo": "Cartão de benefícios",
                "conteudo_texto": TEXTO_SITE_B,
                "url_fonte": "https://boreal.example.com/produto",
                "dominio_fonte": "boreal.example.com",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "blog",
                "titulo": "Como funciona a adesão",
                "conteudo_texto": TEXTO_BLOG_B,
                "url_fonte": "https://blog.example.org/boreal-adesao",
                "dominio_fonte": "blog.example.org",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "notícia",
                "titulo": "Fintechs de benefícios crescem",
                "conteudo_texto": TEXTO_NOTICIA_B,
                "url_fonte": "https://mercado.example.net/fintechs-beneficios",
                "dominio_fonte": "mercado.example.net",
                "data_acesso": "2026-08-30",
            },
        ],
    }


@dataclass(frozen=True)
class BaseDupla:
    base: BaseStartups
    caminho: Path
    id_a: int
    id_b: int
    ids_a: dict[str, int]
    ids_b: dict[str, int]


@pytest.fixture
def dupla(tmp_path: Path) -> BaseDupla:
    curadoria = tmp_path / "curadoria"
    curadoria.mkdir(parents=True, exist_ok=True)
    for arquivo, registro in (
        ("01_acme.json", _startup_a()),
        ("02_boreal.json", _startup_b()),
    ):
        with (curadoria / arquivo).open("w", encoding="utf-8") as saida:
            json.dump(registro, saida, ensure_ascii=False)
    banco = tmp_path / "dupla.db"
    inicializar_banco(banco, curadoria)
    with sqlite3.connect(banco) as conexao:
        conexao.row_factory = sqlite3.Row
        startups = {
            linha["nome"]: linha["id"]
            for linha in conexao.execute("SELECT id, nome FROM startups").fetchall()
        }
        documentos = conexao.execute(
            "SELECT id, startup_id, tipo FROM documentos ORDER BY id"
        ).fetchall()
    apelidos = {
        "site institucional": "site",
        "vaga": "vaga",
        "notícia": "noticia",
        "blog": "blog",
    }
    id_a = startups["Acme Robotics"]
    id_b = startups["Boreal Benefícios"]
    ids_a = {
        apelidos[linha["tipo"]]: linha["id"]
        for linha in documentos
        if linha["startup_id"] == id_a
    }
    ids_b = {
        apelidos[linha["tipo"]]: linha["id"]
        for linha in documentos
        if linha["startup_id"] == id_b
    }
    return BaseDupla(BaseStartups(banco), banco, id_a, id_b, ids_a, ids_b)


# --------------------------------------------------------------------------
# Montagem do estado
# --------------------------------------------------------------------------


def afirmacao(
    id_afirmacao: int,
    categoria: str,
    polaridade: str,
    id_documento: int,
    trecho: str,
    texto: str | None = None,
) -> dict:
    return {
        "id_afirmacao": id_afirmacao,
        "texto": texto or f"Fato número {id_afirmacao} sobre a empresa analisada.",
        "categoria": categoria,
        "polaridade": polaridade,
        "id_documento": id_documento,
        "trecho_citado": trecho,
    }


def plano() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor="Indústria"),
        termos_busca=["inspeção visual"],
        foco_analise="uso de visão computacional em chão de fábrica",
    )


def recuperacao(dupla: BaseDupla, ids_documentos: list[int]) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        empresas=[
            EmpresaCandidata(
                id_startup=dupla.id_a,
                nome="Acme Robotics",
                setor="Indústria",
                estagio="série A",
                localizacao="Campinas, SP",
                descricao_curta="Inspeção visual automatizada.",
            )
        ],
        documentos=[
            DocumentoRecuperado(
                id_documento=id_documento,
                id_startup=dupla.id_a,
                tipo="site institucional",
                titulo=f"documento {id_documento}",
                url_fonte=f"https://exemplo.test/{id_documento}",
                dominio_fonte="exemplo.test",
                data_acesso="2026-08-30",
                score_bm25=-1.0,
            )
            for id_documento in ids_documentos
        ],
        filtros_aplicados=plano().filtros,
    )


def estado(
    dupla: BaseDupla,
    afirmacoes: list[dict],
    ids_recuperados: list[int] | None = None,
    ids_suporte: list[int] | None = None,
    **ajustes,
) -> dict:
    if ids_recuperados is None:
        ids_recuperados = sorted(dupla.ids_a.values())
    base_estado = {
        "consulta_usuario": "startups de visão computacional industrial",
        "startup_selecionada": dupla.id_a,
        "plano_consulta": plano(),
        "resultado_recuperacao": recuperacao(dupla, ids_recuperados),
        "perfil_extraido": {
            "id_startup": dupla.id_a,
            "resumo_produto": (
                "A Acme Robotics vende inspeção visual automatizada. "
                "A empresa atende fábricas clientes no Brasil."
            ),
            "afirmacoes": afirmacoes,
        },
        "classificacao": {
            "classe": "AI-native",
            "justificativa": (
                "A empresa treina modelos próprios de detecção de defeitos. "
                "Sem esses modelos não resta produto para o cliente."
            ),
            "ids_afirmacoes_suporte": ids_suporte or [1],
        },
        "tentativas_extracao": 1,
    }
    base_estado.update(ajustes)
    return base_estado


def validar(dupla: BaseDupla, **kwargs) -> dict:
    return EvidenceValidator(dupla.base)(estado(dupla, **kwargs))


def afirmacoes_confirmadas(dupla: BaseDupla) -> list[dict]:
    return [
        afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
        afirmacao(2, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
    ]


# --------------------------------------------------------------------------
# Fronteira de leitura dedicada
# --------------------------------------------------------------------------


def test_leitura_de_verificacao_expoe_texto_e_dominio(dupla: BaseDupla):
    lidos = dupla.base.carregar_documentos_verificaveis([dupla.ids_a["site"]])
    documento = lidos[dupla.ids_a["site"]]
    assert isinstance(documento, DocumentoVerificavel)
    assert documento.id_startup == dupla.id_a
    assert documento.dominio_fonte == "acmerobotics.example.com"
    assert TRECHO_DADOS in documento.conteudo_texto


def test_leitura_de_verificacao_nunca_expoe_classe_referencia(dupla: BaseDupla):
    assert set(DocumentoVerificavel.model_fields) == {
        "id_documento",
        "id_startup",
        "conteudo_texto",
        "dominio_fonte",
    }


def test_leitura_de_verificacao_e_tolerante_a_id_inexistente(dupla: BaseDupla):
    lidos = dupla.base.carregar_documentos_verificaveis([dupla.ids_a["site"], 987654])
    assert set(lidos) == {dupla.ids_a["site"]}


def test_leitura_de_verificacao_usa_sql_parametrizado(dupla: BaseDupla):
    lidos = dupla.base.carregar_documentos_verificaveis(
        ["1); DROP TABLE documentos; --"]
    )
    assert lidos == {}
    with sqlite3.connect(dupla.caminho) as conexao:
        total = conexao.execute("SELECT COUNT(*) FROM documentos").fetchone()[0]
    assert total == 6


def test_leitura_de_verificacao_deduplica_ids_repetidos(dupla: BaseDupla):
    id_site = dupla.ids_a["site"]
    lidos = dupla.base.carregar_documentos_verificaveis([id_site, id_site])
    assert set(lidos) == {id_site}


# --------------------------------------------------------------------------
# Veredito por afirmação
# --------------------------------------------------------------------------


def test_trecho_literal_no_documento_citado_confirma_a_afirmacao(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert [item.situacao for item in perfil.afirmacoes_validadas] == [
        "confirmada",
        "confirmada",
    ]
    assert all(item.motivo is None for item in perfil.afirmacoes_validadas)


def test_normalizacao_aceita_apenas_espaco_colapsado_e_casefold(dupla: BaseDupla):
    citado = "  BANCO   de imagens\nrotuladas da ACME Robotics É PROPRIETÁRIO  "
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], citado)
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[0].situacao == "confirmada"


def test_diferenca_de_acento_nao_e_absorvida_e_derruba_a_afirmacao(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["site"],
                "banco de imagens rotuladas da Acme Robotics e proprietario",
            )
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[0].situacao == "derrubada"
    assert "literal" in perfil.afirmacoes_validadas[0].motivo


def test_documento_inexistente_derruba_com_motivo_explicito(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", 987654, TRECHO_DADOS)
        ],
        ids_recuperados=[*sorted(dupla.ids_a.values()), 987654],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    validada = perfil.afirmacoes_validadas[0]
    assert validada.situacao == "derrubada"
    assert "não existe" in validada.motivo
    assert validada.id_documento == 987654


def test_documento_de_outra_startup_derruba_com_motivo_explicito(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_b["site"],
                TRECHO_OUTRA_STARTUP,
            )
        ],
        ids_recuperados=[*sorted(dupla.ids_a.values()), dupla.ids_b["site"]],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    validada = perfil.afirmacoes_validadas[0]
    assert validada.situacao == "derrubada"
    assert "outra startup" in validada.motivo


def test_documento_fora_do_conjunto_recuperado_derruba_a_afirmacao(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS)
        ],
        ids_recuperados=[dupla.ids_a["vaga"], dupla.ids_a["noticia"]],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    validada = perfil.afirmacoes_validadas[0]
    assert validada.situacao == "derrubada"
    assert "recuperado" in validada.motivo


def test_releitura_consulta_apenas_ids_referenciados_e_recuperados(
    dupla: BaseDupla, monkeypatch
):
    ids_consultados = []
    leitura_real = dupla.base.carregar_documentos_verificaveis

    def registrar(ids_documentos):
        ids_consultados.extend(ids_documentos)
        return leitura_real(ids_documentos)

    monkeypatch.setattr(
        dupla.base, "carregar_documentos_verificaveis", registrar
    )
    validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["site"],
                TRECHO_DADOS,
            ),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "presenca",
                dupla.ids_a["vaga"],
                TRECHO_OTIMIZACAO,
            ),
        ],
        ids_recuperados=[dupla.ids_a["site"], dupla.ids_a["noticia"]],
    )

    assert ids_consultados == [dupla.ids_a["site"]]


def test_perfil_de_startup_diferente_da_selecionada_falha_alto(dupla: BaseDupla):
    entrada = estado(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    entrada["startup_selecionada"] = dupla.id_b

    with pytest.raises(ErroValidadorEvidencias, match="diferente da selecionada"):
        EvidenceValidator(dupla.base)(entrada)


def test_startup_do_perfil_precisa_estar_no_resultado_recuperado(dupla: BaseDupla):
    entrada = estado(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    entrada["resultado_recuperacao"] = entrada["resultado_recuperacao"].model_copy(
        update={"empresas": []}
    )

    with pytest.raises(ErroValidadorEvidencias, match="resultado recuperado"):
        EvidenceValidator(dupla.base)(entrada)


def test_metadado_recuperado_de_outra_startup_nao_autoriza_documento(
    dupla: BaseDupla,
):
    entrada = estado(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["site"],
                TRECHO_DADOS,
            )
        ],
        ids_recuperados=[dupla.ids_a["site"]],
    )
    recuperacao_atual = entrada["resultado_recuperacao"]
    entrada["resultado_recuperacao"] = recuperacao_atual.model_copy(
        update={
            "documentos": [
                recuperacao_atual.documentos[0].model_copy(
                    update={"id_startup": dupla.id_b}
                )
            ]
        }
    )

    saida = EvidenceValidator(dupla.base)(entrada)
    validada = saida["perfil_validado"].afirmacoes_validadas[0]
    assert validada.situacao == "derrubada"
    assert "recuperado" in validada.motivo


def test_trecho_ausente_do_documento_derruba_a_afirmacao(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["vaga"],
                TRECHO_DADOS,
            )
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[0].situacao == "derrubada"
    assert "literal" in perfil.afirmacoes_validadas[0].motivo


def test_ids_originais_sao_preservados_no_perfil_validado(dupla: BaseDupla):
    afirmacoes = afirmacoes_confirmadas(dupla)
    saida = validar(dupla, afirmacoes=afirmacoes)
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert [item.id_afirmacao for item in perfil.afirmacoes_validadas] == [1, 2]
    assert [item.id_documento for item in perfil.afirmacoes_validadas] == [
        dupla.ids_a["site"],
        dupla.ids_a["vaga"],
    ]
    assert [item.trecho_citado for item in perfil.afirmacoes_validadas] == [
        TRECHO_DADOS,
        TRECHO_OTIMIZACAO,
    ]


# --------------------------------------------------------------------------
# Taxa de derrubada e hosts
# --------------------------------------------------------------------------


def test_taxa_de_derrubada_e_a_fracao_de_afirmacoes_derrubadas(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(2, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(3, "distribuicao", "presenca", dupla.ids_a["vaga"], TRECHO_DISTRIBUICAO),
            afirmacao(4, "workflow_profundo", "presenca", dupla.ids_a["vaga"], TRECHO_DADOS),
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.taxa_derrubada == 0.5


def test_perfil_totalmente_confirmado_tem_taxa_zero(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert PerfilValidado.model_validate(saida["perfil_validado"]).taxa_derrubada == 0.0


def test_hosts_vem_apenas_de_documentos_de_afirmacoes_confirmadas(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(2, "otimizacao_tecnica", "presenca", dupla.ids_a["noticia"], TRECHO_OTIMIZACAO),
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.hosts_distintos == ["acmerobotics.example.com"]


def test_documentos_do_mesmo_host_contam_uma_vez(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(2, "distribuicao", "presenca", dupla.ids_a["site"], TRECHO_DISTRIBUICAO),
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.hosts_distintos == ["acmerobotics.example.com"]


def test_hosts_sao_normalizados_em_minusculas_e_sem_www(dupla: BaseDupla):
    with sqlite3.connect(dupla.caminho) as conexao:
        conexao.execute(
            "UPDATE documentos SET dominio_fonte = ? WHERE id = ?",
            ("WWW.AcmeRobotics.Example.COM", dupla.ids_a["site"]),
        )
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.hosts_distintos == ["acmerobotics.example.com", "vagas.example.org"]


def test_lista_de_hosts_e_deterministica_e_sem_duplicatas(dupla: BaseDupla):
    afirmacoes = [
        afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
        afirmacao(2, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
        afirmacao(3, "distribuicao", "presenca", dupla.ids_a["site"], TRECHO_DISTRIBUICAO),
    ]
    primeira = PerfilValidado.model_validate(
        validar(dupla, afirmacoes=afirmacoes)["perfil_validado"]
    )
    segunda = PerfilValidado.model_validate(
        validar(dupla, afirmacoes=afirmacoes)["perfil_validado"]
    )
    assert primeira.hosts_distintos == segunda.hosts_distintos
    assert primeira.hosts_distintos == ["acmerobotics.example.com", "vagas.example.org"]


# --------------------------------------------------------------------------
# Dimensões de gap
# --------------------------------------------------------------------------


def dimensoes(saida: dict) -> dict[str, tuple[str, list[int]]]:
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    return {
        item.dimensao: (item.estado, item.ids_evidencias)
        for item in perfil.estado_dimensoes_gap
    }


def test_as_quatro_dimensoes_saem_sempre_na_mesma_ordem(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert tuple(item.dimensao for item in perfil.estado_dimensoes_gap) == DIMENSOES_GAP


def test_presenca_confirmada_vira_capacidade_confirmada(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert dimensoes(saida)["dados_proprietarios"] == ("capacidade_confirmada", [1])


def test_ausencia_explicita_confirmada_vira_gap_confirmado(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "workflow_profundo",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_WORKFLOW,
            )
        ],
    )
    assert dimensoes(saida)["workflow_profundo"] == ("gap_confirmado", [1])


def test_dimensao_sem_afirmacao_decisiva_fica_desconhecida(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert dimensoes(saida)["distribuicao"] == ("desconhecido", [])


def test_afirmacao_derrubada_nao_produz_capacidade_nem_gap(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "workflow_profundo",
                "ausencia_explicita",
                dupla.ids_a["site"],
                TRECHO_AUSENCIA_WORKFLOW,
            )
        ],
    )
    assert dimensoes(saida)["workflow_profundo"] == ("desconhecido", [])


def test_neutro_confirmado_nunca_vira_gap(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "distribuicao",
                "neutro",
                dupla.ids_a["site"],
                TRECHO_DISTRIBUICAO,
            )
        ],
    )
    assert dimensoes(saida)["distribuicao"] == ("desconhecido", [])


def test_presenca_e_ausencia_confirmadas_na_mesma_dimensao_ficam_desconhecidas(
    dupla: BaseDupla,
):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_OTIMIZACAO,
            ),
        ],
    )
    assert dimensoes(saida)["otimizacao_tecnica"] == ("desconhecido", [1, 2])


def test_conflito_gera_um_unico_aviso_no_canal_de_erros(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_OTIMIZACAO,
            ),
        ],
        erros=["falha anterior de outro nó"],
    )
    assert len(saida["erros"]) == 1
    aviso = saida["erros"][0]
    assert aviso.startswith("aviso ")
    assert "extração 1" in aviso
    assert "otimizacao_tecnica" in aviso
    assert "falha anterior de outro nó" not in saida["erros"]


def test_aviso_de_conflito_carrega_a_tentativa_de_extracao_que_o_produziu(
    dupla: BaseDupla,
):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_OTIMIZACAO,
            ),
        ],
        tentativas_extracao=2,
    )
    assert "extração 2" in saida["erros"][0]


@pytest.mark.parametrize(
    "trajeto_anterior, validacao_esperada",
    [
        ([], 1),
        (["query_planner", "retriever", "extractor", "classifier"], 1),
        (["extractor", "classifier", "evidence_validator"], 2),
        (
            ["evidence_validator", "extractor", "classifier", "evidence_validator"],
            3,
        ),
    ],
)
def test_numero_da_validacao_vem_do_trajeto_acumulado(
    dupla: BaseDupla, trajeto_anterior: list[str], validacao_esperada: int
):
    """Identificador global determinístico: nada de relógio nem de aleatório."""
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_OTIMIZACAO,
            ),
        ],
        trajeto=trajeto_anterior,
    )
    assert f"validação {validacao_esperada}" in saida["erros"][0]
    # O nó continua devolvendo apenas o próprio passo ao reducer.
    assert saida["trajeto"] == ["evidence_validator"]


def test_sem_conflito_o_no_nao_inventa_aviso(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert saida.get("erros", []) == []


# --------------------------------------------------------------------------
# Confiança do perfil: fórmula fechada
# --------------------------------------------------------------------------


def test_confianca_normal_exige_confirmacao_taxa_zero_e_dois_hosts(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert saida["confianca_perfil"] == "normal"


def test_zero_afirmacoes_confirmadas_rebaixa_a_confianca(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["vaga"], TRECHO_DADOS)
        ],
    )
    assert saida["confianca_perfil"] == "baixa"


def test_uma_unica_derrubada_rebaixa_a_confianca(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(2, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(3, "distribuicao", "presenca", dupla.ids_a["vaga"], TRECHO_DISTRIBUICAO),
        ],
    )
    assert saida["confianca_perfil"] == "baixa"


def test_um_unico_host_rebaixa_a_confianca(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(2, "distribuicao", "presenca", dupla.ids_a["site"], TRECHO_DISTRIBUICAO),
        ],
    )
    assert saida["confianca_perfil"] == "baixa"


def test_conflito_sozinho_nao_altera_a_formula_fechada(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "otimizacao_tecnica", "presenca", dupla.ids_a["vaga"], TRECHO_OTIMIZACAO),
            afirmacao(
                2,
                "otimizacao_tecnica",
                "ausencia_explicita",
                dupla.ids_a["noticia"],
                TRECHO_AUSENCIA_OTIMIZACAO,
            ),
        ],
    )
    assert saida["confianca_perfil"] == "normal"
    assert saida["erros"]


# --------------------------------------------------------------------------
# Fronteiras do nó
# --------------------------------------------------------------------------


def test_no_registra_o_proprio_passo_no_trajeto(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert saida["trajeto"] == ["evidence_validator"]


def test_no_nao_devolve_listas_acumuladas_anteriores(dupla: BaseDupla):
    saida = validar(
        dupla,
        afirmacoes=afirmacoes_confirmadas(dupla),
        trajeto=["query_planner", "retriever", "extractor", "classifier"],
        criterios_relaxados=["estagio"],
    )
    assert saida["trajeto"] == ["evidence_validator"]
    assert "criterios_relaxados" not in saida


def _despejo(saida: dict) -> str:
    serializavel = dict(saida)
    serializavel["perfil_validado"] = saida["perfil_validado"].model_dump()
    return json.dumps(serializavel, default=str, ensure_ascii=False)


def test_no_nunca_coloca_documentos_no_estado(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert set(saida) <= {"perfil_validado", "confianca_perfil", "trajeto", "erros"}
    despejo = _despejo(saida)
    assert "conteudo_texto" not in despejo
    assert TEXTO_SITE_A not in despejo


def test_no_nao_le_classe_referencia(dupla: BaseDupla):
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert "classe_referencia" not in _despejo(saida)
    assert "AI-native" not in _despejo(saida)


def test_suporte_da_classificacao_fora_do_perfil_falha_alto(dupla: BaseDupla):
    with pytest.raises(ErroValidadorEvidencias, match="ausentes do perfil"):
        validar(
            dupla,
            afirmacoes=afirmacoes_confirmadas(dupla),
            ids_suporte=[1, 99],
        )


def test_perfil_ausente_no_estado_falha_alto(dupla: BaseDupla):
    entrada = estado(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    del entrada["perfil_extraido"]
    with pytest.raises(ErroValidadorEvidencias, match="PerfilExtraido"):
        EvidenceValidator(dupla.base)(entrada)


def test_classificacao_ausente_no_estado_falha_alto(dupla: BaseDupla):
    entrada = estado(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    del entrada["classificacao"]
    with pytest.raises(ErroValidadorEvidencias, match="Classificacao"):
        EvidenceValidator(dupla.base)(entrada)


def test_recuperacao_ausente_no_estado_falha_alto(dupla: BaseDupla):
    entrada = estado(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    del entrada["resultado_recuperacao"]
    with pytest.raises(ErroValidadorEvidencias, match="ResultadoRecuperacao"):
        EvidenceValidator(dupla.base)(entrada)


def test_validador_nao_recebe_provedor_de_llm():
    import inspect

    parametros = inspect.signature(EvidenceValidator.__init__).parameters
    assert set(parametros) == {"self", "base"}


# --------------------------------------------------------------------------
# C6 — Fronteira congelada da normalização de proveniência
#
# A normalização é exatamente espaço colapsado + casefold. Toda outra
# diferença entre citação e fonte é diferença real e precisa derrubar.
# --------------------------------------------------------------------------


VARIANTES_ACEITAS = {
    "caixa alta": "BANCO DE IMAGENS ROTULADAS DA ACME ROBOTICS É PROPRIETÁRIO",
    "caixa mista": "BaNcO de Imagens Rotuladas da acme robotics É pRoPrIeTáRiO",
    "espacos repetidos": "banco   de    imagens rotuladas da Acme Robotics é proprietário",
    "quebras de linha e tabulacao": (
        "banco de imagens\n\trotuladas da Acme\nRobotics é proprietário"
    ),
    "espacos nas bordas": "   banco de imagens rotuladas da Acme Robotics é proprietário   ",
}

VARIANTES_RECUSADAS = {
    "virgula inserida": "banco de imagens, rotuladas da Acme Robotics é proprietário",
    "ponto final inserido": "banco de imagens rotuladas da Acme Robotics é proprietário.",
    "hifen inserido": "banco de imagens-rotuladas da Acme Robotics é proprietário",
    "acento removido": "banco de imagens rotuladas da Acme Robotics e proprietario",
    "acento adicionado": "bánco de imagens rotuladas da Acme Robotics é proprietário",
}


@pytest.mark.parametrize("rotulo", sorted(VARIANTES_ACEITAS))
def test_validador_absorve_apenas_caixa_e_espaco(dupla: BaseDupla, rotulo: str):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["site"],
                VARIANTES_ACEITAS[rotulo],
            )
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[0].situacao == "confirmada"
    assert perfil.taxa_derrubada == 0.0


@pytest.mark.parametrize("rotulo", sorted(VARIANTES_RECUSADAS))
def test_validador_nao_absorve_pontuacao_acento_nem_hifen(dupla: BaseDupla, rotulo: str):
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(
                1,
                "dados_proprietarios",
                "presenca",
                dupla.ids_a["site"],
                VARIANTES_RECUSADAS[rotulo],
            )
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[0].situacao == "derrubada"
    assert "literal" in perfil.afirmacoes_validadas[0].motivo
    assert perfil.taxa_derrubada == 1.0


# --------------------------------------------------------------------------
# C1 — O validador é a única autoridade sobre proveniência literal
# --------------------------------------------------------------------------


def test_ausencia_fabricada_e_derrubada_e_nunca_vira_gap(dupla: BaseDupla):
    """Garantia herdada do Extractor: silêncio não vira ausência explícita."""
    saida = validar(
        dupla,
        afirmacoes=[
            afirmacao(1, "dados_proprietarios", "presenca", dupla.ids_a["site"], TRECHO_DADOS),
            afirmacao(
                2,
                "workflow_profundo",
                "ausencia_explicita",
                dupla.ids_a["site"],
                "nenhum documento menciona workflow proprietário de operação",
            ),
        ],
    )
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.afirmacoes_validadas[1].situacao == "derrubada"
    assert dimensoes(saida)["workflow_profundo"] == ("desconhecido", [])
    assert perfil.taxa_derrubada == 0.5
    assert saida["confianca_perfil"] == "baixa"


# --------------------------------------------------------------------------
# C7 — hosts sujos na base e envelope de falha segura
# --------------------------------------------------------------------------


def test_host_com_www_repetido_na_base_nao_quebra_o_validador(dupla: BaseDupla):
    with sqlite3.connect(dupla.caminho) as conexao:
        conexao.execute(
            "UPDATE documentos SET dominio_fonte = ? WHERE id = ?",
            ("WWW.www.AcmeRobotics.Example.COM", dupla.ids_a["site"]),
        )
    saida = validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    perfil = PerfilValidado.model_validate(saida["perfil_validado"])
    assert perfil.hosts_distintos == ["acmerobotics.example.com", "vagas.example.org"]


def test_contrato_final_invalido_vira_erro_seguro_e_encadeado(
    dupla: BaseDupla, monkeypatch
):
    """Nenhuma ValidationError crua escapa do nó, e nenhum perfil parcial sai."""

    def dimensoes_contraditorias(_confirmadas, _tentativa=0, _validacao=1):
        return (
            [
                EstadoDimensaoGap(
                    dimensao=dimensao,
                    estado="gap_confirmado" if indice == 0 else "desconhecido",
                    ids_evidencias=[1] if indice == 0 else [],
                )
                for indice, dimensao in enumerate(DIMENSOES_GAP)
            ],
            [],
        )

    monkeypatch.setattr(
        EvidenceValidator, "_dimensoes_de_gap", staticmethod(dimensoes_contraditorias)
    )
    with pytest.raises(ErroValidadorEvidencias) as capturado:
        validar(dupla, afirmacoes=afirmacoes_confirmadas(dupla))
    assert "perfil validado" in str(capturado.value)
    assert isinstance(capturado.value.__cause__, ValidationError)
