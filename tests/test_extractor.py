import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from radar.agentes.extractor import ErroExtractor, Extractor
from radar.base_startups import BaseStartups, inicializar_banco
from radar.contratos import (
    Classificacao,
    DocumentoRecuperado,
    EmpresaCandidata,
    FiltrosEstruturados,
    PerfilExtraido,
    PerfilValidado,
    PlanoConsulta,
    ResultadoRecuperacao,
)


TEXTO_SITE = (
    "A Acme Robotics vende inspeção visual automatizada para linhas de montagem industriais. "
    "A plataforma processa 4 milhões de imagens por mês em 12 fábricas clientes no Brasil. "
    "O banco de imagens rotuladas da Acme Robotics é proprietário e foi construído junto dos "
    "clientes ao longo de seis anos de operação."
)
TEXTO_VAGA = (
    "A Acme Robotics contrata pessoa engenheira de machine learning para o time de percepção. "
    "A vaga pede experiência em CUDA e otimização de inferência em GPUs, além de rotina de "
    "MLOps com monitoramento contínuo de modelos em produção."
)
TEXTO_NOTICIA = (
    "Reportagem sobre o setor de visão industrial no Brasil menciona a Acme Robotics. "
    "Segundo a matéria, a empresa não utiliza qualquer otimização própria de inferência e "
    "depende integralmente de uma API externa para os casos de linguagem natural."
)

TRECHO_DADOS = "O banco de imagens rotuladas da Acme Robotics é proprietário"
TRECHO_ESCALA = "processa 4 milhões de imagens por mês em 12 fábricas clientes"
TRECHO_EQUIPE = "experiência em CUDA e otimização de inferência em GPUs"
TRECHO_AUSENCIA = "não utiliza qualquer otimização própria de inferência"


@dataclass(frozen=True)
class BaseControlada:
    base: BaseStartups
    id_startup: int
    ids: dict[str, int]
    textos: dict[str, str]


def _curadoria(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    registro = {
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
                "conteudo_texto": TEXTO_SITE,
                "url_fonte": "https://acmerobotics.example.com/produto",
                "dominio_fonte": "acmerobotics.example.com",
                "data_publicacao": "2026-03-10",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "vaga",
                "titulo": "Engenharia de machine learning",
                "conteudo_texto": TEXTO_VAGA,
                "url_fonte": "https://vagas.example.org/acme-robotics-ml",
                "dominio_fonte": "vagas.example.org",
                "data_publicacao": "2026-06-01",
                "data_acesso": "2026-08-30",
            },
            {
                "tipo": "notícia",
                "titulo": "Visão industrial cresce no país",
                "conteudo_texto": TEXTO_NOTICIA,
                "url_fonte": "https://jornal.example.net/visao-industrial",
                "dominio_fonte": "jornal.example.net",
                "data_publicacao": "2026-07-15",
                "data_acesso": "2026-08-30",
            },
        ],
    }
    with (destino / "01_acme_robotics.json").open("w", encoding="utf-8") as saida:
        json.dump(registro, saida, ensure_ascii=False)


@pytest.fixture
def controlada(tmp_path: Path) -> BaseControlada:
    curadoria = tmp_path / "curadoria"
    _curadoria(curadoria)
    banco = tmp_path / "controlada.db"
    inicializar_banco(banco, curadoria)
    with sqlite3.connect(banco) as conexao:
        conexao.row_factory = sqlite3.Row
        id_startup = conexao.execute("SELECT id FROM startups").fetchone()[0]
        linhas = conexao.execute(
            "SELECT id, tipo, conteudo_texto FROM documentos ORDER BY id"
        ).fetchall()
    apelidos = {"site institucional": "site", "vaga": "vaga", "notícia": "noticia"}
    ids = {apelidos[linha["tipo"]]: linha["id"] for linha in linhas}
    textos = {apelidos[linha["tipo"]]: linha["conteudo_texto"] for linha in linhas}
    return BaseControlada(BaseStartups(banco), id_startup, ids, textos)


def plano() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor="Indústria"),
        termos_busca=["inspeção visual"],
        sinais_ia=["inferência"],
        foco_analise="uso de visão computacional em chão de fábrica",
    )


def recuperacao(controlada: BaseControlada, apelidos=("site", "vaga", "noticia")):
    documentos = [
        DocumentoRecuperado(
            id_documento=controlada.ids[apelido],
            id_startup=controlada.id_startup,
            tipo="site institucional" if apelido == "site" else "vaga",
            titulo=f"documento {apelido}",
            url_fonte=f"https://exemplo.test/{apelido}",
            dominio_fonte="exemplo.test",
            data_acesso="2026-08-30",
            score_bm25=-1.0,
        )
        for apelido in apelidos
    ]
    empresa = EmpresaCandidata(
        id_startup=controlada.id_startup,
        nome="Acme Robotics",
        setor="Indústria",
        estagio="série A",
        localizacao="Campinas, SP",
        descricao_curta="Inspeção visual automatizada para linhas de montagem.",
    )
    return ResultadoRecuperacao(
        empresas=[empresa], documentos=documentos, filtros_aplicados=plano().filtros
    )


def estado(controlada: BaseControlada, **ajustes):
    base_estado = {
        "consulta_usuario": "startups de visão computacional industrial",
        "startup_selecionada": controlada.id_startup,
        "plano_consulta": plano(),
        "resultado_recuperacao": recuperacao(controlada),
    }
    base_estado.update(ajustes)
    return base_estado


def perfil_valido(controlada: BaseControlada, **ajustes) -> dict:
    bruto = {
        "id_startup": controlada.id_startup,
        "resumo_produto": (
            "A Acme Robotics vende inspeção visual automatizada para linhas de montagem. "
            "A empresa atende doze fábricas clientes no Brasil."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A Acme Robotics mantém um banco proprietário de imagens rotuladas.",
                "categoria": "dados_proprietarios",
                "polaridade": "presenca",
                "id_documento": controlada.ids["site"],
                "trecho_citado": TRECHO_DADOS,
            },
            {
                "id_afirmacao": 2,
                "texto": "A plataforma processa 4 milhões de imagens por mês.",
                "categoria": "escala_e_dor_operacional",
                "polaridade": "neutro",
                "id_documento": controlada.ids["site"],
                "trecho_citado": TRECHO_ESCALA,
            },
            {
                "id_afirmacao": 3,
                "texto": "A empresa contrata engenharia de machine learning com CUDA.",
                "categoria": "equipe_e_contratacao",
                "polaridade": "neutro",
                "id_documento": controlada.ids["vaga"],
                "trecho_citado": TRECHO_EQUIPE,
            },
        ],
    }
    bruto.update(ajustes)
    return bruto


class ProvedorSequencial:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas.append(mensagens)
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    @property
    def ultimo_prompt(self) -> str:
        return "\n".join(texto for _, texto in self.chamadas[-1])


# --------------------------------------------------------------------------
# Extração válida e proveniência literal
# --------------------------------------------------------------------------


def test_extracao_valida_preserva_proveniencia_literal(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))

    perfil = resultado["perfil_extraido"]
    assert isinstance(perfil, PerfilExtraido)
    assert perfil.id_startup == controlada.id_startup
    assert [afirmacao.id_afirmacao for afirmacao in perfil.afirmacoes] == [1, 2, 3]
    assert resultado["tentativas_extracao"] == 1
    assert resultado["trajeto"] == ["extractor"]
    assert len(provedor.chamadas) == 1

    apelido_por_id = {valor: chave for chave, valor in controlada.ids.items()}
    for afirmacao in perfil.afirmacoes:
        armazenado = controlada.textos[apelido_por_id[afirmacao.id_documento]]
        assert afirmacao.trecho_citado in armazenado


def test_tentativas_de_extracao_sao_incrementadas(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    resultado = Extractor(controlada.base, provedor)(
        estado(controlada, tentativas_extracao=1)
    )
    assert resultado["tentativas_extracao"] == 2


def test_documentos_completos_nao_entram_no_estado(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    assert set(resultado) == {
        "perfil_extraido",
        "tentativas_extracao",
        "classificacao",
        "perfil_validado",
        "confianca_perfil",
        "contexto_nvidia",
        "recomendacoes",
        "fit_score",
        "briefing",
        "trajeto",
    }
    assert resultado["classificacao"] is None
    assert resultado["perfil_validado"] is None
    assert resultado["confianca_perfil"] is None
    assert resultado["contexto_nvidia"] is None
    assert resultado["recomendacoes"] is None
    assert resultado["fit_score"] is None
    assert resultado["briefing"] is None
    serializado = json.dumps(
        {
            "perfil_extraido": resultado["perfil_extraido"].model_dump(),
            "tentativas_extracao": resultado["tentativas_extracao"],
            "trajeto": resultado["trajeto"],
        },
        ensure_ascii=False,
    )
    assert TEXTO_SITE not in serializado
    assert "conteudo_texto" not in serializado


def test_extractor_nao_julga_literalidade_do_trecho_citado(controlada):
    """A autoridade sobre proveniência literal é o Evidence Validator.

    Se o Extractor também derrubasse aqui, nenhuma afirmação não literal
    chegaria ao validador, ``taxa_derrubada`` seria estruturalmente zero e o
    laço R2 viraria código morto.
    """
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes[0]["trecho_citado"] = "a empresa possui um acervo exclusivo de imagens"
    provedor = ProvedorSequencial(perfil_valido(controlada, afirmacoes=afirmacoes))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    assert len(provedor.chamadas) == 1
    assert (
        resultado["perfil_extraido"].afirmacoes[0].trecho_citado
        == "a empresa possui um acervo exclusivo de imagens"
    )


# --------------------------------------------------------------------------
# Somente os documentos recuperados
# --------------------------------------------------------------------------


def test_apenas_os_documentos_recuperados_alimentam_o_llm(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    Extractor(controlada.base, provedor)(
        estado(controlada, resultado_recuperacao=recuperacao(controlada, ("site", "vaga")))
    )
    prompt = provedor.ultimo_prompt
    assert TEXTO_SITE in prompt
    assert TEXTO_VAGA in prompt
    assert TEXTO_NOTICIA not in prompt
    assert f"{controlada.ids['site']}" in prompt
    assert f"documento {controlada.ids['noticia']}" not in prompt


def test_documento_fora_do_conjunto_permitido_e_rejeitado(controlada):
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes[0]["id_documento"] = controlada.ids["noticia"]
    afirmacoes[0]["trecho_citado"] = TRECHO_AUSENCIA
    afirmacoes[0]["categoria"] = "otimizacao_tecnica"
    afirmacoes[0]["polaridade"] = "ausencia_explicita"
    invalido = perfil_valido(controlada, afirmacoes=afirmacoes)
    provedor = ProvedorSequencial(invalido, invalido)
    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(
            estado(controlada, resultado_recuperacao=recuperacao(controlada, ("site", "vaga")))
        )
    assert len(provedor.chamadas) == 2
    assert "fora do conjunto permitido" in provedor.chamadas[1][-1][1]


def test_prompt_identifica_a_startup_o_foco_e_proibe_ausencia_inferida(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    Extractor(controlada.base, provedor)(estado(controlada))
    prompt = provedor.ultimo_prompt
    assert "Acme Robotics" in prompt
    assert "uso de visão computacional em chão de fábrica" in prompt
    assert "Silêncio não é ausência" in prompt
    assert "ausencia_explicita" in prompt
    assert "'neutro' quando o documento citar o tema" in prompt
    assert "12 a 300 caracteres" in prompt
    assert "ao menos 3 palavras" in prompt
    assert "literal" in prompt


def test_o_extractor_nao_expoe_classe_referencia(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    Extractor(controlada.base, provedor)(estado(controlada))
    prompt = provedor.ultimo_prompt
    assert "classe_referencia" not in prompt
    assert "AI-native" not in prompt


# --------------------------------------------------------------------------
# Retry único e falha segura
# --------------------------------------------------------------------------


def test_correcao_bem_sucedida_na_unica_tentativa_permitida(controlada):
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes[1]["id_afirmacao"] = 5
    invalido = perfil_valido(controlada, afirmacoes=afirmacoes)
    provedor = ProvedorSequencial(invalido, perfil_valido(controlada))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    assert len(provedor.chamadas) == 2
    assert "Falha de validação" in provedor.chamadas[1][-1][1]
    assert [a.id_afirmacao for a in resultado["perfil_extraido"].afirmacoes] == [1, 2, 3]


def test_erro_pydantic_lancado_pelo_adaptador_tambem_usa_o_retry(controlada):
    try:
        PerfilExtraido.model_validate(
            {"id_startup": controlada.id_startup, "afirmacoes": []}
        )
    except ValidationError as erro_validacao:
        falha_do_adaptador = erro_validacao
    else:  # pragma: no cover - proteção contra alteração acidental do contrato
        raise AssertionError("o perfil inválido deveria produzir ValidationError")

    provedor = ProvedorSequencial(falha_do_adaptador, perfil_valido(controlada))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))

    assert len(provedor.chamadas) == 2
    assert "Falha de validação" in provedor.chamadas[1][-1][1]
    assert resultado["perfil_extraido"].id_startup == controlada.id_startup


def test_duas_respostas_invalidas_falham_com_seguranca(controlada):
    invalido = {"id_startup": controlada.id_startup, "afirmacoes": []}
    provedor = ProvedorSequencial(invalido, invalido)
    entrada = estado(controlada)
    with pytest.raises(ErroExtractor, match="duas vezes fora do contrato"):
        Extractor(controlada.base, provedor)(entrada)
    assert len(provedor.chamadas) == 2
    assert "perfil_extraido" not in entrada


def test_ids_de_afirmacao_nao_sequenciais_sao_rejeitados(controlada):
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes[2]["id_afirmacao"] = 9
    invalido = perfil_valido(controlada, afirmacoes=afirmacoes)
    provedor = ProvedorSequencial(invalido, invalido)
    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))
    assert "sequencial" in provedor.chamadas[1][-1][1]


def test_id_de_startup_divergente_e_rejeitado(controlada):
    invalido = perfil_valido(controlada, id_startup=controlada.id_startup + 41)
    provedor = ProvedorSequencial(invalido, invalido)
    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))
    assert len(provedor.chamadas) == 2
    assert "id_startup" in provedor.chamadas[1][-1][1]


def test_trecho_nao_literal_nao_consome_o_retry_corretivo(controlada):
    """Retry corretivo existe para violação de contrato, não para proveniência."""
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes[0]["trecho_citado"] = "a empresa possui um acervo exclusivo de imagens"
    invalido = perfil_valido(controlada, afirmacoes=afirmacoes)
    provedor = ProvedorSequencial(invalido, invalido)
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    assert len(provedor.chamadas) == 1
    assert resultado["perfil_extraido"].id_startup == controlada.id_startup


def test_saida_estruturada_malformada_e_rejeitada(controlada):
    provedor = ProvedorSequencial("perfil em texto livre", {"afirmacoes": "três fatos"})
    with pytest.raises(ErroExtractor):
        Extractor(controlada.base, provedor)(estado(controlada))
    assert len(provedor.chamadas) == 2


def test_falha_do_provedor_nao_fabrica_perfil(controlada):
    provedor = ProvedorSequencial(ConnectionError("segredo que não deve aparecer"))
    entrada = estado(controlada)
    with pytest.raises(ErroExtractor) as falha:
        Extractor(controlada.base, provedor)(entrada)
    assert "segredo" not in str(falha.value)
    assert "perfil_extraido" not in entrada
    assert len(provedor.chamadas) == 1


# --------------------------------------------------------------------------
# Desconhecido não vira ausência
# --------------------------------------------------------------------------


def test_ausencia_explicita_com_evidencia_literal_e_aceita(controlada):
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes.append(
        {
            "id_afirmacao": 4,
            "texto": "A empresa declara não ter otimização própria de inferência.",
            "categoria": "otimizacao_tecnica",
            "polaridade": "ausencia_explicita",
            "id_documento": controlada.ids["noticia"],
            "trecho_citado": TRECHO_AUSENCIA,
        }
    )
    provedor = ProvedorSequencial(perfil_valido(controlada, afirmacoes=afirmacoes))
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    gap = resultado["perfil_extraido"].afirmacoes[3]
    assert gap.polaridade == "ausencia_explicita"
    assert gap.trecho_citado in controlada.textos["noticia"]
    assert len(provedor.chamadas) == 1


def test_silencio_nao_vira_ausencia_explicita_mas_o_extractor_nao_derruba(controlada):
    afirmacoes = perfil_valido(controlada)["afirmacoes"]
    afirmacoes.append(
        {
            "id_afirmacao": 4,
            "texto": "A empresa não possui dados proprietários de linguagem.",
            "categoria": "dados_proprietarios",
            "polaridade": "ausencia_explicita",
            "id_documento": controlada.ids["vaga"],
            "trecho_citado": "nenhum documento menciona dados proprietários de linguagem",
        }
    )
    fabricado = perfil_valido(controlada, afirmacoes=afirmacoes)
    provedor = ProvedorSequencial(fabricado)
    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    # O Extractor não julga a fabricação: ele a proíbe no prompt e delega a
    # conferência ao validador, que derruba a afirmação sem lastro literal.
    assert len(provedor.chamadas) == 1
    assert "Silêncio não é ausência" in provedor.ultimo_prompt
    assert resultado["perfil_extraido"].afirmacoes[3].polaridade == "ausencia_explicita"


# --------------------------------------------------------------------------
# Pré-condições do nó
# --------------------------------------------------------------------------


def test_recuperacao_ausente_interrompe_o_no(controlada):
    provedor = ProvedorSequencial(perfil_valido(controlada))
    with pytest.raises(ErroExtractor, match="ResultadoRecuperacao"):
        Extractor(controlada.base, provedor)({"plano_consulta": plano()})
    assert provedor.chamadas == []


def test_startup_ambigua_interrompe_o_no(controlada):
    outra = EmpresaCandidata(
        id_startup=controlada.id_startup + 1,
        nome="Outra",
        setor="Indústria",
        estagio="seed",
        localizacao=None,
        descricao_curta=None,
    )
    resultado = recuperacao(controlada)
    ambiguo = ResultadoRecuperacao(
        empresas=[*resultado.empresas, outra],
        documentos=resultado.documentos,
        filtros_aplicados=resultado.filtros_aplicados,
    )
    provedor = ProvedorSequencial(perfil_valido(controlada))
    entrada = estado(controlada, resultado_recuperacao=ambiguo)
    del entrada["startup_selecionada"]
    with pytest.raises(ErroExtractor, match="uma startup por invocação"):
        Extractor(controlada.base, provedor)(entrada)
    assert provedor.chamadas == []


def test_startup_sem_documentos_recuperados_interrompe_o_no(controlada):
    resultado = recuperacao(controlada)
    sem_documentos = ResultadoRecuperacao(
        empresas=resultado.empresas,
        documentos=[],
        filtros_aplicados=resultado.filtros_aplicados,
    )
    provedor = ProvedorSequencial(perfil_valido(controlada))
    with pytest.raises(ErroExtractor, match="nenhum documento"):
        Extractor(controlada.base, provedor)(
            estado(controlada, resultado_recuperacao=sem_documentos)
        )
    assert provedor.chamadas == []


def test_recuperacao_com_documento_de_outra_startup_falha_sem_filtrar(controlada):
    resultado = recuperacao(controlada)
    documento = resultado.documentos[0]
    invasor = documento.model_copy(
        update={"id_startup": controlada.id_startup + 1}
    )
    inconsistente = ResultadoRecuperacao(
        empresas=resultado.empresas,
        documentos=[invasor, *resultado.documentos[1:]],
        filtros_aplicados=resultado.filtros_aplicados,
    )
    provedor = ProvedorSequencial(perfil_valido(controlada))

    with pytest.raises(ErroExtractor, match="outra startup"):
        Extractor(controlada.base, provedor)(
            estado(controlada, resultado_recuperacao=inconsistente)
        )

    assert provedor.chamadas == []


# --------------------------------------------------------------------------
# C5 — Toda reextração de R2 é estrita: um único predicado compartilhado
# --------------------------------------------------------------------------


def perfil_validado_bruto(situacoes: list[str]) -> dict:
    afirmacoes = [
        {
            "id_afirmacao": indice,
            "texto": f"Fato número {indice} sobre a empresa analisada.",
            "categoria": "dados_proprietarios",
            "polaridade": "presenca",
            "id_documento": 101,
            "trecho_citado": "banco proprietário de imagens rotuladas",
            "situacao": situacao,
            "motivo": None if situacao == "confirmada" else "o trecho não é literal",
        }
        for indice, situacao in enumerate(situacoes, start=1)
    ]
    confirmados = [
        indice for indice, situacao in enumerate(situacoes, start=1)
        if situacao == "confirmada"
    ]
    return {
        "afirmacoes_validadas": afirmacoes,
        "taxa_derrubada": (len(situacoes) - len(confirmados)) / len(situacoes),
        "hosts_distintos": ["acme.example.com"] if confirmados else [],
        "estado_dimensoes_gap": [
            {
                "dimensao": "dados_proprietarios",
                "estado": "capacidade_confirmada" if confirmados else "desconhecido",
                "ids_evidencias": confirmados,
            },
            {"dimensao": "workflow_profundo", "estado": "desconhecido", "ids_evidencias": []},
            {"dimensao": "distribuicao", "estado": "desconhecido", "ids_evidencias": []},
            {"dimensao": "otimizacao_tecnica", "estado": "desconhecido", "ids_evidencias": []},
        ],
    }


def classificacao_com_suporte(ids: list[int]) -> dict:
    return {
        "classe": "AI-native",
        "justificativa": (
            "A empresa treina modelos próprios de detecção. "
            "Sem esses modelos não resta produto para o cliente."
        ),
        "ids_afirmacoes_suporte": ids,
    }


def test_modo_estrito_liga_exatamente_na_metade_derrubada():
    estado_meio = {
        "perfil_validado": perfil_validado_bruto(["confirmada", "derrubada"]),
        "classificacao": classificacao_com_suporte([1]),
    }
    assert Extractor._modo_estrito(estado_meio) is True


def test_modo_estrito_nao_liga_abaixo_da_metade_com_suporte_confirmado():
    estado_abaixo = {
        "perfil_validado": perfil_validado_bruto(
            ["confirmada", "derrubada", "confirmada"]
        ),
        "classificacao": classificacao_com_suporte([1, 3]),
    }
    assert Extractor._modo_estrito(estado_abaixo) is False


def test_modo_estrito_liga_com_suporte_derrubado_abaixo_do_limiar():
    """Toda volta ao Extractor é estrita, inclusive a motivada pelo suporte."""
    estado_suporte = {
        "perfil_validado": perfil_validado_bruto(
            ["confirmada", "derrubada", "confirmada"]
        ),
        "classificacao": classificacao_com_suporte([2]),
    }
    assert Extractor._modo_estrito(estado_suporte) is True


def test_primeira_extracao_nao_usa_modo_estrito(controlada):
    assert Extractor._modo_estrito({}) is False
    provedor = ProvedorSequencial(perfil_valido(controlada))
    Extractor(controlada.base, provedor)(estado(controlada))
    assert "REEXTRAÇÃO ESTRITA" not in provedor.ultimo_prompt


def test_perfil_validado_anterior_sem_classificacao_falha_com_seguranca():
    entrada = {
        "perfil_validado": perfil_validado_bruto(["confirmada", "derrubada"]),
        "classificacao": None,
    }

    with pytest.raises(ErroExtractor, match="exige a classificação correspondente"):
        Extractor._modo_estrito(entrada)


def test_modo_estrito_e_o_mesmo_predicado_de_r2():
    from radar.agentes.roteadores import precisa_reextrair, rotear_r2

    casos = (
        (["confirmada", "derrubada"], [1]),
        (["confirmada", "derrubada", "confirmada"], [1, 3]),
        (["confirmada", "derrubada", "confirmada"], [2]),
        (["derrubada"], [1]),
    )
    for situacoes, suporte in casos:
        estado_caso = {
            "perfil_validado": perfil_validado_bruto(situacoes),
            "classificacao": classificacao_com_suporte(suporte),
            "tentativas_extracao": 1,
        }
        esperado = rotear_r2(estado_caso) == "reextrair"
        assert Extractor._modo_estrito(estado_caso) is esperado
        assert (
            precisa_reextrair(
                PerfilValidado.model_validate(estado_caso["perfil_validado"]),
                Classificacao.model_validate(estado_caso["classificacao"]),
            )
            is esperado
        )
