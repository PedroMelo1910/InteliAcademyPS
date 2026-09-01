import json
import sqlite3

import pytest

from radar.agentes.classifier import ErroClassificador
from radar.agentes.extractor import ErroExtractor
from radar.agentes.roteadores import rotear_r1, rotear_r3
from radar.aplicacao import criar_aplicacao
from radar.base_startups import BaseStartups
from radar.configuracao import ErroConfiguracao
from radar.contratos import (
    Classificacao,
    DocumentoVerificavel,
    FiltrosEstruturados,
    PerfilValidado,
    PlanoConsulta,
)
from radar.grafo import montar_grafo


class ProvedorFixo:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        return self.resposta


class BaseComPrimeiraVerificacaoCorrompida(BaseStartups):
    """Simula corrupção na primeira releitura sem alterar o SQLite do teste."""

    def __init__(self, caminho_banco):
        super().__init__(caminho_banco)
        self.verificacoes = 0

    def carregar_documentos_verificaveis(self, ids_documentos):
        documentos = super().carregar_documentos_verificaveis(ids_documentos)
        self.verificacoes += 1
        if self.verificacoes != 1:
            return documentos
        return {
            id_documento: DocumentoVerificavel(
                id_documento=documento.id_documento,
                id_startup=documento.id_startup,
                conteudo_texto="Conteúdo deliberadamente sem a citação esperada.",
                dominio_fonte=documento.dominio_fonte,
            )
            for id_documento, documento in documentos.items()
        }


def plano_caju():
    return PlanoConsulta(
        filtros=FiltrosEstruturados(setor="Fintech / RH"),
        termos_busca=["benefícios", "cartão"],
        sinais_ia=[],
        foco_analise="plataforma de benefícios corporativos",
    )


def plano_sem_resultado():
    return PlanoConsulta(
        filtros=FiltrosEstruturados(),
        termos_busca=["termo-inexistente-para-provar-rota"],
        sinais_ia=[],
        foco_analise="consulta sem correspondência na base",
    )


def perfil_caju(caminho_banco):
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        linha = conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.conteudo_texto
            FROM startups s
            JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = 'Caju'
            ORDER BY d.id
            LIMIT 1
            """
        ).fetchone()
    return {
        "id_startup": linha["id_startup"],
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A fonte apresenta informações sobre o produto da Caju.",
                "categoria": "outro",
                "polaridade": "neutro",
                "id_documento": linha["id_documento"],
                "trecho_citado": linha["conteudo_texto"][:200],
            }
        ],
    }


def classificacao_caju():
    return {
        "classe": "AI-enabled",
        "justificativa": (
            "A plataforma de benefícios corporativos é o produto contratado pelas empresas. "
            "O perfil não descreve modelos próprios como aquilo que a Caju vende."
        ),
        "ids_afirmacoes_suporte": [1],
    }


def provedores(plano, perfil=None, classificacao=None):
    """Os três provedores da injeção offline, na ordem da fábrica."""
    return (
        ProvedorFixo(plano),
        ProvedorFixo(perfil if perfil is not None else RuntimeError("não deve ser chamado")),
        ProvedorFixo(
            classificacao if classificacao is not None else RuntimeError("não deve ser chamado")
        ),
    )


def estado_selecionado(id_startup, **ajustes):
    base = {
        "consulta_usuario": "detalhar Caju",
        "startup_selecionada": id_startup,
        "plano_consulta": plano_caju(),
        "tentativas_relaxamento": 0,
        "tentativas_extracao": 0,
        "criterios_relaxados": [],
        "erros": [],
        "trajeto": [],
    }
    base.update(ajustes)
    return base


def test_ranking_da_aplicacao_recebe_resultado_real_do_retriever(
    tmp_path, caminho_banco
):
    provedor, provedor_extracao, provedor_classificacao = provedores(plano_caju())
    checkpoints = tmp_path / "checkpoints.db"
    aplicacao = criar_aplicacao(
        provedor, caminho_banco, checkpoints, provedor_extracao, provedor_classificacao
    )

    saida = aplicacao.executar_descoberta("fintech brasileira de benefícios com cartão")

    assert provedor.chamadas == 1
    assert provedor_extracao.chamadas == 0
    assert provedor_classificacao.chamadas == 0
    assert saida.rota == "candidatas_prontas"
    assert [item.empresa.nome for item in saida.ranking] == ["Caju"]
    assert saida.ranking[0].documentos
    assert saida.trajeto == ("query_planner", "retriever")
    with sqlite3.connect(checkpoints) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


def test_chave_gemini_ausente_gera_erro_de_configuracao(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("radar.aplicacao.load_dotenv", lambda *_args, **_kwargs: False)

    with pytest.raises(ErroConfiguracao, match="GOOGLE_API_KEY"):
        criar_aplicacao(
            caminho_banco=tmp_path / "radar_sem_chave.db",
            caminho_checkpoints=tmp_path / "checkpoints_sem_chave.db",
        )


def test_caminho_selecionado_percorre_extractor_classifier_e_validator(
    tmp_path, caminho_banco
):
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    provedor, provedor_extracao, provedor_classificacao = provedores(
        RuntimeError("o Query Planner não deve ser chamado"),
        perfil_caju(caminho_banco),
        classificacao_caju(),
    )
    checkpoints = tmp_path / "checkpoints_selecionada.db"
    aplicacao = criar_aplicacao(
        provedor,
        caminho_banco,
        checkpoints,
        provedor_extracao,
        provedor_classificacao,
    )
    estado = aplicacao.grafo.invoke(
        estado_selecionado(id_caju),
        config={"configurable": {"thread_id": "selecionada"}},
    )

    assert rotear_r1(estado) == "analisar"
    assert provedor.chamadas == 0
    assert provedor_extracao.chamadas == 1
    assert provedor_classificacao.chamadas == 1
    assert estado["perfil_extraido"].id_startup == id_caju
    assert estado["tentativas_extracao"] == 1
    assert isinstance(estado["classificacao"], Classificacao)
    assert estado["classificacao"].classe == "AI-enabled"
    assert estado["classificacao"].ids_afirmacoes_suporte == [1]
    assert isinstance(estado["perfil_validado"], PerfilValidado)
    assert estado["confianca_perfil"] == "baixa"
    assert rotear_r3(estado) == "prosseguir"
    assert estado["trajeto"] == [
        "query_planner",
        "retriever",
        "extractor",
        "classifier",
        "evidence_validator",
    ]
    assert "conteudo_texto" not in json.dumps(
        estado["perfil_extraido"].model_dump(), ensure_ascii=False
    )
    with sqlite3.connect(checkpoints) as conexao:
        assert conexao.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


def test_classificacao_sobrevive_ao_checkpoint(tmp_path, caminho_banco):
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    provedor, provedor_extracao, provedor_classificacao = provedores(
        RuntimeError("não deve ser chamado"),
        perfil_caju(caminho_banco),
        classificacao_caju(),
    )
    aplicacao = criar_aplicacao(
        provedor,
        caminho_banco,
        tmp_path / "checkpoints_serializacao.db",
        provedor_extracao,
        provedor_classificacao,
    )
    config = {"configurable": {"thread_id": "serializacao"}}
    aplicacao.grafo.invoke(estado_selecionado(id_caju), config=config)

    recuperado = aplicacao.grafo.get_state(config).values
    assert recuperado["classificacao"].classe == "AI-enabled"
    assert isinstance(recuperado["perfil_validado"], PerfilValidado)
    assert recuperado["confianca_perfil"] == "baixa"
    assert recuperado["trajeto"] == [
        "query_planner",
        "retriever",
        "extractor",
        "classifier",
        "evidence_validator",
    ]


def test_falha_do_classifier_interrompe_o_caminho_sem_fabricar_classificacao(
    tmp_path, caminho_banco
):
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    provedor, provedor_extracao, provedor_classificacao = provedores(
        RuntimeError("não deve ser chamado"),
        perfil_caju(caminho_banco),
        {"classe": "AI-enabled"},
    )
    aplicacao = criar_aplicacao(
        provedor,
        caminho_banco,
        tmp_path / "checkpoints_classifier_invalido.db",
        provedor_extracao,
        provedor_classificacao,
    )
    config = {"configurable": {"thread_id": "classifier-invalido"}}

    with pytest.raises(ErroClassificador, match="nenhuma classificação"):
        aplicacao.grafo.invoke(estado_selecionado(id_caju), config=config)

    assert provedor_classificacao.chamadas == 2
    assert aplicacao.grafo.get_state(config).values.get("classificacao") is None


def test_r2_reexecuta_toda_a_cadeia_em_modo_estrito_e_remove_estado_velho(
    tmp_path, caminho_banco
):
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    provedor_plano, provedor_extracao, provedor_classificacao = provedores(
        RuntimeError("não deve ser chamado"),
        perfil_caju(caminho_banco),
        classificacao_caju(),
    )
    base = BaseComPrimeiraVerificacaoCorrompida(caminho_banco)
    grafo, conexao = montar_grafo(
        base,
        provedor_plano,
        provedor_extracao,
        provedor_classificacao,
        tmp_path / "checkpoints_reextracao.db",
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(
                id_caju,
                classificacao=Classificacao.model_validate(classificacao_caju()),
                perfil_validado={
                    "afirmacoes_validadas": [
                        {
                            **perfil_caju(caminho_banco)["afirmacoes"][0],
                            "situacao": "confirmada",
                            "motivo": None,
                        }
                    ],
                    "taxa_derrubada": 0.0,
                    "hosts_distintos": ["estado-antigo.test"],
                    "estado_dimensoes_gap": [
                        {"dimensao": dimensao, "estado": "desconhecido", "ids_evidencias": []}
                        for dimensao in (
                            "dados_proprietarios",
                            "workflow_profundo",
                            "distribuicao",
                            "otimizacao_tecnica",
                        )
                    ],
                },
                confianca_perfil="normal",
            ),
            config={"configurable": {"thread_id": "reextracao"}},
        )
    finally:
        conexao.close()

    assert base.verificacoes == 2
    assert provedor_extracao.chamadas == 2
    assert provedor_classificacao.chamadas == 2
    assert saida["tentativas_extracao"] == 2
    assert saida["perfil_validado"].taxa_derrubada == 0.0
    assert saida["perfil_validado"].hosts_distintos != ["estado-antigo.test"]
    assert saida["trajeto"] == [
        "query_planner",
        "retriever",
        "extractor",
        "classifier",
        "evidence_validator",
        "extractor",
        "classifier",
        "evidence_validator",
    ]
    segundo_prompt = "\n".join(
        conteudo for _papel, conteudo in provedor_extracao.mensagens[1]
    )
    assert "REEXTRAÇÃO ESTRITA" in segundo_prompt


def test_falha_do_extractor_interrompe_o_caminho_sem_fabricar_perfil(
    tmp_path, caminho_banco
):
    provedor_plano, provedor_extracao, provedor_classificacao = provedores(
        RuntimeError("não deve ser chamado"),
        {"afirmacoes": []},
        RuntimeError("o Classifier não deve ser chamado"),
    )
    aplicacao = criar_aplicacao(
        provedor_plano,
        caminho_banco,
        tmp_path / "checkpoints_extractor_invalido.db",
        provedor_extracao,
        provedor_classificacao,
    )
    id_caju = perfil_caju(caminho_banco)["id_startup"]

    with pytest.raises(ErroExtractor, match="duas vezes fora do contrato"):
        aplicacao.grafo.invoke(
            estado_selecionado(id_caju),
            config={"configurable": {"thread_id": "extractor-invalido"}},
        )

    assert provedor_plano.chamadas == 0
    assert provedor_extracao.chamadas == 2
    assert provedor_classificacao.chamadas == 0


def test_injecao_offline_exige_os_tres_provedores(tmp_path, caminho_banco):
    with pytest.raises(ErroConfiguracao, match="informe juntos"):
        criar_aplicacao(
            ProvedorFixo(plano_caju()),
            caminho_banco,
            tmp_path / "checkpoints_incompletos.db",
        )

    with pytest.raises(ErroConfiguracao, match="informe juntos"):
        criar_aplicacao(
            ProvedorFixo(plano_caju()),
            caminho_banco,
            tmp_path / "checkpoints_sem_classificador.db",
            ProvedorFixo(perfil_caju(caminho_banco)),
        )


def test_grafo_preserva_relaxamento_e_termino_sem_resultado(
    tmp_path, caminho_banco
):
    provedor_plano, provedor_extracao, provedor_classificacao = provedores(
        plano_sem_resultado()
    )
    aplicacao = criar_aplicacao(
        provedor_plano,
        caminho_banco,
        tmp_path / "checkpoints_sem_resultado.db",
        provedor_extracao,
        provedor_classificacao,
    )

    estado = aplicacao.grafo.invoke(
        {
            "consulta_usuario": "consulta deliberadamente sem resultado",
            "startup_selecionada": None,
            "tentativas_relaxamento": 0,
            "tentativas_extracao": 0,
            "criterios_relaxados": [],
            "erros": [],
            "trajeto": [],
        },
        config={"configurable": {"thread_id": "sem-resultado"}},
    )

    assert rotear_r1(estado) == "sem_resultado"
    assert estado["tentativas_relaxamento"] == 2
    assert estado["resultado_recuperacao"].empresas == []
    assert provedor_plano.chamadas == 1
    assert provedor_extracao.chamadas == 0
    assert provedor_classificacao.chamadas == 0
    assert estado["classificacao"] is None
    assert estado["perfil_extraido"] is None
    assert estado["perfil_validado"] is None
    assert estado["confianca_perfil"] is None
    assert estado["trajeto"] == [
        "query_planner",
        "retriever",
        "query_planner",
        "retriever",
        "query_planner",
        "retriever",
    ]


class ProvedorEmSequencia:
    """Provedor offline que muda de resposta a cada chamada do grafo."""

    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        resposta = self.respostas[min(self.chamadas - 1, len(self.respostas) - 1)]
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


def _texto_documento_caju(caminho_banco) -> tuple[int, int, str]:
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        linha = conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.conteudo_texto
            FROM startups s JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = 'Caju' ORDER BY d.id LIMIT 1
            """
        ).fetchone()
    return linha["id_startup"], linha["id_documento"], linha["conteudo_texto"]


def test_evidencia_nao_literal_percorre_r2_ate_o_teto_e_para_em_evidencia_insuficiente(
    tmp_path, caminho_banco
):
    """C1 ponta a ponta: o validador é quem derruba, e R2 deixa de ser código morto."""
    id_caju, id_documento, _texto = _texto_documento_caju(caminho_banco)
    perfil_nao_literal = {
        "id_startup": id_caju,
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A Caju mantém um acervo exclusivo que nenhuma fonte cita.",
                "categoria": "dados_proprietarios",
                "polaridade": "presenca",
                "id_documento": id_documento,
                "trecho_citado": "acervo exclusivo inexistente nos documentos curados",
            }
        ],
    }
    provedor_plano, provedor_extracao, provedor_classificacao = provedores(
        plano_caju(), perfil_nao_literal, classificacao_caju()
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor_plano,
        provedor_extracao,
        provedor_classificacao,
        tmp_path / "checkpoints_nao_literal.db",
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(id_caju),
            config={"configurable": {"thread_id": "nao-literal"}},
        )
    finally:
        conexao.close()

    perfil = saida["perfil_validado"]
    assert perfil.afirmacoes_validadas[0].situacao == "derrubada"
    assert "literal" in perfil.afirmacoes_validadas[0].motivo
    assert perfil.taxa_derrubada == 1.0
    assert saida["confianca_perfil"] == "baixa"
    assert saida["tentativas_extracao"] == 2
    assert provedor_extracao.chamadas == 2
    assert saida["trajeto"].count("evidence_validator") == 2
    assert "REEXTRAÇÃO ESTRITA" in "\n".join(
        conteudo for _papel, conteudo in provedor_extracao.mensagens[1]
    )
    assert rotear_r3(saida) == "evidencia_insuficiente"


def test_conflito_resolvido_por_reextracao_deixa_o_aviso_datado_no_historico(
    tmp_path, caminho_banco
):
    """C3: o aviso antigo continua auditável, mas identificado como de outra tentativa."""
    id_caju, id_documento, texto = _texto_documento_caju(caminho_banco)
    assert len(texto) >= 320, "o documento curado precisa sustentar dois trechos"

    def afirmacao_caju(id_afirmacao, categoria, polaridade, trecho):
        return {
            "id_afirmacao": id_afirmacao,
            "texto": f"A fonte sustenta o fato {id_afirmacao} sobre a Caju.",
            "categoria": categoria,
            "polaridade": polaridade,
            "id_documento": id_documento,
            "trecho_citado": trecho,
        }

    perfil_conflitante = {
        "id_startup": id_caju,
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [
            afirmacao_caju(1, "distribuicao", "presenca", texto[:150]),
            afirmacao_caju(2, "distribuicao", "ausencia_explicita", texto[160:310]),
            afirmacao_caju(
                3, "outro", "neutro", "trecho inexistente em qualquer documento"
            ),
        ],
    }
    perfil_limpo = {
        "id_startup": id_caju,
        "resumo_produto": (
            "A Caju oferece uma plataforma de benefícios corporativos. "
            "A solução atende empresas e seus colaboradores."
        ),
        "afirmacoes": [afirmacao_caju(1, "distribuicao", "presenca", texto[:150])],
    }
    provedor_plano = ProvedorFixo(plano_caju())
    provedor_extracao = ProvedorEmSequencia(perfil_conflitante, perfil_limpo)
    provedor_classificacao = ProvedorEmSequencia(
        {**classificacao_caju(), "ids_afirmacoes_suporte": [3]},
        {**classificacao_caju(), "ids_afirmacoes_suporte": [1]},
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor_plano,
        provedor_extracao,
        provedor_classificacao,
        tmp_path / "checkpoints_conflito.db",
    )
    try:
        saida = grafo.invoke(
            estado_selecionado(id_caju),
            config={"configurable": {"thread_id": "conflito"}},
        )
    finally:
        conexao.close()

    assert saida["tentativas_extracao"] == 2
    assert len(saida["perfil_validado"].afirmacoes_validadas) == 1
    dimensao = next(
        item
        for item in saida["perfil_validado"].estado_dimensoes_gap
        if item.dimensao == "distribuicao"
    )
    assert dimensao.estado == "capacidade_confirmada"
    assert len(saida["erros"]) == 1
    assert saida["erros"][0].startswith("aviso ")
    assert "extração 1" in saida["erros"][0]


def test_nova_descoberta_no_mesmo_thread_nao_herda_a_analise_antiga(
    tmp_path, caminho_banco
):
    """C5 do relatório: retomar um thread não pode ressuscitar análise de outra busca."""
    id_caju = perfil_caju(caminho_banco)["id_startup"]
    provedor_plano, provedor_extracao, provedor_classificacao = provedores(
        plano_caju(), perfil_caju(caminho_banco), classificacao_caju()
    )
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor_plano,
        provedor_extracao,
        provedor_classificacao,
        tmp_path / "checkpoints_reuso.db",
    )
    config = {"configurable": {"thread_id": "reuso"}}
    try:
        primeira = grafo.invoke(estado_selecionado(id_caju), config=config)
        assert primeira["perfil_validado"] is not None
        assert primeira["classificacao"] is not None
        segunda = grafo.invoke(
            {
                "consulta_usuario": "startups de benefícios",
                "startup_selecionada": None,
                "tentativas_relaxamento": 0,
            },
            config=config,
        )
    finally:
        conexao.close()

    assert segunda["resultado_recuperacao"].empresas
    assert segunda["perfil_extraido"] is None
    assert segunda["classificacao"] is None
    assert segunda["perfil_validado"] is None
    assert segunda["confianca_perfil"] is None
    assert segunda["tentativas_extracao"] == 0
    # Os canais acumulados são histórico e não podem ser apagados.
    assert segunda["trajeto"][: len(primeira["trajeto"])] == primeira["trajeto"]
    assert segunda["trajeto"][-2:] == ["query_planner", "retriever"]


# --------------------------------------------------------------------------
# Avisos de conflito precisam ser distinguíveis ao longo do thread
#
# `erros` e `trajeto` acumulam por desenho, mas o Retriever reinicia
# `tentativas_extracao` a cada novo contexto recuperado. Sem um identificador
# global, duas análises diferentes no mesmo thread produzem avisos idênticos e
# um aviso antigo passa a parecer descrever o perfil validado corrente.
# --------------------------------------------------------------------------


def _startup_e_documento(caminho_banco, nome: str) -> tuple[int, int, str]:
    with sqlite3.connect(caminho_banco) as conexao:
        conexao.row_factory = sqlite3.Row
        linha = conexao.execute(
            """
            SELECT s.id AS id_startup, d.id AS id_documento, d.conteudo_texto
            FROM startups s JOIN documentos d ON d.startup_id = s.id
            WHERE s.nome = ? ORDER BY d.id LIMIT 1
            """,
            (nome,),
        ).fetchone()
    return linha["id_startup"], linha["id_documento"], linha["conteudo_texto"]


def _perfil_com_conflito(id_startup: int, id_documento: int, texto: str) -> dict:
    """Presença e ausência explícita confirmadas na mesma dimensão."""
    return {
        "id_startup": id_startup,
        "resumo_produto": (
            "A empresa opera uma plataforma digital para clientes corporativos. "
            "A fonte curada descreve o produto entregue."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A fonte descreve a distribuição do produto da empresa.",
                "categoria": "distribuicao",
                "polaridade": "presenca",
                "id_documento": id_documento,
                "trecho_citado": texto[:150],
            },
            {
                "id_afirmacao": 2,
                "texto": "A mesma fonte registra ausência de distribuição própria.",
                "categoria": "distribuicao",
                "polaridade": "ausencia_explicita",
                "id_documento": id_documento,
                "trecho_citado": texto[160:310],
            },
        ],
    }


def test_duas_analises_no_mesmo_thread_produzem_avisos_de_conflito_distinguiveis(
    tmp_path, caminho_banco
):
    id_a, doc_a, texto_a = _startup_e_documento(caminho_banco, "Caju")
    id_b, doc_b, texto_b = _startup_e_documento(caminho_banco, "Alice")
    assert id_a != id_b
    assert len(texto_a) >= 320 and len(texto_b) >= 320

    provedor_plano = ProvedorFixo(plano_caju())
    provedor_extracao = ProvedorEmSequencia(
        _perfil_com_conflito(id_a, doc_a, texto_a),
        _perfil_com_conflito(id_b, doc_b, texto_b),
    )
    provedor_classificacao = ProvedorFixo(classificacao_caju())
    grafo, conexao = montar_grafo(
        BaseStartups(caminho_banco),
        provedor_plano,
        provedor_extracao,
        provedor_classificacao,
        tmp_path / "checkpoints_dois_conflitos.db",
    )
    config = {"configurable": {"thread_id": "dois-conflitos"}}
    try:
        primeira = grafo.invoke(estado_selecionado(id_a), config=config)
        segunda = grafo.invoke(
            {
                "consulta_usuario": "detalhar a segunda empresa",
                "startup_selecionada": id_b,
                "tentativas_relaxamento": 0,
            },
            config=config,
        )
    finally:
        conexao.close()

    # O contador local reinicia a cada novo contexto recuperado.
    assert primeira["tentativas_extracao"] == 1
    assert segunda["tentativas_extracao"] == 1

    # O histórico acumulado preserva os dois avisos, sem apagar o antigo.
    assert len(primeira["erros"]) == 1
    assert len(segunda["erros"]) == 2
    assert segunda["erros"][0] == primeira["erros"][0]

    # Identificador global de validação: monotônico e distinto entre análises.
    primeiro_aviso, segundo_aviso = segunda["erros"]
    assert primeiro_aviso != segundo_aviso
    assert len(set(segunda["erros"])) == 2
    assert "validação 1" in primeiro_aviso
    assert "validação 2" in segundo_aviso

    # A tentativa local de extração continua registrada em cada aviso.
    assert "extração 1" in primeiro_aviso
    assert "extração 1" in segundo_aviso

    # O perfil validado final descreve apenas a última análise.
    assert segunda["perfil_extraido"].id_startup == id_b
    dimensao = next(
        item
        for item in segunda["perfil_validado"].estado_dimensoes_gap
        if item.dimensao == "distribuicao"
    )
    assert dimensao.estado == "desconhecido"
    assert dimensao.ids_evidencias == [1, 2]

    # Os reducers acumulam sem que nenhum nó devolva a lista anterior inteira.
    assert segunda["trajeto"][: len(primeira["trajeto"])] == primeira["trajeto"]
    assert segunda["trajeto"].count("evidence_validator") == 2
    assert len(segunda["trajeto"]) == 2 * len(primeira["trajeto"])
