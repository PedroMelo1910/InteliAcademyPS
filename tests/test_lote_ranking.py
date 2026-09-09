"""Testa a pré-análise, o cache e a ordenação do ranking persistido."""

from __future__ import annotations

import inspect
import sqlite3
from dataclasses import replace
from datetime import date

import pytest
from pydantic import ValidationError

from radar.aplicacao import ErroAplicacao, construir_ranking, personalizar_ranking
from radar.base_startups import BaseStartups, conectar, preparar_cache_analises
from radar.contratos import (
    AnalisePersistida,
    Classificacao,
    DocumentoRecuperado,
    EmpresaCandidata,
    EstadoDimensaoGap,
    FiltrosEstruturados,
    FitScore,
    PerfilExtraido,
    PerfilValidado,
    PilarFitScore,
    ResultadoRecuperacao,
)
from radar.grafo import montar_grafo_lote
from radar.lote import (
    LIMITE_MENSAGEM_FALHA,
    AnalisadorLote,
    ErroAnaliseLote,
    _mensagem_falha,
    estimar_chamadas_externas,
)


class ProvedorFila:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = 0
        self.mensagens = []

    def invocar(self, mensagens):
        self.chamadas += 1
        self.mensagens.append(mensagens)
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


def _dimensoes():
    return [
        EstadoDimensaoGap(dimensao=dimensao, estado="desconhecido")
        for dimensao in (
            "dados_proprietarios",
            "workflow_profundo",
            "distribuicao",
            "otimizacao_tecnica",
        )
    ]


def _perfil_extraido(base: BaseStartups, startup_id: int, *, literal: bool = True):
    recuperacao = base.recuperar_para_lote(startup_id)
    documento = base.carregar_documentos(
        startup_id, [recuperacao.documentos[0].id_documento]
    )[0]
    trecho = documento.conteudo_texto[:180]
    if not literal:
        trecho = "Este trecho inventado não ocorre em nenhum documento público."
    return PerfilExtraido(
        id_startup=startup_id,
        resumo_produto=(
            "A empresa oferece um produto documentado pelas fontes. "
            "A análise usa somente o conteúdo recuperado."
        ),
        afirmacoes=[
            {
                "id_afirmacao": 1,
                "texto": "A empresa oferece modelos de linguagem documentados.",
                "categoria": "workflow_profundo",
                "polaridade": "presenca",
                "id_documento": documento.id_documento,
                "trecho_citado": trecho,
                "sinais_tecnicos": ["inferencia_llm"],
            }
        ],
    )


def _classificacao(classe: str):
    return Classificacao(
        classe=classe,
        justificativa=(
            "A decisão considera apenas o produto descrito no perfil. "
            "A afirmação selecionada sustenta a classificação."
        ),
        ids_afirmacoes_suporte=[1],
    )


def _analisador(tmp_path, base, extrator, classificador):
    grafo, conexao = montar_grafo_lote(
        base, extrator, classificador, tmp_path / "checkpoints-lote.db"
    )
    return AnalisadorLote(
        base, grafo, conexao, relogio=lambda: date(2026, 9, 4)
    )


@pytest.mark.parametrize("classe", ["AI-native", "AI-enabled"])
def test_lote_persiste_classes_aderentes(tmp_path, base, classe):
    extrator = ProvedorFila(_perfil_extraido(base, 1))
    classificador = ProvedorFila(_classificacao(classe))
    analisador = _analisador(tmp_path, base, extrator, classificador)
    try:
        analise = analisador.executar_startup(1)
    finally:
        analisador.fechar()

    assert analise.status == "concluida"
    assert analise.classe == classe
    assert analise.fit_score is not None
    assert base.carregar_analises([1])[1] == analise


def test_lote_non_ai_persiste_fit_score_zero_real(tmp_path, base):
    analisador = _analisador(
        tmp_path,
        base,
        ProvedorFila(_perfil_extraido(base, 1)),
        ProvedorFila(_classificacao("non-AI")),
    )
    try:
        analise = analisador.executar_startup(1)
    finally:
        analisador.fechar()

    assert analise.status == "concluida"
    assert analise.classe == "non-AI"
    assert analise.fit_score is not None
    assert analise.fit_score.total == 0
    assert all(
        "gate_non_ai" in pilar.travas_aplicadas
        for pilar in analise.fit_score.pilares
    )


def test_lote_reextrai_e_substitui_evidencia_ruim(tmp_path, base):
    extrator = ProvedorFila(
        _perfil_extraido(base, 1, literal=False),
        _perfil_extraido(base, 1, literal=True),
    )
    classificador = ProvedorFila(
        _classificacao("AI-enabled"), _classificacao("AI-enabled")
    )
    analisador = _analisador(tmp_path, base, extrator, classificador)
    try:
        analise = analisador.executar_startup(1)
    finally:
        analisador.fechar()

    assert analise.status == "concluida"
    assert extrator.chamadas == 2
    assert classificador.chamadas == 2
    assert analise.perfil_validado.taxa_derrubada == 0


def test_lote_persiste_insuficiencia_sem_classe_ou_score(tmp_path, base):
    extrator = ProvedorFila(
        _perfil_extraido(base, 1, literal=False),
        _perfil_extraido(base, 1, literal=False),
    )
    classificador = ProvedorFila(
        _classificacao("AI-native"), _classificacao("AI-native")
    )
    analisador = _analisador(tmp_path, base, extrator, classificador)
    try:
        analise = analisador.executar_startup(1)
    finally:
        analisador.fechar()

    assert analise.status == "evidencia_insuficiente"
    assert analise.classe is None
    assert analise.fit_score is None
    assert analise.perfil_validado.taxa_derrubada == 1
    assert "Nenhuma afirmação" in analise.motivo_evidencia_insuficiente


def test_falha_de_provedor_nao_e_gravada_como_insuficiencia(tmp_path, base):
    analisador = _analisador(
        tmp_path,
        base,
        ProvedorFila(TimeoutError("serviço indisponível")),
        ProvedorFila(_classificacao("AI-enabled")),
    )
    try:
        resultado = analisador.executar_todas([1])
    finally:
        analisador.fechar()

    assert not resultado.concluidas
    assert not resultado.evidencias_insuficientes
    assert resultado.falhas[0].tipo == "falha_operacional"
    assert "TimeoutError" in resultado.falhas[0].mensagem
    assert "serviço indisponível" in resultado.falhas[0].mensagem
    assert base.carregar_analises([1]) == {}


def test_status_http_operacional_reutiliza_classificador_do_provedor(tmp_path, base):
    class ErroHttp(RuntimeError):
        status_code = 429

    analisador = _analisador(
        tmp_path,
        base,
        ProvedorFila(ErroHttp("limite atingido")),
        ProvedorFila(_classificacao("AI-enabled")),
    )
    try:
        resultado = analisador.executar_todas([1])
    finally:
        analisador.fechar()

    assert resultado.falhas[0].tipo == "falha_operacional"
    assert "ErroHttp" in resultado.falhas[0].mensagem
    assert "limite atingido" in resultado.falhas[0].mensagem
    assert base.carregar_analises([1]) == {}


def test_lote_rejeita_perfil_de_outra_startup_sem_persistir(tmp_path, base):
    analisador = _analisador(
        tmp_path,
        base,
        ProvedorFila(_perfil_extraido(base, 2)),
        ProvedorFila(_classificacao("AI-enabled")),
    )
    try:
        resultado = analisador.executar_todas([1])
    finally:
        analisador.fechar()

    assert resultado.falhas[0].startup_id == 1
    assert base.carregar_analises([1]) == {}


def test_grafo_de_lote_nao_contem_agentes_desnecessarios(tmp_path, base):
    grafo, conexao = montar_grafo_lote(
        base, ProvedorFila(), ProvedorFila(), tmp_path / "topologia.db"
    )
    try:
        nomes = set(grafo.get_graph().nodes)
    finally:
        conexao.close()

    assert nomes == {
        "__start__",
        "extractor",
        "classifier",
        "evidence_validator",
        "r3",
        "__end__",
    }
    proibidos = {
        "query_planner",
        "retriever",
        "nvidia_rag",
        "recommendation",
        "briefing",
    }
    assert nomes.isdisjoint(proibidos)


def _fit(
    pontos: tuple[int, int, int, int] = (3, 0, 1, 3),
    dimensoes=None,
) -> FitScore:
    nomes = (
        "centralidade_ia",
        "gap_enderecavel",
        "momento",
        "alinhamento_setorial",
    )
    pilares = [
        PilarFitScore(
            pilar=nome,
            pontos=valor,
            faixa="baixa" if valor <= 3 else "media" if valor <= 7 else "alta",
        )
        for nome, valor in zip(nomes, pontos, strict=True)
    ]
    return FitScore(
        total=round(100 * sum(pontos) / 36),
        pilares=pilares,
        estado_dimensoes_gap=dimensoes or _dimensoes(),
        justificativa_curta="Pontuação determinística baseada em evidências.",
        versao_rubrica="rubrica-v1",
    )


def _perfil_validado() -> PerfilValidado:
    return PerfilValidado(
        afirmacoes_validadas=[
            {
                "id_afirmacao": 1,
                "texto": "O produto está documentado pela fonte pública.",
                "categoria": "workflow_profundo",
                "polaridade": "presenca",
                "id_documento": 1,
                "trecho_citado": "Trecho público com evidência literal suficiente.",
                "situacao": "confirmada",
            }
        ],
        taxa_derrubada=0,
        hosts_distintos=["fonte.example"],
        estado_dimensoes_gap=[
            EstadoDimensaoGap(
                dimensao="dados_proprietarios", estado="desconhecido"
            ),
            EstadoDimensaoGap(
                dimensao="workflow_profundo",
                estado="capacidade_confirmada",
                ids_evidencias=[1],
            ),
            EstadoDimensaoGap(dimensao="distribuicao", estado="desconhecido"),
            EstadoDimensaoGap(
                dimensao="otimizacao_tecnica", estado="desconhecido"
            ),
        ],
    )


def _analise(startup_id: int, pontos=(3, 0, 1, 3)) -> AnalisePersistida:
    perfil = _perfil_validado()
    return AnalisePersistida(
        startup_id=startup_id,
        status="concluida",
        classe="AI-enabled",
        fit_score=_fit(pontos, perfil.estado_dimensoes_gap),
        perfil_validado=perfil,
        data_execucao=date(2026, 9, 4),
        versao_rubrica="rubrica-v1",
    )


def _resultado_ranking() -> ResultadoRecuperacao:
    empresas = [
        EmpresaCandidata(
            id_startup=item,
            nome=nome,
            setor="Software",
            estagio="seed",
            localizacao=None,
            descricao_curta=None,
        )
        for item, nome in ((1, "Zulu"), (2, "Beta"), (3, "Gama"), (4, "Alfa"))
    ]
    documentos = [
        DocumentoRecuperado(
            id_documento=item,
            id_startup=item,
            tipo="notícia",
            titulo=f"Documento {item}",
            url_fonte=f"https://fonte-{item}.example/materia",
            dominio_fonte=f"fonte-{item}.example",
            data_acesso=date(2026, 9, 4),
            score_bm25=score,
        )
        for item, score in ((1, -1.0), (2, -0.5), (3, -4.0), (4, -8.0))
    ]
    return ResultadoRecuperacao(
        empresas=empresas,
        documentos=documentos,
        filtros_aplicados=FiltrosEstruturados(),
    )


def test_ranking_prioriza_status_score_e_depois_relevancia():
    insuficiente = AnalisePersistida(
        startup_id=3,
        status="evidencia_insuficiente",
        perfil_validado=_perfil_validado(),
        motivo_evidencia_insuficiente="Suporte da classe não confirmado.",
        data_execucao=date(2026, 9, 4),
        versao_rubrica="rubrica-v1",
    )
    ranking = construir_ranking(
        _resultado_ranking(),
        {
            1: _analise(1, (3, 0, 1, 3)),
            2: _analise(2, (5, 4, 5, 5)),
            3: insuficiente,
        },
    )

    assert [item.empresa.id_startup for item in ranking] == [2, 1, 4, 3]
    assert [item.status_analise for item in ranking] == [
        "concluida",
        "concluida",
        "ausente",
        "evidencia_insuficiente",
    ]
    assert ranking[0].fit_score_total is not None
    assert ranking[2].fit_score_total is None
    assert ranking[3].fit_score_total is None
    assert ranking[0].melhor_score_bm25 == -0.5


def test_ranking_usa_bm25_nome_e_id_como_desempates():
    resultado = _resultado_ranking()
    mesma_analise = {item: _analise(item) for item in (1, 2, 3, 4)}
    ranking = construir_ranking(resultado, mesma_analise)
    assert [item.empresa.id_startup for item in ranking] == [4, 3, 1, 2]

    sem_documentos = resultado.model_copy(update={"documentos": []})
    ranking_nome = construir_ranking(sem_documentos, mesma_analise)
    assert [item.empresa.nome for item in ranking_nome] == ["Alfa", "Beta", "Gama", "Zulu"]


def test_usuario_pode_priorizar_relevancia_sem_alterar_os_scores():
    ranking = construir_ranking(
        _resultado_ranking(), {item: _analise(item) for item in (1, 2, 3, 4)}
    )

    personalizado = personalizar_ranking(ranking, criterio="relevancia")

    assert [item.empresa.id_startup for item in personalizado] == [4, 3, 1, 2]
    assert [item.posicao for item in personalizado] == [1, 2, 3, 4]
    assert {item.empresa.id_startup: item.fit_score_total for item in personalizado} == {
        item.empresa.id_startup: item.fit_score_total for item in ranking
    }


def test_relevancia_preserva_setor_pedido_antes_do_melhor_termo_isolado():
    """Evita que visão computacional no varejo supere saúde numa busca médica."""
    ranking = construir_ranking(
        _resultado_ranking(), {item: _analise(item) for item in (1, 2, 3, 4)}
    )
    por_id = {item.empresa.id_startup: item for item in ranking}
    alice = replace(
        por_id[1],
        empresa=por_id[1].empresa.model_copy(
            update={"nome": "Alice", "setor": "Saúde"}
        ),
        melhor_score_bm25=-4.5,
    )
    wine = replace(
        por_id[2],
        empresa=por_id[2].empresa.model_copy(
            update={"nome": "Wine", "setor": "Varejo e e-commerce de bebidas"}
        ),
        melhor_score_bm25=-6.0,
    )

    personalizado = personalizar_ranking(
        (wine, alice),
        criterio="relevancia",
        consulta="startups de saúde usando visão computacional em imagem médica",
    )

    assert [item.empresa.nome for item in personalizado] == ["Alice", "Wine"]


def test_usuario_pode_filtrar_classe_e_as_posicoes_sao_renumeradas():
    ranking = construir_ranking(
        _resultado_ranking(), {item: _analise(item) for item in (1, 2, 3, 4)}
    )
    misto = tuple(
        replace(
            item,
            classe=(
                "AI-native"
                if item.empresa.id_startup in (1, 3)
                else "AI-enabled"
            ),
        )
        for item in ranking
    )

    filtrado = personalizar_ranking(
        misto, criterio="fit_score", classe="AI-native"
    )

    assert [item.empresa.id_startup for item in filtrado] == [3, 1]
    assert [item.posicao for item in filtrado] == [1, 2]


def test_persistencia_e_idempotente_e_isola_json_invalido(base, caplog):
    primeira = _analise(1)
    segunda = _analise(1, (5, 4, 5, 5))
    vizinha = _analise(2)
    base.salvar_analise(primeira)
    base.salvar_analise(segunda)
    base.salvar_analise(vizinha)
    carregada = base.carregar_analises([1])[1]
    assert carregada == segunda
    assert base.cobertura_analises().concluidas == 2

    with conectar(base.caminho_banco) as conexao:
        conexao.execute(
            "UPDATE analises SET fit_score_json = ? WHERE startup_id = ?",
            ('{"total":"inválido"}', 1),
        )
    carregadas = base.carregar_analises([1, 2])
    assert set(carregadas) == {2}
    assert "startup_id=1" in caplog.text


def test_total_redundante_divergente_tambem_e_isolado(base, caplog):
    analise = _analise(1)
    base.salvar_analise(analise)
    with conectar(base.caminho_banco) as conexao:
        conexao.execute(
            "UPDATE analises SET fit_score_total = ? WHERE startup_id = ?",
            (analise.fit_score.total + 1, 1),
        )
    assert base.carregar_analises([1]) == {}
    assert "fit_score_total diverge" in caplog.text


def test_falha_de_validacao_nao_substitui_registro_anterior(base):
    original = _analise(1)
    base.salvar_analise(original)
    invalida = original.model_dump(mode="python")
    invalida["status"] = "evidencia_insuficiente"
    with pytest.raises(ValidationError):
        base.salvar_analise(invalida)
    assert base.carregar_analises([1])[1] == original


def test_schema_rejeita_motivo_nulo_e_migracao_legada_e_idempotente(base):
    with conectar(base.caminho_banco) as conexao:
        with pytest.raises(sqlite3.IntegrityError):
            conexao.execute(
                """
                INSERT INTO analises (
                    startup_id, status, classe, fit_score_total, fit_score_json,
                    perfil_validado_json, motivo_evidencia_insuficiente,
                    data_execucao, versao_rubrica
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1,
                    "evidencia_insuficiente",
                    None,
                    None,
                    None,
                    "{}",
                    None,
                    "2026-09-04",
                    "rubrica-v1",
                ),
            )
        conexao.execute("DROP TABLE analises")
        conexao.execute(
            """
            CREATE TABLE analises (
                startup_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                classe TEXT,
                fit_score_total INTEGER,
                fit_score_json TEXT,
                perfil_validado_json TEXT NOT NULL,
                data_execucao TEXT NOT NULL,
                versao_rubrica TEXT NOT NULL
            )
            """
        )

    preparar_cache_analises(base.caminho_banco)
    preparar_cache_analises(base.caminho_banco)
    with conectar(base.caminho_banco) as conexao:
        definicao = conexao.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", ("analises",)
        ).fetchone()["sql"]
    assert "motivo_evidencia_insuficiente IS NOT NULL" in definicao


def test_sql_e_parametrizado_e_startups_nao_expoem_gabarito(base):
    assert base.recuperar_para_lote("1 OR 1=1").empresas == []
    assert base.carregar_analises(["1) OR 1=1 --"]) == {}
    empresa = base.listar_startups_para_lote()[0]
    recuperacao = base.recuperar_para_lote(empresa.id_startup)
    assert "classe_referencia" not in empresa.model_dump()
    assert "classe_referencia" not in recuperacao.model_dump()


def test_estimativa_cobre_caminho_feliz_retries_e_reextracao():
    assert estimar_chamadas_externas(30) == (60, 240)
    with pytest.raises(ValueError):
        estimar_chamadas_externas(-1)


def test_runner_nao_importa_agentes_proibidos():
    import radar.lote as modulo

    fonte = inspect.getsource(modulo)
    for proibido in (
        "QueryPlanner",
        "NvidiaRag",
        "Recommendation",
        "AgenteBriefing",
        "classe_referencia",
    ):
        assert proibido not in fonte


SCHEMA_ANALISES_LEGADO = """
CREATE TABLE analises (
    startup_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    classe TEXT,
    fit_score_total INTEGER,
    fit_score_json TEXT,
    perfil_validado_json TEXT NOT NULL,
    data_execucao TEXT NOT NULL,
    versao_rubrica TEXT NOT NULL
)
"""


def _instalar_cache_legado(base: BaseStartups, linhas: int = 0) -> None:
    """Reproduz o cache anterior ao contrato atual, opcionalmente povoado."""
    with conectar(base.caminho_banco) as conexao:
        conexao.execute("DROP TABLE analises")
        conexao.execute(SCHEMA_ANALISES_LEGADO)
        for indice in range(1, linhas + 1):
            conexao.execute(
                """
                INSERT INTO analises (
                    startup_id, status, perfil_validado_json, data_execucao,
                    versao_rubrica
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (indice, "concluida", "{}", "2026-09-04", "rubrica-v1"),
            )


def test_analise_corrompida_some_da_leitura_e_conta_como_ausente(base, caplog):
    """O relatório do operador não pode prometer o que o ranking não entrega."""
    base.salvar_analise(_analise(1))
    base.salvar_analise(_analise(2))
    intacta = base.cobertura_analises()
    assert intacta.concluidas == 2

    with conectar(base.caminho_banco) as conexao:
        conexao.execute(
            "UPDATE analises SET perfil_validado_json = ? WHERE startup_id = ?",
            ("{isto não é json}", 1),
        )

    caplog.clear()
    assert set(base.carregar_analises([1, 2])) == {2}
    assert "startup_id=1" in caplog.text

    cobertura = base.cobertura_analises()
    assert cobertura.total_startups == intacta.total_startups
    assert cobertura.concluidas == 1
    assert cobertura.evidencias_insuficientes == 0
    assert cobertura.ausentes == intacta.ausentes + 1
    assert (
        cobertura.concluidas
        + cobertura.evidencias_insuficientes
        + cobertura.ausentes
        == cobertura.total_startups
    )


def test_total_redundante_divergente_tambem_derruba_a_cobertura(base):
    """A conferência redundante do total vale igual na leitura e na cobertura."""
    analise = _analise(1)
    base.salvar_analise(analise)
    with conectar(base.caminho_banco) as conexao:
        conexao.execute(
            "UPDATE analises SET fit_score_total = ? WHERE startup_id = ?",
            (analise.fit_score.total + 1, 1),
        )
    cobertura = base.cobertura_analises()
    assert base.carregar_analises([1]) == {}
    assert cobertura.concluidas == 0
    assert cobertura.ausentes == cobertura.total_startups


def test_migracao_legada_avisa_com_a_contagem_descartada(base, caplog):
    _instalar_cache_legado(base, linhas=2)
    caplog.clear()

    preparar_cache_analises(base.caminho_banco)

    assert "cache legado de análises será recriado" in caplog.text
    assert "2 linha(s) serão descartadas" in caplog.text
    assert "scripts.analisar_lote" in caplog.text
    with conectar(base.caminho_banco) as conexao:
        definicao = conexao.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", ("analises",)
        ).fetchone()["sql"]
    assert "motivo_evidencia_insuficiente IS NOT NULL" in definicao


def test_schema_atual_nao_avisa_nem_recria_a_tabela(base, caplog):
    base.salvar_analise(_analise(1))
    caplog.clear()

    preparar_cache_analises(base.caminho_banco)
    preparar_cache_analises(base.caminho_banco)

    assert "cache legado" not in caplog.text
    assert base.carregar_analises([1])[1] == _analise(1)


def test_ranking_recusa_concluida_sem_fit_score():
    """Sem `assert`: a invariante precisa continuar viva sob python -O."""
    corrompida = AnalisePersistida.model_construct(
        startup_id=1,
        status="concluida",
        classe="AI-enabled",
        fit_score=None,
        perfil_validado=_perfil_validado(),
        motivo_evidencia_insuficiente=None,
        data_execucao=date(2026, 9, 4),
        versao_rubrica="rubrica-v1",
    )
    with pytest.raises(ErroAplicacao, match="não inventa pontuação"):
        construir_ranking(_resultado_ranking(), {1: corrompida})


def test_mensagem_de_falha_curta_preserva_o_erro_de_dominio():
    erro = ErroAnaliseLote("a startup 7 não existe ou não possui documentos")
    assert _mensagem_falha(erro) == (
        "a startup 7 não existe ou não possui documentos"
    )


@pytest.mark.parametrize(
    "credencial",
    [
        "api_key=AIzaSyD0123456789abcdefghij",
        "chave nvapi-0123456789abcdefghij rejeitada",
        "Authorization: Bearer abcd1234efgh5678",
    ],
)
def test_mensagem_de_falha_censura_credencial(credencial):
    causa = RuntimeError(f"401 credencial recusada; {credencial}")
    try:
        raise ErroAnaliseLote("a pré-análise falhou") from causa
    except ErroAnaliseLote as erro:
        mensagem = _mensagem_falha(erro)

    assert "[credencial omitida]" in mensagem
    for segredo in ("AIzaSyD0123456789abcdefghij", "nvapi-0123456789abcdefghij",
                    "abcd1234efgh5678"):
        assert segredo not in mensagem
    # A informação operacional útil sobrevive à censura.
    assert "401 credencial recusada" in mensagem
    assert "Causa: RuntimeError:" in mensagem


def test_mensagem_de_falha_trunca_corpo_de_terceiro_sem_perder_o_diagnostico():
    causa = RuntimeError("503 serviço indisponível; " + "corpo da resposta " * 300)
    try:
        raise ErroAnaliseLote("a pré-análise falhou") from causa
    except ErroAnaliseLote as erro:
        mensagem = _mensagem_falha(erro)

    assert len(str(causa)) > 5000
    assert len(mensagem) < 2 * LIMITE_MENSAGEM_FALHA
    assert mensagem.startswith("a pré-análise falhou Causa: RuntimeError: ")
    assert "503 serviço indisponível" in mensagem
    assert mensagem.endswith("… [truncado]")


def test_cli_nao_promete_retry_de_transporte():
    """`retries=1` vira `HttpRetryOptions(attempts=1)`: uma tentativa HTTP, só."""
    import scripts.analisar_lote as modulo

    fonte = inspect.getsource(modulo)
    assert "Retries de transporte" not in fonte
    assert "estimar_chamadas_externas" in fonte
