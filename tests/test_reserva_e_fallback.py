"""Reserva Groq: entra na queda do primário, nunca no erro de contrato.

A reserva existe para indisponibilidade e só para isso. Erro de contrato, de
estado ou de programação continua sendo resolvido pelo retry do próprio agente,
porque trocar de provedor não conserta uma resposta fora do schema.

A primeira parte prova a composição em si: quando a reserva é acionada, o teto
de duas chamadas por fronteira, o que nunca aparece em log ou exceção, e como o
lote classifica cada desfecho.

A segunda parte prova a mesma distinção em cada fronteira de agente. HTTP 400
da reserva é falha de contrato e consome a correção estruturada que o nó já
tem — nunca uma terceira tentativa nem um laço próprio; indisponibilidade dos
dois provedores continua operacional e aborta sem fabricar saída. As duas
coisas precisam continuar separadas em Query Planner, Extractor, Classifier,
Recommendation, Briefing e no lote.

Nenhum teste aqui toca a rede: a composição é real, os dois provedores são
dublês roteirizados.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from radar.agentes.briefing import ErroBriefing
from radar.agentes.classifier import Classifier, ErroClassificador
from radar.agentes.extractor import ErroExtractor, Extractor
from radar.agentes.query_planner import ErroQueryPlanner, QueryPlanner
from radar.agentes.recommendation import ErroRecommendation
from radar.lote import _tipo_falha
from radar.provedores import (
    ErroProvedoresIndisponiveis,
    ErroReservaIncompativel,
    ProvedorComFallback,
    compor_com_reserva,
    falha_operacional,
)
from tests.conftest import (
    CORPO_BRUTO_DO_PROVEDOR,
    TERMOS_QUE_A_FALHA_SEGURA_NAO_REPETE,
    exigir_falha_neutra_de_provedor,
)

# Fixtures e fábricas dos módulos irmãos: reaproveitar o payload válido de cada
# fronteira evita reconstruir contratos inteiros só para provar o retry.
from tests.test_extractor import (  # noqa: F401 - `controlada` é fixture
    controlada,
    estado as estado_extractor,
    perfil_valido,
)
from tests.test_classifier import (
    classificacao as classificacao_valida,
    estado as estado_classifier,
    perfil_ai_native,
)
from tests.test_query_planner import plano_valido
from tests.test_recommendation import (
    executar as executar_recommendation,
    lote as lote_de_rascunhos,
    rascunho as rascunho_recomendacao,
)
from tests.test_briefing import (
    estado_normal as estado_briefing,
    montar_no as montar_no_briefing,
    rascunho as rascunho_briefing,
)
from tests.test_lote_ranking import (
    _analisador,
    _analise,
    _classificacao,
    _perfil_extraido,
    ProvedorFila,
)


# ------------------------------------------------------------------------
# A composição ProvedorComFallback
# ------------------------------------------------------------------------

CHAVE_FALSA = "chave-de-reserva-que-nunca-e-usada"


class ProvedorFalso:
    def __init__(self, resposta=None, erro=None):
        self.resposta = resposta if resposta is not None else {"ok": True}
        self.erro = erro
        self.chamadas = 0

    def invocar(self, mensagens):
        self.chamadas += 1
        if self.erro is not None:
            raise self.erro
        return self.resposta


def erro_http(codigo: int, texto: str = "falha"):
    classe = type("ErroHttpFalso", (Exception,), {"status_code": codigo})
    return classe(texto)


def compor(primario, reserva, fronteira="extractor"):
    return ProvedorComFallback(primario, reserva, fronteira=fronteira)


# ----------------------------------------------------------------------
# Quando a reserva NÃO é chamada
# ----------------------------------------------------------------------


def test_sucesso_do_primario_nao_chama_a_reserva():
    primario, reserva = ProvedorFalso({"plano": 1}), ProvedorFalso()

    assert compor(primario, reserva).invocar([]) == {"plano": 1}
    assert primario.chamadas == 1
    assert reserva.chamadas == 0


@pytest.mark.parametrize(
    "erro",
    [
        ValueError("o rascunho viola o contrato"),
        TypeError("estado malformado"),
        AttributeError("defeito de programação"),
        erro_http(400, "INVALID_ARGUMENT: schema inválido"),
    ],
)
def test_falha_nao_operacional_nao_aciona_a_reserva(erro):
    primario, reserva = ProvedorFalso(erro=erro), ProvedorFalso()

    with pytest.raises(type(erro)):
        compor(primario, reserva).invocar([])

    assert reserva.chamadas == 0
    # a composição também não retenta o primário: quem corrige contrato é o nó
    assert primario.chamadas == 1


def test_erro_de_validacao_pydantic_nao_aciona_a_reserva():
    from radar.contratos import Classificacao

    try:
        Classificacao.model_validate({"classe": "inexistente"})
    except ValidationError as erro:
        validacao = erro
    primario, reserva = ProvedorFalso(erro=validacao), ProvedorFalso()

    with pytest.raises(ValidationError):
        compor(primario, reserva).invocar([])

    assert reserva.chamadas == 0
    assert falha_operacional(validacao) is False


# ----------------------------------------------------------------------
# Quando a reserva É chamada
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "erro",
    [
        erro_http(503, "UNAVAILABLE: high demand"),
        erro_http(429, "Too Many Requests"),
        erro_http(401, "unauthorized"),
        TimeoutError("estourou o tempo"),
        ConnectionError("conexão recusada"),
    ],
)
def test_falha_operacional_aciona_a_reserva_uma_unica_vez(erro):
    primario = ProvedorFalso(erro=erro)
    reserva = ProvedorFalso({"reserva": True})

    assert compor(primario, reserva).invocar([]) == {"reserva": True}
    assert primario.chamadas == 1
    assert reserva.chamadas == 1


def test_a_queda_dos_dois_vira_uma_falha_operacional_explicita():
    primario = ProvedorFalso(erro=erro_http(503, "UNAVAILABLE"))
    reserva = ProvedorFalso(erro=erro_http(503, "UNAVAILABLE"))

    with pytest.raises(ErroProvedoresIndisponiveis) as capturado:
        compor(primario, reserva, "classifier").invocar([])

    mensagem = str(capturado.value)
    assert "classifier" in mensagem
    assert "ErroHttpFalso" in mensagem
    assert falha_operacional(capturado.value) is True


def test_a_composicao_nao_alterna_indefinidamente():
    primario = ProvedorFalso(erro=erro_http(503))
    reserva = ProvedorFalso(erro=erro_http(503))

    with pytest.raises(ErroProvedoresIndisponiveis):
        compor(primario, reserva).invocar([])

    assert primario.chamadas == 1 and reserva.chamadas == 1


def test_o_teto_por_fronteira_e_duas_chamadas_de_cada():
    """Cada agente faz no máximo 2 tentativas; a composição não multiplica."""
    primario = ProvedorFalso(erro=erro_http(503))
    reserva = ProvedorFalso(erro=erro_http(503))
    composicao = compor(primario, reserva)

    for _ in range(2):  # as duas tentativas do agente
        with pytest.raises(ErroProvedoresIndisponiveis):
            composicao.invocar([])

    assert primario.chamadas == 2
    assert reserva.chamadas == 2


# ----------------------------------------------------------------------
# Segredo, observabilidade e composição opcional
# ----------------------------------------------------------------------


def test_nenhum_segredo_aparece_na_excecao_ou_no_log(caplog):
    primario = ProvedorFalso(erro=erro_http(503, f"falha com {CHAVE_FALSA}"))
    reserva = ProvedorFalso(erro=erro_http(503, f"reserva com {CHAVE_FALSA}"))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ErroProvedoresIndisponiveis) as capturado:
            compor(primario, reserva).invocar([])

    assert CHAVE_FALSA not in str(capturado.value)
    assert CHAVE_FALSA not in caplog.text
    assert "high demand" not in str(capturado.value)


def test_o_log_traz_so_fronteira_e_decisao(caplog):
    primario = ProvedorFalso(erro=erro_http(503, "corpo enorme do provedor"))
    reserva = ProvedorFalso({"ok": True})

    with caplog.at_level(logging.WARNING):
        compor(primario, reserva, "briefing").invocar([{"prompt": "sigiloso"}])

    assert "briefing" in caplog.text
    assert "reserva" in caplog.text
    assert "sigiloso" not in caplog.text
    assert "corpo enorme" not in caplog.text


def test_sem_chave_de_reserva_o_comportamento_fica_identico_ao_de_hoje():
    primario = ProvedorFalso()

    def fabrica(_):  # pragma: no cover - não deve ser chamada
        raise AssertionError("a reserva não pode ser construída sem chave")

    for chave in ("", "   ", None):
        composto = compor_com_reserva(
            primario, fabrica, fronteira="extractor", chave_reserva=chave
        )
        assert composto is primario


def test_com_chave_a_reserva_e_construida_uma_vez_por_fronteira():
    primario = ProvedorFalso()
    construidas = []

    def fabrica(chave):
        construidas.append(chave)
        return ProvedorFalso()

    composto = compor_com_reserva(
        primario, fabrica, fronteira="extractor", chave_reserva=CHAVE_FALSA
    )

    assert isinstance(composto, ProvedorComFallback)
    assert construidas == [CHAVE_FALSA]


# ----------------------------------------------------------------------
# Os cinco protocolos aceitam a composição, sem cliente real
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "fronteira",
    ["query_planner", "extractor", "classifier", "recommendation", "briefing"],
)
def test_a_composicao_satisfaz_o_protocolo_de_cada_fronteira(fronteira):
    composicao = compor(ProvedorFalso({"x": 1}), ProvedorFalso(), fronteira)

    assert hasattr(composicao, "invocar")
    assert composicao.invocar([("system", "instrução")]) == {"x": 1}


def test_a_injecao_offline_nao_constroi_nenhum_cliente_de_rede(monkeypatch):
    """Nem Gemini nem Groq: os testes injetam fakes e nada mais.

    O ``ChatGroq`` é trocado por uma bomba: se a composição tentasse construir
    um cliente real em algum momento, o teste explodiria.
    """

    def bomba(**_):  # pragma: no cover - só dispara se houver regressão
        raise AssertionError("nenhum cliente de rede pode ser construído aqui")

    monkeypatch.setattr("langchain_groq.ChatGroq", bomba)

    composicao = compor(ProvedorFalso(), ProvedorFalso())

    assert composicao.invocar([]) == {"ok": True}


# ----------------------------------------------------------------------
# Falha da reserva: operacional e não operacional são coisas diferentes
# ----------------------------------------------------------------------

SEGREDO_NA_RESERVA = "E400: DOCUMENTO CONFIDENCIAL"


# Passados por constante de propósito: assim o traceback formatado mostra o
# NOME da variável nas linhas de código, e o teste mede vazamento real da
# exceção, não eco do próprio fonte do teste.
PROMPT_SIGILOSO = "prompt-sigiloso-do-usuario"
DOCUMENTO_SIGILOSO = "documento-confidencial-do-cliente"


def cair_com(erro_primario, erro_reserva):
    composicao = compor(
        ProvedorFalso(erro=erro_primario), ProvedorFalso(erro=erro_reserva), "extractor"
    )
    with pytest.raises(Exception) as capturado:
        composicao.invocar([("system", PROMPT_SIGILOSO), ("human", DOCUMENTO_SIGILOSO)])
    return capturado.value


def test_reserva_que_cai_por_indisponibilidade_vira_falha_operacional():
    from radar.provedores import ErroProvedoresIndisponiveis as Indisponivel

    final = cair_com(erro_http(503, "UNAVAILABLE"), erro_http(503, "UNAVAILABLE"))

    assert isinstance(final, Indisponivel)
    assert falha_operacional(final) is True


@pytest.mark.parametrize(
    "erro_reserva",
    [
        erro_http(400, SEGREDO_NA_RESERVA),
        TypeError(SEGREDO_NA_RESERVA),
        AttributeError(SEGREDO_NA_RESERVA),
        ValueError(SEGREDO_NA_RESERVA),
    ],
)
def test_reserva_que_recusa_por_contrato_nao_vira_indisponibilidade(erro_reserva):
    from radar.provedores import ErroReservaIncompativel

    final = cair_com(erro_http(503, "UNAVAILABLE"), erro_reserva)

    assert isinstance(final, ErroReservaIncompativel)
    assert falha_operacional(final) is False
    assert "sem provedor" not in str(final)


def test_um_400_da_reserva_nao_e_descrito_como_os_dois_fora_do_ar():
    final = cair_com(erro_http(503), erro_http(400, SEGREDO_NA_RESERVA))

    mensagem = str(final)
    assert "não é indisponibilidade" in mensagem
    assert "503" in mensagem and "400" in mensagem


def test_o_erro_final_guarda_fronteira_classe_e_codigo_mas_nao_o_corpo():
    final = cair_com(erro_http(503, "corpo do gemini"), erro_http(400, SEGREDO_NA_RESERVA))

    mensagem = str(final)
    assert "extractor" in mensagem
    assert "ErroHttpFalso" in mensagem
    assert SEGREDO_NA_RESERVA not in mensagem
    assert "corpo do gemini" not in mensagem


@pytest.mark.parametrize(
    "erro_reserva", [erro_http(503, SEGREDO_NA_RESERVA), erro_http(400, SEGREDO_NA_RESERVA)]
)
def test_a_cadeia_de_excecoes_nao_alcanca_o_corpo_do_provedor(erro_reserva):
    import traceback

    final = cair_com(erro_http(503, "corpo do gemini"), erro_reserva)

    assert final.__cause__ is None
    assert final.__context__ is None
    formatado = "".join(traceback.format_exception(final))
    assert SEGREDO_NA_RESERVA not in formatado
    assert "corpo do gemini" not in formatado
    assert PROMPT_SIGILOSO not in formatado
    assert DOCUMENTO_SIGILOSO not in formatado


@pytest.mark.parametrize(
    "erro_reserva", [erro_http(503, SEGREDO_NA_RESERVA), erro_http(400, SEGREDO_NA_RESERVA)]
)
def test_o_relatorio_do_lote_nao_expoe_o_corpo_da_reserva(erro_reserva):
    from radar.lote import _mensagem_falha

    final = cair_com(erro_http(503, "corpo do gemini"), erro_reserva)

    mensagem = _mensagem_falha(final)

    assert SEGREDO_NA_RESERVA not in mensagem
    assert "corpo do gemini" not in mensagem


def test_o_lote_classifica_queda_dos_dois_como_falha_operacional():
    from radar.lote import _tipo_falha

    final = cair_com(erro_http(503), erro_http(503))

    assert _tipo_falha(final) == "falha_operacional"


def test_o_lote_classifica_defeito_da_reserva_como_falha_de_processamento():
    from radar.lote import _tipo_falha

    final = cair_com(erro_http(503), erro_http(400, "schema recusado"))

    assert _tipo_falha(final) == "falha_de_processamento"


def test_a_reserva_nao_volta_para_o_primario():
    primario = ProvedorFalso(erro=erro_http(503))
    reserva = ProvedorFalso(erro=erro_http(400))

    with pytest.raises(Exception):
        compor(primario, reserva).invocar([])

    assert primario.chamadas == 1 and reserva.chamadas == 1


# ----------------------------------------------------------------------
# A neutralidade é do nó, não da composição
# ----------------------------------------------------------------------


def test_a_fronteira_de_fallback_continua_distinguindo_primario_e_reserva(caplog):
    """Quem observa os dois provedores pode e deve nomeá-los.

    A mensagem final do agente virou neutra porque o nó não tem como saber
    quem respondeu. A composição tem: ela chamou um, viu cair e chamou o
    outro. Neutralizar também esta camada apagaria a única observabilidade
    que diz qual lado falhou — sem devolver nada em troca.
    """
    primario = ProvedorFalso(erro=erro_http(503, "corpo cru do primário"))
    reserva = ProvedorFalso({"ok": True})

    with caplog.at_level(logging.WARNING):
        assert compor(primario, reserva, "classifier").invocar([]) == {"ok": True}

    assert "primário" in caplog.text
    assert "reserva" in caplog.text
    assert "classifier" in caplog.text
    # Classe e código bastam: o corpo do terceiro continua fora do log.
    assert "corpo cru" not in caplog.text
    # A linguagem neutra do nó não pode contaminar o diagnóstico da composição.
    assert "provedor de IA" not in caplog.text


# --------------------------------------------------------------------------
# A mesma distinção em cada fronteira de agente — dublês roteirizados
# --------------------------------------------------------------------------


def queda_do_primario():
    """503 com corpo cru de terceiro embutido; nada disso pode vazar."""
    return erro_http(503, CORPO_BRUTO_DO_PROVEDOR)


def recusa_da_reserva():
    """400: a reserva recusou o pedido estruturado. Não é queda."""
    return erro_http(400, "INVALID_ARGUMENT: schema não aceito pela reserva")


class ProvedorRoteirizado:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas.append(mensagens)
        if not self.respostas:
            raise AssertionError("o provedor recebeu uma chamada a mais do que o teto")
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


class Composicao:
    """Primário + reserva reais, com contagem de chamadas nos dois lados."""

    def __init__(self, primario_respostas, reserva_respostas, fronteira: str):
        self.primario = ProvedorRoteirizado(*primario_respostas)
        self.reserva = ProvedorRoteirizado(*reserva_respostas)
        self.provedor = ProvedorComFallback(
            self.primario, self.reserva, fronteira=fronteira
        )

    def invocar(self, mensagens):
        return self.provedor.invocar(mensagens)


def recupera_na_segunda(resposta_valida, fronteira: str) -> Composicao:
    """1ª tentativa: 503 no primário + 400 na reserva. 2ª: primário responde."""
    return Composicao(
        [queda_do_primario(), resposta_valida],
        [recusa_da_reserva()],
        fronteira,
    )


def recusa_sempre(fronteira: str) -> Composicao:
    return Composicao(
        [queda_do_primario(), queda_do_primario()],
        [recusa_da_reserva(), recusa_da_reserva()],
        fronteira,
    )


def indisponibilidade_dupla(fronteira: str) -> Composicao:
    return Composicao(
        [queda_do_primario(), queda_do_primario()],
        [queda_do_primario(), queda_do_primario()],
        fronteira,
    )


def exigir_falha_de_contrato(erro: Exception, composicao: Composicao) -> None:
    """Falha segura: dois toques em cada provedor, não operacional, neutra."""
    assert len(composicao.primario.chamadas) == 2, "o teto de 2 tentativas mudou"
    assert len(composicao.reserva.chamadas) == 2, "o teto de 2 tentativas mudou"
    assert falha_operacional(erro) is False, (
        "recusa estruturada da reserva não pode virar indisponibilidade"
    )
    assert _tipo_falha(erro) == "falha_de_processamento"
    exigir_falha_neutra_de_provedor(str(erro))
    for termo in TERMOS_QUE_A_FALHA_SEGURA_NAO_REPETE:
        assert termo not in str(erro)
    assert "400" not in str(erro) and "503" not in str(erro)


# --------------------------------------------------------------------------
# Fronteira 1 — Query Planner
# --------------------------------------------------------------------------


def test_query_planner_recupera_apos_recusa_da_reserva(base):
    composicao = recupera_na_segunda(plano_valido(), "query_planner")

    saida = QueryPlanner(base, composicao)({"consulta_usuario": "startups de saúde"})

    assert saida["plano_consulta"].filtros.setor == "Saúde"
    assert len(composicao.primario.chamadas) == 2
    assert len(composicao.reserva.chamadas) == 1


def test_query_planner_falha_com_seguranca_apos_recusa_persistente(base):
    composicao = recusa_sempre("query_planner")

    with pytest.raises(ErroQueryPlanner) as capturado:
        QueryPlanner(base, composicao)({"consulta_usuario": "startups de saúde"})

    exigir_falha_de_contrato(capturado.value, composicao)


def test_query_planner_com_dois_provedores_fora_continua_operacional(base):
    composicao = indisponibilidade_dupla("query_planner")

    with pytest.raises(ErroQueryPlanner) as capturado:
        QueryPlanner(base, composicao)({"consulta_usuario": "startups de saúde"})

    assert falha_operacional(capturado.value) is True
    assert _tipo_falha(capturado.value) == "falha_operacional"
    assert len(composicao.primario.chamadas) == 1, "queda dupla não ganha retry extra"


# --------------------------------------------------------------------------
# Fronteira 2 — Extractor
# --------------------------------------------------------------------------


def test_extractor_recupera_apos_recusa_da_reserva(controlada):
    composicao = recupera_na_segunda(perfil_valido(controlada), "extractor")

    saida = Extractor(controlada.base, composicao)(estado_extractor(controlada))

    assert saida["perfil_extraido"].id_startup == controlada.id_startup
    assert len(composicao.primario.chamadas) == 2
    assert len(composicao.reserva.chamadas) == 1


def test_extractor_nao_injeta_correcao_falsa_no_segundo_prompt(controlada):
    """A reserva recusou o pedido; o modelo não errou nada para ser corrigido."""
    composicao = recupera_na_segunda(perfil_valido(controlada), "extractor")

    Extractor(controlada.base, composicao)(estado_extractor(controlada))

    primeira, segunda = composicao.primario.chamadas
    assert primeira == segunda
    assert "Falha de validação" not in "\n".join(texto for _, texto in segunda)


def test_extractor_falha_com_seguranca_apos_recusa_persistente(controlada):
    composicao = recusa_sempre("extractor")

    with pytest.raises(ErroExtractor) as capturado:
        Extractor(controlada.base, composicao)(estado_extractor(controlada))

    exigir_falha_de_contrato(capturado.value, composicao)


def test_extractor_com_dois_provedores_fora_continua_operacional(controlada):
    composicao = indisponibilidade_dupla("extractor")

    with pytest.raises(ErroExtractor) as capturado:
        Extractor(controlada.base, composicao)(estado_extractor(controlada))

    assert _tipo_falha(capturado.value) == "falha_operacional"


# --------------------------------------------------------------------------
# Fronteira 3 — Classifier
# --------------------------------------------------------------------------


def test_classifier_recupera_apos_recusa_da_reserva():
    composicao = recupera_na_segunda(classificacao_valida(), "classifier")

    saida = Classifier(composicao)(estado_classifier(perfil_ai_native()))

    assert saida["classificacao"].classe == "AI-native"
    assert len(composicao.primario.chamadas) == 2
    assert len(composicao.reserva.chamadas) == 1


def test_classifier_falha_com_seguranca_apos_recusa_persistente():
    composicao = recusa_sempre("classifier")

    with pytest.raises(ErroClassificador) as capturado:
        Classifier(composicao)(estado_classifier(perfil_ai_native()))

    exigir_falha_de_contrato(capturado.value, composicao)


def test_classifier_com_dois_provedores_fora_continua_operacional():
    composicao = indisponibilidade_dupla("classifier")

    with pytest.raises(ErroClassificador) as capturado:
        Classifier(composicao)(estado_classifier(perfil_ai_native()))

    assert _tipo_falha(capturado.value) == "falha_operacional"


# --------------------------------------------------------------------------
# Fronteira 4 — Recommendation
# --------------------------------------------------------------------------


def test_recommendation_recupera_apos_recusa_da_reserva():
    composicao = recupera_na_segunda(
        lote_de_rascunhos(rascunho_recomendacao()), "recommendation"
    )

    saida = executar_recommendation(composicao)

    assert saida["recomendacoes"], "o segundo turno válido deve produzir recomendação"
    assert len(composicao.primario.chamadas) == 2
    assert len(composicao.reserva.chamadas) == 1


def test_recommendation_falha_com_seguranca_apos_recusa_persistente():
    composicao = recusa_sempre("recommendation")

    with pytest.raises(ErroRecommendation) as capturado:
        executar_recommendation(composicao)

    exigir_falha_de_contrato(capturado.value, composicao)


def test_recommendation_com_dois_provedores_fora_continua_operacional():
    composicao = indisponibilidade_dupla("recommendation")

    with pytest.raises(ErroRecommendation) as capturado:
        executar_recommendation(composicao)

    assert _tipo_falha(capturado.value) == "falha_operacional"


# --------------------------------------------------------------------------
# Fronteira 5 — Briefing
# --------------------------------------------------------------------------


def test_briefing_recupera_apos_recusa_da_reserva(caminho_banco):
    composicao = recupera_na_segunda(rascunho_briefing(), "briefing")
    no, _ = montar_no_briefing(caminho_banco)
    no.provedor = composicao

    saida = no(estado_briefing(caminho_banco))

    assert saida["briefing"] is not None
    assert len(composicao.primario.chamadas) == 2
    assert len(composicao.reserva.chamadas) == 1


def test_briefing_falha_com_seguranca_apos_recusa_persistente(caminho_banco):
    composicao = recusa_sempre("briefing")
    no, _ = montar_no_briefing(caminho_banco)
    no.provedor = composicao

    with pytest.raises(ErroBriefing) as capturado:
        no(estado_briefing(caminho_banco))

    exigir_falha_de_contrato(capturado.value, composicao)
    assert "None" not in str(capturado.value)


def test_briefing_com_dois_provedores_fora_continua_operacional(caminho_banco):
    composicao = indisponibilidade_dupla("briefing")
    no, _ = montar_no_briefing(caminho_banco)
    no.provedor = composicao

    with pytest.raises(ErroBriefing) as capturado:
        no(estado_briefing(caminho_banco))

    assert _tipo_falha(capturado.value) == "falha_operacional"


# --------------------------------------------------------------------------
# Classificador de falha do lote e preservação da linha persistida
# --------------------------------------------------------------------------


def test_tipo_de_falha_separa_recusa_estruturada_de_indisponibilidade():
    recusa = ErroExtractor("falha do nó")
    recusa.__cause__ = ErroReservaIncompativel("recusa estruturada")
    queda = ErroExtractor("falha do nó")
    queda.__cause__ = ErroProvedoresIndisponiveis("sem provedor")

    assert _tipo_falha(recusa) == "falha_de_processamento"
    assert _tipo_falha(queda) == "falha_operacional"


def test_lote_preserva_analise_anterior_quando_a_regeneracao_recusa(tmp_path, base):
    startup_id = base.listar_startups_para_lote()[0].id_startup
    anterior = _analise(startup_id)
    base.salvar_analise(anterior)

    composicao = recusa_sempre("extractor")
    analisador = _analisador(
        tmp_path, base, composicao, ProvedorFila(_classificacao("AI-native"))
    )
    try:
        resultado = analisador.executar_todas([startup_id])
    finally:
        analisador.fechar()

    assert resultado.concluidas == ()
    assert len(resultado.falhas) == 1
    assert resultado.falhas[0].tipo == "falha_de_processamento"
    for termo in TERMOS_QUE_A_FALHA_SEGURA_NAO_REPETE:
        assert termo not in resultado.falhas[0].mensagem

    preservada = base.carregar_analises([startup_id])[startup_id]
    assert preservada.status == anterior.status
    assert preservada.fit_score.total == anterior.fit_score.total


def test_lote_com_recusa_nao_regenera_e_nao_apaga_cache(tmp_path, base):
    """A recusa persistente não pode limpar o que já estava persistido."""
    startup_id = base.listar_startups_para_lote()[0].id_startup
    base.salvar_analise(_analise(startup_id))
    antes = base.carregar_analises([startup_id])[startup_id]

    composicao = recusa_sempre("extractor")
    analisador = _analisador(
        tmp_path, base, composicao, ProvedorFila(_classificacao("AI-native"))
    )
    try:
        analisador.executar_todas([startup_id])
    finally:
        analisador.fechar()

    assert base.carregar_analises([startup_id])[startup_id] == antes
