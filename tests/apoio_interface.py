"""Apoio da camada de interface: fábricas de dados e pilotagem do ``AppTest``.

Fica fora do ``conftest`` porque serve só à interface e porque monta objetos da
fronteira da aplicação, não contratos do núcleo. A segunda metade é a pilotagem
do Streamlit — montar a tela com a aplicação injetada, submeter uma consulta e
ler tudo que a tela escreveu. Esses ajudantes moravam dentro de um módulo de
teste e eram importados por outros seis; aqui eles têm dono.
"""

from __future__ import annotations

from datetime import date

from streamlit.testing.v1 import AppTest

from radar.aplicacao import ItemRanking, SaidaAprofundamento, SaidaDescoberta
from radar.configuracao import RAIZ_PROJETO
from radar.contratos import (
    DocumentoRecuperado,
    EmpresaCandidata,
    FiltrosEstruturados,
    PlanoConsulta,
    ResultadoRecuperacao,
)
from radar.interface import estado as sessao
from tests.conftest import briefing_normal_falso


CONSULTA_PADRAO = "fintechs brasileiras com modelos de linguagem em português"
CAMINHO_APP = str(RAIZ_PROJETO / "app.py")
ROTULO_BUSCAR = "Buscar candidatas"


def empresa_falsa(id_startup: int, nome: str, **ajustes) -> EmpresaCandidata:
    campos = {
        "id_startup": id_startup,
        "nome": nome,
        "setor": "Fintech",
        "estagio": "série B",
        "localizacao": "São Paulo, SP",
        "descricao_curta": f"Descrição curta de {nome}.",
    }
    campos.update(ajustes)
    return EmpresaCandidata(**campos)


def documento_falso(id_documento: int, id_startup: int, **ajustes):
    campos = {
        "id_documento": id_documento,
        "id_startup": id_startup,
        "tipo": "notícia",
        "titulo": f"Matéria {id_documento}",
        "url_fonte": f"https://fonte-{id_documento}.example/materia",
        "dominio_fonte": f"fonte-{id_documento}.example",
        "data_acesso": date(2026, 9, 3),
        "score_bm25": -3.5 - id_documento,
    }
    campos.update(ajustes)
    return DocumentoRecuperado(**campos)


def item_falso(
    posicao: int,
    id_startup: int,
    nome: str,
    *,
    status_analise: str = "concluida",
    classe: str | None = "AI-native",
    fit_score_total: int | None = 72,
    justificativa_fit_score: str | None = (
        "Dados proprietários e otimização confirmados."
    ),
    motivo_evidencia_insuficiente: str | None = None,
    documentos=None,
) -> ItemRanking:
    """Um ``ItemRanking`` já montado, do jeito que a aplicação o devolve."""
    docs = (
        documentos
        if documentos is not None
        else (documento_falso(id_startup * 10, id_startup),)
    )
    return ItemRanking(
        posicao=posicao,
        empresa=empresa_falsa(id_startup, nome),
        melhor_score_bm25=min((d.score_bm25 for d in docs), default=0.0),
        documentos=tuple(docs),
        status_analise=status_analise,
        classe=classe,
        fit_score_total=fit_score_total,
        justificativa_fit_score=justificativa_fit_score,
        motivo_evidencia_insuficiente=motivo_evidencia_insuficiente,
    )


def item_concluido_ia(posicao=1, id_startup=1, nome="Maritaca AI") -> ItemRanking:
    return item_falso(posicao, id_startup, nome)


def item_concluido_non_ai(posicao=2, id_startup=2, nome="Wine") -> ItemRanking:
    return item_falso(
        posicao,
        id_startup,
        nome,
        classe="non-AI",
        fit_score_total=0,
        justificativa_fit_score="Nenhum pilar pontua sem IA no produto.",
    )


def item_insuficiente(posicao=3, id_startup=3, nome="Colab") -> ItemRanking:
    return item_falso(
        posicao,
        id_startup,
        nome,
        status_analise="evidencia_insuficiente",
        classe=None,
        fit_score_total=None,
        justificativa_fit_score=None,
        motivo_evidencia_insuficiente="Nenhuma afirmação sobreviveu à conferência.",
    )


def item_ausente(posicao=4, id_startup=4, nome="Mombak") -> ItemRanking:
    return item_falso(
        posicao,
        id_startup,
        nome,
        status_analise="ausente",
        classe=None,
        fit_score_total=None,
        justificativa_fit_score=None,
    )


def plano_falso(**ajustes) -> PlanoConsulta:
    campos = {
        "filtros": FiltrosEstruturados(setor="Fintech"),
        "termos_busca": ["fintech", "modelo de linguagem"],
        "sinais_ia": ["LLM"],
        "foco_analise": "uso de modelos próprios em produção",
    }
    campos.update(ajustes)
    return PlanoConsulta(**campos)


def descoberta_falsa(
    *,
    consulta: str = CONSULTA_PADRAO,
    itens=None,
    criterios_relaxados: tuple[str, ...] = (),
    tentativas_relaxamento: int = 0,
) -> SaidaDescoberta:
    """``SaidaDescoberta`` coerente: o ranking espelha empresas e documentos."""
    itens = (
        tuple(itens)
        if itens is not None
        else (
            item_concluido_ia(),
            item_concluido_non_ai(),
            item_insuficiente(),
            item_ausente(),
        )
    )
    documentos = [documento for item in itens for documento in item.documentos]
    resultado = ResultadoRecuperacao(
        empresas=[item.empresa for item in itens],
        documentos=documentos,
        filtros_aplicados=FiltrosEstruturados(setor="Fintech"),
    )
    return SaidaDescoberta(
        consulta=consulta,
        rota="candidatas_prontas" if itens else "sem_resultado",
        plano=plano_falso(),
        resultado=resultado,
        ranking=itens,
        tentativas_relaxamento=tentativas_relaxamento,
        criterios_relaxados=criterios_relaxados,
        trajeto=("query_planner", "retriever"),
    )


def aprofundamento_falso(
    id_startup: int = 1, briefing=None, **ajustes
) -> SaidaAprofundamento:
    campos = {
        "briefing": briefing if briefing is not None else briefing_normal_falso(),
        "plano": plano_falso(),
        "id_startup": id_startup,
        "trajeto": ("retriever", "extractor", "classifier", "briefing"),
        "erros": (),
    }
    campos.update(ajustes)
    return SaidaAprofundamento(**campos)


class AplicacaoFalsa:
    """Fronteira de aplicação injetável: sem grafo, sem SQLite, sem rede.

    Aceita respostas por chamada (``descobertas``) e por startup
    (``aprofundamentos_por_startup`` / ``erros_por_startup``) para que a
    jornada com várias candidatas e várias buscas seja exercitável. Um item
    ``Exception`` é levantado em vez de devolvido, como no
    ``ProvedorSequencialFalso`` do núcleo.
    """

    def __init__(
        self,
        descoberta=None,
        aprofundamento=None,
        erro_aprofundamento=None,
        *,
        descobertas=None,
        aprofundamentos_por_startup=None,
        erros_por_startup=None,
    ):
        if descobertas is not None:
            self._descobertas = list(descobertas)
        else:
            self._descobertas = None
            self._descoberta = (
                descoberta if descoberta is not None else descoberta_falsa()
            )
        self._aprofundamento = aprofundamento
        self._erro_aprofundamento = erro_aprofundamento
        self._aprofundamentos_por_startup = dict(aprofundamentos_por_startup or {})
        self._erros_por_startup = dict(erros_por_startup or {})
        self.consultas: list[str] = []
        self.aprofundamentos: list[tuple[SaidaDescoberta, int]] = []

    def executar_descoberta(self, consulta: str) -> SaidaDescoberta:
        self.consultas.append(consulta)
        if self._descobertas is None:
            return self._descoberta
        if not self._descobertas:
            raise AssertionError(
                f"a descoberta foi chamada {len(self.consultas)} vezes, além das "
                "respostas programadas"
            )
        resposta = self._descobertas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    def executar_aprofundamento(
        self, descoberta: SaidaDescoberta, id_startup: int
    ) -> SaidaAprofundamento:
        self.aprofundamentos.append((descoberta, id_startup))
        if id_startup in self._erros_por_startup:
            raise self._erros_por_startup[id_startup]
        if self._erro_aprofundamento is not None:
            raise self._erro_aprofundamento
        if id_startup in self._aprofundamentos_por_startup:
            return self._aprofundamentos_por_startup[id_startup]
        if self._aprofundamento is not None:
            return self._aprofundamento
        return aprofundamento_falso(id_startup)


def chamadas_por_startup(aplicacao: AplicacaoFalsa) -> list[int]:
    """Ids na ordem em que a fronteira foi chamada, para contar repetição."""
    return [id_startup for _, id_startup in aplicacao.aprofundamentos]


# --------------------------------------------------------------------------
# Pilotagem do ``AppTest``: montar a tela, submeter e ler o que foi escrito
# --------------------------------------------------------------------------


def montar(aplicacao: AplicacaoFalsa) -> AppTest:
    teste = AppTest.from_file(CAMINHO_APP, default_timeout=30)
    teste.session_state[sessao.CHAVE_APLICACAO] = aplicacao
    return teste.run()


def botao_buscar(teste: AppTest):
    return next(botao for botao in teste.button if botao.label == ROTULO_BUSCAR)


def submeter(teste: AppTest, consulta: str) -> AppTest:
    teste.text_input(key="consulta").input(consulta)
    return botao_buscar(teste).click().run()


def textos(teste: AppTest) -> str:
    """Tudo que a tela escreveu, para asserções de presença e ausência."""
    partes = [elemento.value for elemento in teste.markdown]
    partes += [elemento.value for elemento in teste.caption]
    partes += [elemento.value for elemento in teste.subheader]
    partes += [elemento.value for elemento in teste.header]
    partes += [elemento.value for elemento in teste.title]
    partes += [elemento.value for elemento in teste.info]
    partes += [elemento.value for elemento in teste.warning]
    partes += [elemento.value for elemento in teste.error]
    partes += [elemento.value for elemento in teste.success]
    partes += [elemento.label for elemento in teste.expander]
    partes += [
        f"{elemento.label}: {elemento.value}" for elemento in teste.metric
    ]
    return "\n".join(str(parte) for parte in partes)


def abrir_ranking(aplicacao: AplicacaoFalsa) -> AppTest:
    """Monta a tela e submete a consulta padrão: o ranking fica visível."""
    teste = montar(aplicacao)
    return submeter(teste, CONSULTA_PADRAO)


def briefing_na_tela(briefing) -> AppTest:
    """Abre o ranking e aprofunda a candidata 1, deixando o briefing na tela."""
    aplicacao = AplicacaoFalsa(
        aprofundamento=aprofundamento_falso(1, briefing=briefing)
    )
    teste = abrir_ranking(aplicacao)
    return teste.button(key="aprofundar_1").click().run()
