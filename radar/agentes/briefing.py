"""O nó Briefing: a única saída do sistema (§11 da arquitetura).

A divisão de trabalho é a mesma dos demais nós de LLM, levada ao extremo: no
caminho normal o modelo escreve ~6 frases — tese, síntese e de dois a quatro
pontos — e escolhe **ids de afirmações confirmadas** para cada uma delas. Todo
o resto (cabeçalho, classe, fit-score, recomendações, fontes, avisos, rodapé)
é montagem determinística sobre objetos que já foram validados antes de
chegar aqui. As variantes non-AI e evidência insuficiente não chamam o LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import ValidationError

from radar.agentes.roteadores import rotear_r3
from radar.base_startups import BaseStartups
from radar.contratos import (
    VERSAO_RUBRICA,
    AfirmacaoValidada,
    Briefing,
    BriefingRascunho,
    CabecalhoBriefing,
    Classificacao,
    ConclusaoAncorada,
    ContextoNvidia,
    EmpresaCandidata,
    FitScore,
    FonteBriefing,
    MAXIMO_PALAVRAS_SINTESE,
    PerfilValidado,
    Recomendacao,
    ResultadoR3,
    ResultadoRecuperacao,
    RodapeBriefing,
    VarianteBriefing,
    VereditoBriefing,
)
from radar.provedores import ProvedorBriefingRascunho


# Uma única correção estruturada, como nos demais nós de LLM do repositório.
TENTATIVAS_DE_RASCUNHO = 2

NOME_DO_NO = "briefing"

# Avisos determinísticos (§11.2). São constantes para que duas execuções com o
# mesmo estado produzam exatamente a mesma lista, na mesma ordem.
AVISO_CONFIANCA_BAIXA = (
    "Confiança do perfil baixa: a evidência confirmada não atingiu o patamar de "
    "corroboração da rubrica. Leia as conclusões com reserva."
)
AVISO_NAO_ADERENTE = (
    "Veredito non-AI: a base pública não sustenta uso de IA no produto vendido, "
    "e por isso nenhuma tecnologia é recomendada."
)
AVISO_SEM_AFIRMACAO_CONFIRMADA = (
    "Evidência insuficiente: nenhuma afirmação do perfil sobreviveu à "
    "conferência de proveniência."
)
AVISO_SUPORTE_DA_CLASSE_DERRUBADO = (
    "Evidência insuficiente: o suporte citado pela classificação foi derrubado "
    "e não se recuperou dentro do teto de reextração."
)
AVISO_SEM_RECOMENDACAO_SUSTENTADA = (
    "Evidência insuficiente: nenhuma recomendação com proveniência verificável "
    "pôde ser construída para esta empresa."
)

# As três causas de insuficiência da §11.3. É um valor privado do nó — não
# entra em contrato público —, calculado **uma única vez** por execução. A
# síntese e o aviso principal derivam dele: manter duas condições paralelas foi
# o que permitiu a síntese afirmar falha de proveniência num caminho em que a
# evidência havia sido confirmada.
CAUSA_SEM_AFIRMACAO_CONFIRMADA = "sem_afirmacao_confirmada"
CAUSA_SUPORTE_DA_CLASSE_DERRUBADO = "suporte_da_classe_derrubado"
CAUSA_SEM_RECOMENDACAO_SUSTENTADA = "sem_recomendacao_sustentada"

AVISO_POR_CAUSA: dict[str, str] = {
    CAUSA_SEM_AFIRMACAO_CONFIRMADA: AVISO_SEM_AFIRMACAO_CONFIRMADA,
    CAUSA_SUPORTE_DA_CLASSE_DERRUBADO: AVISO_SUPORTE_DA_CLASSE_DERRUBADO,
    CAUSA_SEM_RECOMENDACAO_SUSTENTADA: AVISO_SEM_RECOMENDACAO_SUSTENTADA,
}

# Cada síntese descreve a causa real. Só a primeira é falha de proveniência.
SINTESE_POR_CAUSA: dict[str, str] = {
    CAUSA_SEM_AFIRMACAO_CONFIRMADA: (
        "Nenhuma afirmação extraída foi confirmada na conferência de "
        "proveniência, e o sistema não formula tese sem lastro."
    ),
    CAUSA_SUPORTE_DA_CLASSE_DERRUBADO: (
        "As afirmações que sustentavam a classificação não sobreviveram à "
        "validação depois do teto de reextração, e sem classe validada não há "
        "conclusão a publicar."
    ),
    CAUSA_SEM_RECOMENDACAO_SUSTENTADA: (
        "Há evidência confirmada sobre a empresa, mas ela não sustenta nenhuma "
        "recomendação NVIDIA rastreável até a fonte."
    ),
}


class ErroBriefing(RuntimeError):
    """Falha segura: sem lastro conferido, nenhum briefing é gravado."""


class AgenteBriefing:
    """Monta o ``Briefing`` final a partir do estado já validado.

    Recebe o relógio por injeção explícita: a data de geração é um dado do
    artefato, e escondê-la dentro de uma função pura tornaria o nó impossível
    de testar de forma determinística.
    """

    def __init__(
        self,
        base: BaseStartups,
        provedor: ProvedorBriefingRascunho,
        relogio: Callable[[], date] | None = None,
    ):
        self.base = base
        self.provedor = provedor
        self.relogio = relogio if relogio is not None else date.today

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------

    def __call__(self, estado: dict[str, Any]) -> dict[str, Any]:
        consulta = self._consulta_original(estado)
        classificacao = _validar(
            estado.get("classificacao"),
            Classificacao,
            "o Briefing exige uma Classificacao no estado",
        )
        perfil = _validar(
            estado.get("perfil_validado"),
            PerfilValidado,
            "o Briefing exige um PerfilValidado no estado",
        )
        recuperacao = _validar(
            estado.get("resultado_recuperacao"),
            ResultadoRecuperacao,
            "o Briefing exige um ResultadoRecuperacao no estado",
        )
        empresa = self._empresa(estado, recuperacao)

        # A revalidação acontece antes de qualquer chamada de provedor: uma
        # recomendação fora do contrato no estado é defeito a montante, e
        # gastar uma chamada de LLM sobre ela seria desperdício.
        recomendacoes = self._revalidar_recomendacoes(estado)
        rota = self._rota_r3(estado)
        variante = self._variante(rota, recomendacoes, estado.get("fit_score"))
        # A conferência precede o provedor: uma recomendação sem lastro no
        # estado atual não pode nem custar uma chamada de LLM, nem aparecer.
        if variante == "normal":
            self._conferir_recomendacoes(
                recomendacoes, perfil=perfil, empresa=empresa, estado=estado
            )

        confirmadas = {
            item.id_afirmacao: item
            for item in perfil.afirmacoes_validadas
            if item.situacao == "confirmada"
        }
        data = self.relogio()

        causa: str | None = None
        if variante == "normal":
            veredito, sintese, pontos = self._conclusoes_do_llm(
                classificacao=classificacao,
                perfil=perfil,
                empresa=empresa,
                confirmadas=confirmadas,
                fit_score=_fit_score(estado),
            )
        elif variante == "nao_aderente":
            veredito, sintese, pontos = self._conclusoes_non_ai(
                classificacao, empresa
            )
        else:
            causa = _causa_insuficiencia(perfil, rota)
            veredito, sintese, pontos = self._conclusoes_insuficientes(causa)

        fontes = self._fontes(
            empresa=empresa,
            perfil=perfil,
            veredito=veredito,
            sintese=sintese,
            pontos=pontos,
            recomendacoes=recomendacoes if variante == "normal" else [],
        )

        briefing = self._montar(
            variante=variante,
            rota=rota,
            empresa=empresa,
            estado=estado,
            perfil=perfil,
            veredito=veredito,
            sintese=sintese,
            pontos=pontos,
            recomendacoes=recomendacoes if variante == "normal" else [],
            fontes=fontes,
            consulta=consulta,
            causa=causa,
            data=data,
        )
        # ``Briefing`` embute ``AnyHttpUrl``, que o msgpack do checkpointer não
        # serializa; a forma JSON do mesmo contrato atravessa sem alterá-lo.
        return {
            "briefing": briefing.model_dump(mode="json"),
            "trajeto": [NOME_DO_NO],
        }

    # ------------------------------------------------------------------
    # Pré-condições e escolha da variante
    # ------------------------------------------------------------------

    @staticmethod
    def _empresa(
        estado: dict[str, Any], recuperacao: ResultadoRecuperacao
    ) -> EmpresaCandidata:
        selecionada = estado.get("startup_selecionada")
        if selecionada is None:
            raise ErroBriefing(
                "o Briefing só roda no aprofundamento de uma startup selecionada"
            )
        for empresa in recuperacao.empresas:
            if empresa.id_startup == int(selecionada):
                return empresa
        raise ErroBriefing(
            f"a startup {selecionada} não pertence ao conjunto recuperado desta "
            "análise; nenhum briefing foi produzido"
        )

    @staticmethod
    def _revalidar_recomendacoes(estado: dict[str, Any]) -> list[Recomendacao]:
        """Revalida a forma JSON do estado sem reescrever nenhum campo."""
        brutas = estado.get("recomendacoes") or []
        try:
            return [Recomendacao.model_validate(item) for item in brutas]
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroBriefing(
                "o estado traz uma recomendação fora do contrato; o briefing "
                f"não embute o que não revalida: {_resumir(erro)}"
            ) from erro

    @staticmethod
    def _consulta_original(estado: dict[str, Any]) -> str:
        """A consulta do usuário é dado obrigatório do cabeçalho (§11.2).

        Não existe texto substituto: um briefing que inventa a pergunta que o
        originou perde a rastreabilidade da própria análise, e o cabeçalho
        deixaria de ser determinístico a partir da entrada real.
        """
        bruta = estado.get("consulta_usuario")
        if not isinstance(bruta, str) or not bruta.strip():
            raise ErroBriefing(
                "o Briefing exige a consulta original do usuário em "
                "consulta_usuario; sem ela o cabeçalho não pode ser montado e "
                "nenhum briefing é produzido"
            )
        return bruta.strip()

    def _conferir_recomendacoes(
        self,
        recomendacoes: list[Recomendacao],
        *,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        estado: dict[str, Any],
    ) -> None:
        """Confere cada recomendação embutida contra o estado ATUAL.

        Revalidar a forma Pydantic prova apenas que o objeto é bem formado. Uma
        recomendação de outra execução — ou adulterada no checkpoint — pode ser
        bem formada e ainda assim citar afirmação derrubada, documento de outra
        startup, URL trocada, trecho reescrito ou chunk de um contexto NVIDIA
        que já não existe. O briefing é a saída pública do sistema: aqui a
        proveniência é reconferida elo a elo, não assumida.
        """
        if not recomendacoes:
            return
        contexto = _validar(
            estado.get("contexto_nvidia"),
            ContextoNvidia,
            "o briefing embute recomendação e por isso exige, no estado, o "
            "ContextoNvidia que sustenta as citações dela",
        )
        trechos = {trecho.id_chunk: trecho for trecho in contexto.trechos}
        validadas = {
            item.id_afirmacao: item for item in perfil.afirmacoes_validadas
        }
        # Mesma fronteira parametrizada e isolada por startup usada pelo índice
        # de fontes: documento de outra empresa simplesmente não volta.
        fontes = self.base.carregar_fontes_briefing(
            empresa.id_startup,
            sorted(
                {
                    evidencia.id_documento
                    for recomendacao in recomendacoes
                    for evidencia in recomendacao.evidencias_startup
                }
            ),
        )
        for recomendacao in recomendacoes:
            rotulo = f"a recomendação do gap {recomendacao.gap_enderecado}"
            for evidencia in recomendacao.evidencias_startup:
                _conferir_evidencia(evidencia, rotulo, validadas, fontes, empresa)
            for citacao in recomendacao.citacoes_nvidia:
                _conferir_citacao(citacao, rotulo, trechos)

    @staticmethod
    def _rota_r3(estado: dict[str, Any]) -> ResultadoR3:
        """A rota registrada no rodapé é a que o próprio R3 decidiu."""
        try:
            return rotear_r3(estado)
        except (KeyError, ValidationError, ValueError, TypeError) as erro:
            raise ErroBriefing(
                f"não foi possível reconstruir a rota do R3 para o rodapé: {erro}"
            ) from erro

    @staticmethod
    def _variante(
        rota: ResultadoR3,
        recomendacoes: list[Recomendacao],
        fit_score: object,
    ) -> VarianteBriefing:
        """§11.3: a forma do estado decide a variante, sem flag de controle."""
        if rota == "evidencia_insuficiente":
            return "evidencia_insuficiente"
        if rota == "nao_aderente":
            return "nao_aderente"
        # Caminho aderente: sem recomendação sustentada ou sem fit-score, o
        # Recommendation não completou e a saída honesta é a variante terminal.
        if not recomendacoes or fit_score is None:
            return "evidencia_insuficiente"
        return "normal"

    # ------------------------------------------------------------------
    # Caminho normal: fronteira do LLM com uma correção única
    # ------------------------------------------------------------------

    def _conclusoes_do_llm(
        self,
        *,
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        confirmadas: dict[int, AfirmacaoValidada],
        fit_score: FitScore,
    ) -> tuple[VereditoBriefing, ConclusaoAncorada, list[ConclusaoAncorada]]:
        erro_anterior: str | None = None
        for tentativa in range(TENTATIVAS_DE_RASCUNHO):
            ultima = tentativa == TENTATIVAS_DE_RASCUNHO - 1
            mensagens = self._montar_mensagens(
                classificacao, perfil, empresa, confirmadas, fit_score, erro_anterior
            )
            try:
                bruto = self.provedor.invocar(mensagens)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = _resumir(exc)
                if ultima:
                    raise ErroBriefing(_falha_dupla(erro_anterior)) from exc
                continue
            except Exception as exc:
                raise ErroBriefing(
                    "O Gemini não respondeu ao Briefing; nenhum briefing "
                    "parcial foi gravado no estado."
                ) from exc

            try:
                rascunho = BriefingRascunho.model_validate(bruto)
                self._conferir(rascunho, confirmadas)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = _resumir(exc)
                if ultima:
                    raise ErroBriefing(_falha_dupla(erro_anterior)) from exc
                continue

            veredito = VereditoBriefing(
                classe=classificacao.classe,
                fit_score_total=fit_score.total,
                tese=rascunho.tese.texto,
                ids_afirmacoes_suporte=sorted(rascunho.tese.ids_afirmacoes_suporte),
            )
            sintese = ConclusaoAncorada(
                texto=rascunho.sintese_executiva.texto,
                ids_afirmacoes_suporte=sorted(
                    rascunho.sintese_executiva.ids_afirmacoes_suporte
                ),
            )
            pontos = [
                ConclusaoAncorada(
                    texto=ponto.texto,
                    ids_afirmacoes_suporte=sorted(ponto.ids_afirmacoes_suporte),
                )
                for ponto in rascunho.pontos_de_conversa
            ]
            return veredito, sintese, pontos
        raise AssertionError("laço de rascunhos terminou em estado impossível")

    @staticmethod
    def _conferir(
        rascunho: BriefingRascunho, confirmadas: dict[int, AfirmacaoValidada]
    ) -> None:
        """Todo id precisa referenciar afirmação confirmada deste perfil."""
        conclusoes = [
            ("tese", rascunho.tese),
            ("sintese_executiva", rascunho.sintese_executiva),
            *(
                (f"pontos_de_conversa[{indice}]", ponto)
                for indice, ponto in enumerate(rascunho.pontos_de_conversa)
            ),
        ]
        for rotulo, conclusao in conclusoes:
            invalidos = [
                id_afirmacao
                for id_afirmacao in conclusao.ids_afirmacoes_suporte
                if id_afirmacao not in confirmadas
            ]
            if invalidos:
                raise ValueError(
                    f"{rotulo} cita {invalidos}, que não são afirmações "
                    "confirmadas deste perfil validado; confirmadas: "
                    f"{sorted(confirmadas)}"
                )

        palavras = len(rascunho.sintese_executiva.texto.split())
        if palavras > MAXIMO_PALAVRAS_SINTESE:
            raise ValueError(
                f"sintese_executiva tem {palavras} palavras e o contrato admite "
                f"no máximo {MAXIMO_PALAVRAS_SINTESE} palavras"
            )
        if (
            rascunho.sintese_executiva.texto.strip().casefold()
            == rascunho.tese.texto.strip().casefold()
        ):
            raise ValueError(
                "sintese_executiva precisa acrescentar contexto à tese, não "
                "repeti-la"
            )

    # ------------------------------------------------------------------
    # Variantes determinísticas: nenhum LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _conclusoes_non_ai(
        classificacao: Classificacao, empresa: EmpresaCandidata
    ) -> tuple[VereditoBriefing, ConclusaoAncorada, list[ConclusaoAncorada]]:
        ids = sorted(classificacao.ids_afirmacoes_suporte)
        veredito = VereditoBriefing(
            classe="non-AI",
            # §11.3: o zero pertence ao contrato desta saída, não a um
            # ``FitScore`` que o nó Recommendation nunca calculou.
            fit_score_total=0,
            tese=(
                f"A evidência confirmada sobre {empresa.nome} descreve um produto "
                f"de {empresa.setor} sem uso de IA no que a empresa vende."
            ),
            ids_afirmacoes_suporte=ids,
        )
        sintese = ConclusaoAncorada(
            texto=(
                f"O material público de {empresa.nome} sustenta um produto "
                "operacional, e não uma capacidade de IA própria. A empresa fica "
                "fora do escopo desta análise técnica."
            ),
            ids_afirmacoes_suporte=ids,
        )
        pontos = [
            ConclusaoAncorada(
                texto=(
                    "Confirmar com a empresa se existe iniciativa de IA fora do "
                    "material público consultado."
                ),
                ids_afirmacoes_suporte=ids,
            ),
            ConclusaoAncorada(
                texto=(
                    "Entender qual processo operacional concentra hoje o maior "
                    "volume manual da empresa."
                ),
                ids_afirmacoes_suporte=ids,
            ),
        ]
        return veredito, sintese, pontos

    @staticmethod
    def _conclusoes_insuficientes(causa: str) -> tuple[
        VereditoBriefing, ConclusaoAncorada, list[ConclusaoAncorada]
    ]:
        """A tese é genérica; a síntese nomeia a causa real da insuficiência."""
        veredito = VereditoBriefing(
            classe=None,
            fit_score_total=None,
            tese=(
                "A base disponível não sustenta uma conclusão sobre esta empresa."
            ),
            ids_afirmacoes_suporte=[],
        )
        sintese = ConclusaoAncorada(
            texto=SINTESE_POR_CAUSA[causa],
            ids_afirmacoes_suporte=[],
        )
        return veredito, sintese, []

    # ------------------------------------------------------------------
    # Índice de fontes: união determinística, deduplicada e isolada
    # ------------------------------------------------------------------

    def _fontes(
        self,
        *,
        empresa: EmpresaCandidata,
        perfil: PerfilValidado,
        veredito: VereditoBriefing,
        sintese: ConclusaoAncorada,
        pontos: list[ConclusaoAncorada],
        recomendacoes: list[Recomendacao],
    ) -> list[FonteBriefing]:
        citados: set[int] = set(veredito.ids_afirmacoes_suporte)
        citados.update(sintese.ids_afirmacoes_suporte)
        for ponto in pontos:
            citados.update(ponto.ids_afirmacoes_suporte)
        for recomendacao in recomendacoes:
            citados.update(
                evidencia.id_afirmacao
                for evidencia in recomendacao.evidencias_startup
            )
        if not citados:
            return []

        documentos_por_afirmacao = {
            item.id_afirmacao: item.id_documento
            for item in perfil.afirmacoes_validadas
        }
        orfas = sorted(
            id_afirmacao
            for id_afirmacao in citados
            if id_afirmacao not in documentos_por_afirmacao
        )
        if orfas:
            raise ErroBriefing(
                f"as afirmações {orfas} são citadas por alguma conclusão mas não "
                "existem no perfil validado desta análise"
            )

        ids_documentos = sorted(
            {documentos_por_afirmacao[id_afirmacao] for id_afirmacao in citados}
        )
        resolvidas = self.base.carregar_fontes_briefing(
            empresa.id_startup, ids_documentos
        )
        ausentes = [item for item in ids_documentos if item not in resolvidas]
        if ausentes:
            raise ErroBriefing(
                f"os documentos {ausentes} não pertencem à startup "
                f"{empresa.id_startup}; o índice de fontes não expõe documento "
                "de outra startup"
            )
        # Ordenação por host e url: determinística e independente de ids
        # internos, que não fazem parte da projeção pública da §11.2.
        return sorted(
            resolvidas.values(),
            key=lambda fonte: (fonte.host_normalizado, str(fonte.url_fonte)),
        )

    # ------------------------------------------------------------------
    # Montagem final: avisos e rodapé
    # ------------------------------------------------------------------

    def _montar(
        self,
        *,
        variante: VarianteBriefing,
        rota: ResultadoR3,
        empresa: EmpresaCandidata,
        estado: dict[str, Any],
        perfil: PerfilValidado,
        veredito: VereditoBriefing,
        sintese: ConclusaoAncorada,
        pontos: list[ConclusaoAncorada],
        recomendacoes: list[Recomendacao],
        fontes: list[FonteBriefing],
        consulta: str,
        causa: str | None,
        data: date,
    ) -> Briefing:
        confirmadas = sum(
            1 for item in perfil.afirmacoes_validadas if item.situacao == "confirmada"
        )
        derrubadas = len(perfil.afirmacoes_validadas) - confirmadas
        try:
            cabecalho = CabecalhoBriefing(
                nome=empresa.nome,
                site=self.base.carregar_site_oficial(empresa.id_startup),
                setor=empresa.setor,
                estagio=empresa.estagio,
                localizacao=empresa.localizacao,
                data_geracao=data,
                consulta_original=consulta,
            )
            rodape = RodapeBriefing(
                versao_rubrica=VERSAO_RUBRICA,
                data_execucao=data,
                afirmacoes_confirmadas=confirmadas,
                afirmacoes_derrubadas=derrubadas,
                trajeto=_trajeto(estado),
                rota_r3=rota,
            )
            return Briefing(
                variante=variante,
                cabecalho=cabecalho,
                veredito=veredito,
                sintese_executiva=sintese,
                pontos_de_conversa=pontos,
                recomendacoes=recomendacoes,
                fontes=fontes,
                avisos=_avisos(variante, causa, estado, perfil, derrubadas),
                rodape=rodape,
            )
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroBriefing(
                "o briefing montado viola o contrato da §11 e não foi gravado: "
                f"{_resumir(erro)}"
            ) from erro

    # ------------------------------------------------------------------
    # Prompt: só o mínimo aprovado
    # ------------------------------------------------------------------

    @staticmethod
    def _instrucao() -> str:
        return (
            "Você é o Briefing do NVIDIA Startup AI Radar. Escreva, em "
            "português, o texto que um gerente de Startups & VCs lê antes de "
            "ligar para a empresa.\n"
            "Regras da resposta:\n"
            "- tese: uma ou duas frases dizendo por que esta empresa merece (ou "
            "não) a conversa.\n"
            "- sintese_executiva: contexto em 20 segundos, no máximo "
            f"{MAXIMO_PALAVRAS_SINTESE} palavras, acrescentando informação à "
            "tese em vez de repeti-la.\n"
            "- pontos_de_conversa: de 2 a 4 perguntas ou ganchos objetivos para "
            "a própria ligação.\n"
            "- ids_afirmacoes_suporte: cada conclusão carrega os seus próprios "
            "ids, escolhidos SOMENTE entre as afirmações confirmadas listadas "
            "abaixo. Nunca invente um id, nunca use id de outra empresa e nunca "
            "deixe a lista vazia: uma conclusão sem lastro próprio é descartada "
            "inteira.\n"
            "- Não afirme fato que as afirmações confirmadas não sustentem.\n"
            "- Você NÃO escreve classe, fit-score, cabeçalho, recomendações, "
            "fontes, avisos, rodapé nem datas: esses campos não existem neste "
            "schema e são montados por regra determinística fora do modelo."
        )

    @staticmethod
    def _dados(
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        confirmadas: dict[int, AfirmacaoValidada],
        fit_score: FitScore,
    ) -> str:
        """Só o mínimo aprovado: nem documento inteiro, nem rótulo de curadoria."""
        afirmacoes = "\n".join(
            f"[afirmação {id_afirmacao} | categoria: {item.categoria} "
            f"| polaridade: {item.polaridade}] {item.texto}"
            for id_afirmacao, item in sorted(confirmadas.items())
        )
        dimensoes = "\n".join(
            f"- {item.dimensao}: {item.estado}"
            for item in perfil.estado_dimensoes_gap
        )
        return (
            f"Empresa analisada: {empresa.nome}; setor {empresa.setor}; "
            f"estágio {empresa.estagio}.\n"
            f"Classe validada: {classificacao.classe}. "
            f"Fit-score já calculado: {fit_score.total} de 100 "
            f"({fit_score.justificativa_curta})\n\n"
            f"Afirmações confirmadas do perfil:\n{afirmacoes}\n\n"
            f"Situação das dimensões estruturais:\n{dimensoes}"
        )

    def _montar_mensagens(
        self,
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        confirmadas: dict[int, AfirmacaoValidada],
        fit_score: FitScore,
        erro_anterior: str | None,
    ) -> list[tuple[str, str]]:
        mensagens = [
            ("system", self._instrucao()),
            (
                "human",
                self._dados(classificacao, perfil, empresa, confirmadas, fit_score),
            ),
        ]
        if erro_anterior:
            mensagens.append(
                (
                    "system",
                    "A resposta anterior violou o contrato. Corrija sem texto "
                    f"livre. Falha de validação: {erro_anterior}",
                )
            )
        return mensagens


# ----------------------------------------------------------------------
# Funções auxiliares determinísticas
# ----------------------------------------------------------------------


def _causa_insuficiencia(perfil: PerfilValidado, rota: ResultadoR3) -> str:
    """A causa real da insuficiência (§11.3), decidida em um só lugar.

    A ordem importa: sem afirmação confirmada, nada mais precisa ser explicado;
    havendo evidência confirmada, R3 distingue o suporte da classe derrubado do
    caminho aderente que não produziu recomendação sustentada. Este último **não
    é** falha de proveniência — a evidência passou, o que faltou foi uma
    recomendação rastreável até a fonte.
    """
    confirmadas = sum(
        1 for item in perfil.afirmacoes_validadas if item.situacao == "confirmada"
    )
    if not confirmadas:
        return CAUSA_SEM_AFIRMACAO_CONFIRMADA
    if rota == "evidencia_insuficiente":
        return CAUSA_SUPORTE_DA_CLASSE_DERRUBADO
    return CAUSA_SEM_RECOMENDACAO_SUSTENTADA


def _avisos(
    variante: VarianteBriefing,
    causa: str | None,
    estado: dict[str, Any],
    perfil: PerfilValidado,
    derrubadas: int,
) -> list[str]:
    """Ordem fixa: o mesmo estado sempre produz a mesma lista de avisos."""
    avisos: list[str] = []
    if estado.get("confianca_perfil") == "baixa":
        avisos.append(AVISO_CONFIANCA_BAIXA)
    relaxados = list(estado.get("criterios_relaxados") or [])
    if relaxados:
        avisos.append(
            "Critérios de busca relaxados para encontrar candidatas: "
            + ", ".join(relaxados)
            + "."
        )
    if variante == "nao_aderente":
        avisos.append(AVISO_NAO_ADERENTE)
    # Mesmo valor que decidiu a síntese: aviso e síntese não podem divergir.
    if causa is not None:
        avisos.append(AVISO_POR_CAUSA[causa])
    # §11.3: o ``PerfilValidado`` não viaja no payload público, então o motivo
    # de cada derrubada precisa chegar aqui colado ao id que ele explica. Ordem
    # por id: duas execuções iguais produzem a mesma lista.
    for item in sorted(
        (
            afirmacao
            for afirmacao in perfil.afirmacoes_validadas
            if afirmacao.situacao == "derrubada"
        ),
        key=lambda afirmacao: afirmacao.id_afirmacao,
    ):
        avisos.append(
            f"Afirmação {item.id_afirmacao} derrubada na conferência de "
            f"proveniência: {(item.motivo or '').strip()}"
        )
    return avisos


def _conferir_evidencia(evidencia, rotulo, validadas, fontes, empresa) -> None:
    """Cada elo da evidência de startup, conferido contra o perfil e a base."""
    afirmacao = validadas.get(evidencia.id_afirmacao)
    if afirmacao is None:
        raise ErroBriefing(
            f"{rotulo} cita a afirmação {evidencia.id_afirmacao}, que não "
            "existe no perfil validado desta análise"
        )
    if afirmacao.situacao != "confirmada":
        raise ErroBriefing(
            f"{rotulo} se apoia na afirmação {evidencia.id_afirmacao}, "
            "derrubada na conferência de proveniência; só afirmação confirmada "
            "sustenta recomendação"
        )
    if evidencia.id_documento != afirmacao.id_documento:
        raise ErroBriefing(
            f"{rotulo} associa a afirmação {evidencia.id_afirmacao} ao "
            f"documento {evidencia.id_documento}, mas o perfil validado a "
            f"ancora no documento {afirmacao.id_documento}"
        )
    if evidencia.trecho_citado != afirmacao.trecho_citado:
        raise ErroBriefing(
            f"{rotulo} exibe um trecho_citado diferente do que foi validado "
            f"para a afirmação {evidencia.id_afirmacao}; a evidência publicada "
            "precisa ser literalmente a que passou pela conferência"
        )
    fonte = fontes.get(evidencia.id_documento)
    if fonte is None:
        raise ErroBriefing(
            f"{rotulo} cita o documento {evidencia.id_documento}, que não "
            f"pertence à startup {empresa.id_startup}; o briefing não exibe "
            "documento de outra startup"
        )
    if str(evidencia.url_fonte) != str(fonte.url_fonte):
        raise ErroBriefing(
            f"{rotulo} declara url_fonte {evidencia.url_fonte} para o documento "
            f"{evidencia.id_documento}, mas a base registra {fonte.url_fonte}"
        )


def _conferir_citacao(citacao, rotulo, trechos) -> None:
    """A citação NVIDIA precisa ser o chunk desta recuperação, campo a campo."""
    trecho = trechos.get(citacao.id_chunk)
    if trecho is None:
        raise ErroBriefing(
            f"{rotulo} cita o chunk NVIDIA {citacao.id_chunk}, que não pertence "
            "ao ContextoNvidia desta execução"
        )
    divergentes = [
        campo
        for campo, embutido, atual in (
            ("topico", citacao.topico, trecho.topico),
            ("origem", citacao.origem, trecho.origem),
            ("tecnologia", citacao.tecnologia, trecho.tecnologia),
            ("fonte_url", str(citacao.fonte_url), str(trecho.fonte_url)),
            ("breadcrumb", citacao.breadcrumb, trecho.breadcrumb),
        )
        if embutido != atual
    ]
    if divergentes:
        raise ErroBriefing(
            f"{rotulo} cita o chunk {citacao.id_chunk} com {divergentes} "
            "diferente(s) do trecho recuperado; a citação não corresponde ao "
            "contexto NVIDIA atual"
        )


def _trajeto(estado: dict[str, Any]) -> list[str]:
    """O trajeto do rodapé termina neste briefing, e o registra uma só vez.

    Num thread reaproveitado o canal acumulado pode trazer o briefing de uma
    execução anterior, já superada por esta. O rodapé descreve o caminho **deste**
    artefato, então o registro antigo sai e o atual entra no fim.
    """
    percorrido = [
        nome for nome in estado.get("trajeto") or [] if nome != NOME_DO_NO
    ]
    return percorrido + [NOME_DO_NO]


def _fit_score(estado: dict[str, Any]) -> FitScore:
    bruto = estado.get("fit_score")
    try:
        return FitScore.model_validate(bruto)
    except (ValidationError, ValueError, TypeError) as erro:
        raise ErroBriefing(
            f"o caminho normal exige um FitScore válido no estado: {erro}"
        ) from erro


def _falha_dupla(erro: str) -> str:
    return (
        "O Gemini respondeu duas vezes fora do contrato estruturado; nenhum "
        f"briefing foi gravado no estado. Última falha: {erro}"
    )


def _resumir(erro: Exception) -> str:
    if isinstance(erro, ValidationError):
        return "; ".join(
            f"{'.'.join(str(item) for item in falha['loc']) or 'rascunho'}: "
            f"{falha['msg']}"
            for falha in erro.errors()
        )
    return str(erro)


def _validar(bruto: object, contrato, mensagem: str):
    if bruto is None:
        raise ErroBriefing(mensagem)
    try:
        return contrato.model_validate(bruto)
    except (ValidationError, ValueError, TypeError) as erro:
        raise ErroBriefing(
            f"{mensagem}; recebido fora do contrato: {erro}"
        ) from erro
