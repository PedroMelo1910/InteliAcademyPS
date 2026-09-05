from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import ValidationError

from radar.base_startups import BaseStartups
from radar.contratos import (
    AfirmacaoValidada,
    CitacaoNvidia,
    Classificacao,
    ContextoNvidia,
    EmpresaCandidata,
    EntradaFitScore,
    EstadoRadar,
    EvidenciaStartup,
    FitScore,
    MetadadoDocumentoFitScore,
    PerfilValidado,
    Recomendacao,
    RecomendacaoRascunho,
    RelatorioRecomendacoes,
    ResultadoRecuperacao,
    TrechoNvidia,
)
from radar.provedores import ProvedorRecomendacaoRascunho, RascunhosRecomendacao
from radar.recomendacao import calcular_fit_score
from radar.regras_recomendacao import (
    CATEGORIAS_DE_DOR,
    TECNOLOGIAS_POR_GAP,
    calcular_complexidade,
    calcular_prioridade,
    conferir_gap_sustentado,
    gaps_sustentados,
    tecnologias_candidatas,
)


# Uma única correção estruturada, como nos demais nós de LLM do repositório.
TENTATIVAS_DE_RASCUNHO = 2

# O pilar de Centralidade de IA é o primeiro do contrato; a complexidade
# depende dele, e não de uma releitura própria da classe.
PILAR_CENTRALIDADE_IA = "centralidade_ia"


class ErroRecommendation(RuntimeError):
    """Falha segura: sem proveniência verificável, nenhuma recomendação é gravada."""


class Recommendation:
    """Cruza o perfil validado com o contexto NVIDIA recuperado.

    A divisão de trabalho é o ponto central deste nó: o LLM preenche apenas
    ``RecomendacaoRascunho`` — gap, tecnologias, as duas justificativas, a
    próxima ação e **ids**. Quem resolve os ids para objetos completos, quem
    calcula prioridade e complexidade e quem chama a função pura de fit-score é
    o nó. Prioridade, complexidade e fit-score sequer existem no schema que o
    modelo preenche.
    """

    def __init__(
        self,
        base: BaseStartups,
        provedor: ProvedorRecomendacaoRascunho,
        data_referencia: date | None = None,
    ):
        self.base = base
        self.provedor = provedor
        # A rubrica precisa de uma data explícita. Sem uma injetada, o nó usa a
        # data de acesso mais recente do próprio conjunto recuperado: é o
        # instante em que esta base de evidências foi coletada, é determinística
        # e mantém a pontuação reproduzível fora de um relógio global.
        self.data_referencia = data_referencia

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        classificacao = _validar_entrada(
            estado.get("classificacao"),
            Classificacao,
            "o Recommendation exige uma Classificacao no estado",
        )
        perfil = _validar_entrada(
            estado.get("perfil_validado"),
            PerfilValidado,
            "o Recommendation exige um PerfilValidado no estado",
        )
        contexto = _validar_entrada(
            estado.get("contexto_nvidia"),
            ContextoNvidia,
            "o Recommendation exige um ContextoNvidia no estado",
        )
        recuperacao = _validar_entrada(
            estado.get("resultado_recuperacao"),
            ResultadoRecuperacao,
            "o Recommendation exige um ResultadoRecuperacao no estado",
        )
        empresa = self._empresa(estado, recuperacao)

        confirmadas = {
            item.id_afirmacao: item
            for item in perfil.afirmacoes_validadas
            if item.situacao == "confirmada"
        }
        if not confirmadas:
            raise ErroRecommendation(
                "nenhuma afirmação confirmada sobreviveu à validação; não há "
                "evidência de startup para sustentar recomendação"
            )

        # Elegibilidade vem antes do provedor e antes do fit-score: sem gap
        # sustentado não existe recomendação legítima a pedir, e chamar o LLM
        # aqui só produziria rascunhos que seriam todos descartados.
        sustentados = gaps_sustentados(perfil)
        if not sustentados:
            # §11.3: não há o que recomendar, mas isso não é falha operacional.
            # O estado sai vazio e honesto, e o Briefing monta a variante de
            # evidência insuficiente. O provedor não é chamado: os rascunhos
            # seriam todos descartados.
            return self._sem_recomendacao(
                "nenhum gap está sustentado por evidência confirmada neste "
                "perfil: as dimensões estruturais estão desconhecidas ou com "
                "capacidade confirmada e não há dor documentada. O provedor de "
                "recomendação não foi chamado."
            )

        trechos = {trecho.id_chunk: trecho for trecho in contexto.trechos}

        metadados = self._metadados(perfil)
        fit_score = self._fit_score(
            classificacao, perfil, empresa, metadados, recuperacao
        )
        recomendacoes, descartes = self._recomendar(
            classificacao=classificacao,
            perfil=perfil,
            empresa=empresa,
            contexto=contexto,
            confirmadas=confirmadas,
            sustentados=sustentados,
            trechos=trechos,
            metadados=metadados,
            pontos_centralidade_ia=_pontos_centralidade(fit_score),
        )

        # ``Recomendacao`` carrega ``AnyHttpUrl``, que o msgpack do checkpointer
        # não serializa; a forma JSON do mesmo contrato atravessa o checkpoint
        # sem alterar o modelo. ``FitScore`` não tem URL e vai como instância,
        # como os demais artefatos do estado.
        if not recomendacoes:
            saida = self._sem_recomendacao(*descartes)
            return saida
        saida: dict[str, Any] = {
            "recomendacoes": [
                recomendacao.model_dump(mode="json")
                for recomendacao in recomendacoes
            ],
            "fit_score": fit_score,
            "trajeto": ["recommendation"],
        }
        # ``erros`` é canal acumulado: recebe só o que este nó produziu.
        if descartes:
            saida["erros"] = descartes
        return saida

    @staticmethod
    def _sem_recomendacao(*motivos: str) -> dict[str, Any]:
        """Saída honesta quando nenhuma recomendação com lastro pôde existir.

        ``fit_score`` fica nulo de propósito: publicar uma pontuação sem o
        pacote de recomendações que ela resume seria oferecer conclusão sem o
        conteúdo que a sustenta. O Briefing lê este estado e monta a variante
        de evidência insuficiente (§11.3).
        """
        saida: dict[str, Any] = {
            "recomendacoes": [],
            "fit_score": None,
            "trajeto": ["recommendation"],
        }
        if motivos:
            saida["erros"] = list(motivos)
        return saida

    # ------------------------------------------------------------------
    # Pré-condições e dados estruturados
    # ------------------------------------------------------------------

    @staticmethod
    def _empresa(
        estado: EstadoRadar, recuperacao: ResultadoRecuperacao
    ) -> EmpresaCandidata:
        selecionada = estado.get("startup_selecionada")
        if selecionada is None:
            raise ErroRecommendation(
                "o Recommendation só roda no aprofundamento de uma startup "
                "selecionada"
            )
        for empresa in recuperacao.empresas:
            if empresa.id_startup == int(selecionada):
                return empresa
        raise ErroRecommendation(
            f"a startup {selecionada} não pertence ao conjunto recuperado desta "
            "análise; nenhuma recomendação foi produzida"
        )

    def _metadados(
        self, perfil: PerfilValidado
    ) -> dict[int, MetadadoDocumentoFitScore]:
        ids = sorted(
            {item.id_documento for item in perfil.afirmacoes_validadas}
        )
        return self.base.carregar_metadados_fit_score(ids)

    def _data_referencia(
        self, empresa: EmpresaCandidata, recuperacao: ResultadoRecuperacao
    ) -> date:
        if self.data_referencia is not None:
            return self.data_referencia
        datas = [
            documento.data_acesso
            for documento in recuperacao.documentos
            if documento.id_startup == empresa.id_startup
        ]
        if not datas:
            raise ErroRecommendation(
                "o conjunto recuperado não traz documentos desta startup; sem "
                "data de acesso não há data de referência determinística para o "
                "fit-score"
            )
        return max(datas)

    def _fit_score(
        self,
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        metadados: dict[int, MetadadoDocumentoFitScore],
        recuperacao: ResultadoRecuperacao,
    ) -> FitScore:
        try:
            entrada = EntradaFitScore(
                classe=classificacao.classe,
                ids_afirmacoes_suporte_classe=sorted(
                    classificacao.ids_afirmacoes_suporte
                ),
                perfil_validado=perfil,
                setor=empresa.setor,
                estagio=empresa.estagio,
                documentos=[metadados[chave] for chave in sorted(metadados)],
                data_referencia=self._data_referencia(empresa, recuperacao),
            )
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroRecommendation(
                "não foi possível montar a entrada do fit-score a partir do "
                f"estado aprovado; nada foi gravado: {erro}"
            ) from erro
        return calcular_fit_score(entrada)

    # ------------------------------------------------------------------
    # Fronteira do LLM: um retry estruturado, nada além disso
    # ------------------------------------------------------------------

    def _recomendar(
        self,
        *,
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        contexto: ContextoNvidia,
        confirmadas: dict[int, AfirmacaoValidada],
        sustentados: dict[str, frozenset[int]],
        trechos: dict[int, TrechoNvidia],
        metadados: dict[int, MetadadoDocumentoFitScore],
        pontos_centralidade_ia: int,
    ) -> tuple[list[Recomendacao], list[str]]:
        erro_anterior: str | None = None
        for tentativa in range(TENTATIVAS_DE_RASCUNHO):
            ultima = tentativa == TENTATIVAS_DE_RASCUNHO - 1
            mensagens = self._montar_mensagens(
                classificacao,
                perfil,
                empresa,
                contexto,
                confirmadas,
                sustentados,
                erro_anterior,
            )
            try:
                bruto = self.provedor.invocar(mensagens)
            except (ValidationError, ValueError, TypeError) as exc:
                # O adaptador de structured output pode validar antes de
                # devolver; a resposta segue fora do contrato e consome o mesmo
                # retry corretivo dos demais nós de LLM.
                erro_anterior = _resumir_erro(exc)
                if ultima:
                    raise ErroRecommendation(self._MENSAGEM_FALHA_DUPLA) from exc
                continue
            except Exception as exc:
                raise ErroRecommendation(
                    "O Gemini não respondeu ao Recommendation; nenhuma "
                    "recomendação foi fabricada."
                ) from exc

            try:
                rascunhos = list(RascunhosRecomendacao.model_validate(bruto).rascunhos)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = _resumir_erro(exc)
                if ultima:
                    raise ErroRecommendation(self._MENSAGEM_FALHA_DUPLA) from exc
                continue

            validas, falhas = self._converter(
                rascunhos,
                empresa=empresa,
                confirmadas=confirmadas,
                sustentados=sustentados,
                trechos=trechos,
                metadados=metadados,
                pontos_centralidade_ia=pontos_centralidade_ia,
            )
            if not falhas:
                return self._relatorio(validas), []
            if not ultima:
                erro_anterior = " | ".join(falhas)
                continue
            # Depois do retry único, o que continua sem lastro é descartado —
            # menos recomendações com proveniência verificável é melhor que uma
            # sem lastro.
            if not validas:
                # §11.3: melhor nenhuma recomendação do que uma sem lastro. A
                # análise segue para o Briefing terminal com os motivos do
                # descarte preservados em ``erros``.
                return [], [
                    (
                        "nenhuma recomendação sobreviveu à conferência de "
                        f"proveniência: {' | '.join(falhas)}"
                    )
                ]
            return self._relatorio(validas), falhas
        raise AssertionError("laço de rascunhos terminou em estado impossível")

    _MENSAGEM_FALHA_DUPLA = (
        "O Gemini respondeu duas vezes fora do contrato estruturado; nenhuma "
        "recomendação foi gravada no estado."
    )

    @staticmethod
    def _relatorio(validas: list[Recomendacao]) -> list[Recomendacao]:
        """O teto de cinco do contrato vale também para a saída construída."""
        try:
            return list(RelatorioRecomendacoes(recomendacoes=validas).recomendacoes)
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroRecommendation(
                f"o conjunto construído viola o contrato do relatório: {erro}"
            ) from erro

    # ------------------------------------------------------------------
    # Construção determinística de uma recomendação
    # ------------------------------------------------------------------

    def _converter(
        self,
        rascunhos: list[RecomendacaoRascunho],
        *,
        empresa: EmpresaCandidata,
        confirmadas: dict[int, AfirmacaoValidada],
        sustentados: dict[str, frozenset[int]],
        trechos: dict[int, TrechoNvidia],
        metadados: dict[int, MetadadoDocumentoFitScore],
        pontos_centralidade_ia: int,
    ) -> tuple[list[Recomendacao], list[str]]:
        validas: list[Recomendacao] = []
        falhas: list[str] = []
        # §6.1 — um pacote coeso **por gap**. A duplicata é detectada aqui, e
        # não só no contrato, para que um lote repetido vire ``falhas`` e
        # participe da mesma correção única que os demais defeitos de rascunho.
        # Um gap só é considerado ocupado quando a recomendação dele foi de
        # fato construída: se a primeira tentativa daquele gap foi descartada,
        # a seguinte ainda é uma candidata legítima, não uma duplicata.
        gaps_ja_aceitos: set[str] = set()
        for rascunho in rascunhos:
            if rascunho.gap_enderecado in gaps_ja_aceitos:
                falhas.append(
                    f"recomendação descartada [gap {rascunho.gap_enderecado}]: "
                    "este gap já é endereçado por outra recomendação do lote; "
                    "cada gap entra uma única vez no relatório"
                )
                continue
            try:
                recomendacao = self._construir(
                    rascunho,
                    empresa=empresa,
                    confirmadas=confirmadas,
                    sustentados=sustentados,
                    trechos=trechos,
                    metadados=metadados,
                    pontos_centralidade_ia=pontos_centralidade_ia,
                )
            except (ValidationError, ValueError, TypeError) as erro:
                falhas.append(
                    f"recomendação descartada [gap {rascunho.gap_enderecado}]: "
                    f"{_resumir_erro(erro)}"
                )
            else:
                validas.append(recomendacao)
                gaps_ja_aceitos.add(rascunho.gap_enderecado)
        return validas, falhas

    @staticmethod
    def _construir(
        rascunho: RecomendacaoRascunho,
        *,
        empresa: EmpresaCandidata,
        confirmadas: dict[int, AfirmacaoValidada],
        sustentados: dict[str, frozenset[int]],
        trechos: dict[int, TrechoNvidia],
        metadados: dict[int, MetadadoDocumentoFitScore],
        pontos_centralidade_ia: int,
    ) -> Recomendacao:
        """Resolve ids, calcula as duas regras e deixa o contrato validar o todo."""
        candidatas = tecnologias_candidatas(rascunho.gap_enderecado)
        fora_do_conjunto = [
            tecnologia
            for tecnologia in rascunho.tecnologias
            if tecnologia not in candidatas
        ]
        if fora_do_conjunto:
            raise ValueError(
                f"as tecnologias {fora_do_conjunto} não são candidatas do gap "
                f"{rascunho.gap_enderecado}; permitidas: {list(candidatas)}"
            )

        desconhecidas = [
            id_afirmacao
            for id_afirmacao in rascunho.ids_afirmacoes
            if id_afirmacao not in confirmadas
        ]
        if desconhecidas:
            raise ValueError(
                f"ids_afirmacoes cita {desconhecidas}, que não são afirmações "
                f"confirmadas deste perfil validado; confirmadas: "
                f"{sorted(confirmadas)}"
            )

        ausentes = [
            id_chunk
            for id_chunk in rascunho.ids_chunks
            if id_chunk not in trechos
        ]
        if ausentes:
            raise ValueError(
                f"ids_chunks cita {ausentes}, que não pertencem ao ContextoNvidia "
                f"desta recuperação; disponíveis: {sorted(trechos)}"
            )

        # O elo que faltava: a evidência citada precisa sustentar este gap.
        conferir_gap_sustentado(
            rascunho.gap_enderecado, rascunho.ids_afirmacoes, sustentados
        )

        evidencias = []
        for id_afirmacao in rascunho.ids_afirmacoes:
            afirmacao = confirmadas[id_afirmacao]
            metadado = metadados.get(afirmacao.id_documento)
            if metadado is None:
                raise ValueError(
                    f"a afirmação {id_afirmacao} cita o documento "
                    f"{afirmacao.id_documento}, sem metadados de fonte na base"
                )
            evidencias.append(
                EvidenciaStartup(
                    id_afirmacao=id_afirmacao,
                    id_documento=afirmacao.id_documento,
                    url_fonte=metadado.url_fonte,
                    trecho_citado=afirmacao.trecho_citado,
                )
            )

        citacoes = [
            CitacaoNvidia(
                id_chunk=trechos[id_chunk].id_chunk,
                topico=trechos[id_chunk].topico,
                origem=trechos[id_chunk].origem,
                tecnologia=trechos[id_chunk].tecnologia,
                fonte_url=trechos[id_chunk].fonte_url,
                breadcrumb=trechos[id_chunk].breadcrumb,
            )
            for id_chunk in rascunho.ids_chunks
        ]

        prioridade = calcular_prioridade(
            categorias_citadas=[
                confirmadas[id_afirmacao].categoria
                for id_afirmacao in rascunho.ids_afirmacoes
            ],
            estagio=empresa.estagio,
            gap_confirmado=rascunho.gap_enderecado in sustentados,
        )
        complexidade = calcular_complexidade(
            rascunho.tecnologias, pontos_centralidade_ia
        )
        return Recomendacao(
            gap_enderecado=rascunho.gap_enderecado,
            tecnologias=list(rascunho.tecnologias),
            justificativa_tecnica=rascunho.justificativa_tecnica,
            justificativa_negocio=rascunho.justificativa_negocio,
            prioridade=prioridade,
            complexidade=complexidade,
            proxima_acao=rascunho.proxima_acao,
            evidencias_startup=evidencias,
            citacoes_nvidia=citacoes,
        )

    # ------------------------------------------------------------------
    # Prompt: só o mínimo aprovado
    # ------------------------------------------------------------------

    @staticmethod
    def _instrucao(sustentados: dict[str, frozenset[int]]) -> str:
        catalogo = "\n".join(
            f"  - {gap}: {', '.join(TECNOLOGIAS_POR_GAP[gap])}"
            for gap in sustentados
        )
        return (
            "Você é o Recommendation do NVIDIA Startup AI Radar. Produza de 1 a "
            "5 rascunhos de recomendação, cada um endereçando um gap distinto.\n"
            "Regras da resposta:\n"
            "- gap_enderecado: exatamente um dos gaps sustentados listados "
            "abaixo. Gap fora desta lista é descartado pelo nó, porque a "
            "evidência confirmada não o sustenta.\n"
            "- tecnologias: de 1 a 3, escolhidas SOMENTE entre as candidatas do "
            "gap escolhido:\n"
            f"{catalogo}\n"
            "- justificativa_tecnica: ancorada nos trechos NVIDIA citados.\n"
            "- justificativa_negocio: em termos de resultado operacional para "
            "quem compra, não em termos de engenharia.\n"
            "- proxima_acao: um tipo do catálogo fechado e uma única frase de "
            "detalhe contextualizada.\n"
            "- ids_afirmacoes: somente ids das afirmações confirmadas listadas "
            "abaixo; nunca invente um id nem use id de outra empresa. Ao menos "
            "um dos ids precisa ser um dos que sustentam o gap escolhido — "
            "citar evidência de outro gap não dá lastro a este.\n"
            "- ids_chunks: somente ids dos trechos NVIDIA listados abaixo; ao "
            "menos um precisa ser de um trecho com origem 'tecnologia'.\n"
            "- Você NÃO define prioridade, NÃO define complexidade e NÃO calcula "
            "fit-score: esses três campos não existem neste schema e são "
            "calculados por regra determinística fora do modelo.\n"
            "- Um rascunho sem id de afirmação e id de trecho válidos é "
            "descartado inteiro; prefira menos recomendações bem sustentadas."
        )

    @staticmethod
    def _dados(
        classificacao: Classificacao,
        empresa: EmpresaCandidata,
        contexto: ContextoNvidia,
        confirmadas: dict[int, AfirmacaoValidada],
    ) -> str:
        """Só o mínimo aprovado: nem trecho citado, nem rótulo de curadoria."""
        afirmacoes = "\n".join(
            f"[afirmação {id_afirmacao} | categoria: {item.categoria} "
            f"| polaridade: {item.polaridade}] {item.texto}"
            for id_afirmacao, item in sorted(confirmadas.items())
        )
        trechos = "\n\n".join(
            f"[chunk {trecho.id_chunk} | origem: {trecho.origem} "
            f"| tecnologia: {trecho.tecnologia or 'nenhuma'}]\n"
            f"{trecho.breadcrumb}\n{trecho.texto}"
            for trecho in contexto.trechos
        )
        return (
            f"Empresa analisada: setor {empresa.setor}; classe validada "
            f"{classificacao.classe}.\n\n"
            f"Afirmações confirmadas do perfil:\n{afirmacoes}\n\n"
            f"Trechos recuperados da base NVIDIA:\n\n{trechos}"
        )

    @staticmethod
    def _mapa_de_gaps(
        perfil: PerfilValidado, sustentados: dict[str, frozenset[int]]
    ) -> str:
        """As três situações, separadas: o que é gap, o que dói e o que não vale."""
        estruturais = [
            f"{item.dimensao} (sustentado pelas afirmações "
            f"{sorted(sustentados[item.dimensao])})"
            for item in perfil.estado_dimensoes_gap
            if item.dimensao in sustentados
        ]
        dores = [
            f"{gap} (sustentado pelas afirmações {sorted(ids)})"
            for gap, ids in sustentados.items()
            if gap in CATEGORIAS_DE_DOR
        ]
        bloqueadas = [
            f"{item.dimensao} ({item.estado})"
            for item in perfil.estado_dimensoes_gap
            if item.dimensao not in sustentados
        ]
        return (
            "Dimensões estruturais confirmadas como gap: "
            + ("; ".join(estruturais) or "nenhuma")
            + "\nCategorias de dor documentada por afirmação confirmada: "
            + ("; ".join(dores) or "nenhuma")
            + "\nDimensões desconhecidas ou com capacidade já confirmada — NÃO "
            "podem ser recomendadas como gap: "
            + ("; ".join(bloqueadas) or "nenhuma")
        )

    def _montar_mensagens(
        self,
        classificacao: Classificacao,
        perfil: PerfilValidado,
        empresa: EmpresaCandidata,
        contexto: ContextoNvidia,
        confirmadas: dict[int, AfirmacaoValidada],
        sustentados: dict[str, frozenset[int]],
        erro_anterior: str | None,
    ) -> list[tuple[str, str]]:
        mensagens = [
            ("system", self._instrucao(sustentados)),
            (
                "human",
                self._dados(classificacao, empresa, contexto, confirmadas)
                + "\n\n"
                + self._mapa_de_gaps(perfil, sustentados),
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


def _pontos_centralidade(fit_score: FitScore) -> int:
    for pilar in fit_score.pilares:
        if pilar.pilar == PILAR_CENTRALIDADE_IA:
            return pilar.pontos
    raise ErroRecommendation(
        "o fit-score calculado não traz o pilar de Centralidade de IA; a "
        "complexidade não pode ser derivada"
    )


def _resumir_erro(erro: Exception) -> str:
    if isinstance(erro, ValidationError):
        return "; ".join(
            f"{'.'.join(str(item) for item in falha['loc']) or 'rascunho'}: "
            f"{falha['msg']}"
            for falha in erro.errors()
        )
    return str(erro)


def _validar_entrada(bruto: object, contrato, mensagem: str):
    if bruto is None:
        raise ErroRecommendation(mensagem)
    try:
        return contrato.model_validate(bruto)
    except (ValidationError, ValueError, TypeError) as erro:
        raise ErroRecommendation(
            f"{mensagem}; recebido fora do contrato: {erro}"
        ) from erro
