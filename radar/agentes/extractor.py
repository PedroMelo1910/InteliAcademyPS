from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from radar.agentes.roteadores import precisa_reextrair
from radar.base_startups import BaseStartups, ErroDocumentosStartup
from radar.contratos import (
    CATEGORIAS_AFIRMACAO,
    Classificacao,
    CATEGORIAS_ESTRUTURAIS,
    LIMITE_TRECHO_CITADO,
    MINIMO_CARACTERES_TRECHO_CITADO,
    MINIMO_PALAVRAS_TRECHO_CITADO,
    DocumentoIntegral,
    EmpresaCandidata,
    EstadoRadar,
    PerfilExtraido,
    PerfilValidado,
    PlanoConsulta,
    ResultadoRecuperacao,
)
from radar.provedores import ProvedorPerfilExtraido


CAMPOS_DERIVADOS_DA_EXTRACAO: tuple[str, ...] = (
    "classificacao",
    "perfil_validado",
    "confianca_perfil",
    "contexto_nvidia",
    "recomendacoes",
    "fit_score",
    "briefing",
)


class ErroExtractor(RuntimeError):
    """Falha segura: não inventa perfil nem descarta afirmação em silêncio."""


class Extractor:
    """Transforma o texto não estruturado dos documentos recuperados em perfil.

    O nó lê o texto completo do SQLite pelos ids do ``ResultadoRecuperacao`` e
    devolve ao estado apenas o perfil validado — nunca os documentos.
    """

    def __init__(self, base: BaseStartups, provedor: ProvedorPerfilExtraido):
        self.base = base
        self.provedor = provedor

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        resultado = self._recuperacao(estado)
        plano = self._plano(estado)
        id_startup = self._startup_alvo(estado, resultado)
        documentos_invasores = [
            documento.id_documento
            for documento in resultado.documentos
            if documento.id_startup != id_startup
        ]
        if documentos_invasores:
            raise ErroExtractor(
                "a recuperação mistura documentos de outra startup: "
                f"{documentos_invasores}"
            )
        ids_permitidos = [documento.id_documento for documento in resultado.documentos]
        if not ids_permitidos:
            raise ErroExtractor(
                f"a recuperação não trouxe nenhum documento da startup {id_startup}"
            )
        try:
            documentos = self.base.carregar_documentos(id_startup, ids_permitidos)
        except ErroDocumentosStartup as erro:
            raise ErroExtractor(
                f"os documentos recuperados não puderam ser lidos: {erro}"
            ) from erro

        empresa = next(
            (item for item in resultado.empresas if item.id_startup == id_startup), None
        )
        modo_estrito = self._modo_estrito(estado)
        perfil = self._extrair_com_validacao(
            id_startup, empresa, plano, documentos, modo_estrito
        )
        saida: dict[str, Any] = {
            "perfil_extraido": perfil,
            "tentativas_extracao": int(estado.get("tentativas_extracao", 0)) + 1,
            "trajeto": ["extractor"],
        }
        for campo in CAMPOS_DERIVADOS_DA_EXTRACAO:
            saida[campo] = None
        return saida

    # ------------------------------------------------------------------
    # Pré-condições do nó
    # ------------------------------------------------------------------

    @staticmethod
    def _recuperacao(estado: EstadoRadar) -> ResultadoRecuperacao:
        bruto = estado.get("resultado_recuperacao")
        if bruto is None:
            raise ErroExtractor(
                "o Extractor exige um ResultadoRecuperacao no estado"
            )
        return ResultadoRecuperacao.model_validate(bruto)

    @staticmethod
    def _plano(estado: EstadoRadar) -> PlanoConsulta:
        bruto = estado.get("plano_consulta")
        if bruto is None:
            raise ErroExtractor("o Extractor exige um PlanoConsulta no estado")
        return PlanoConsulta.model_validate(bruto)

    @staticmethod
    def _startup_alvo(estado: EstadoRadar, resultado: ResultadoRecuperacao) -> int:
        candidatas = {empresa.id_startup for empresa in resultado.empresas}
        selecionada = estado.get("startup_selecionada")
        if selecionada is not None:
            if int(selecionada) not in candidatas:
                raise ErroExtractor(
                    f"a startup {selecionada} não está no resultado da recuperação"
                )
            return int(selecionada)
        if len(candidatas) != 1:
            raise ErroExtractor(
                "o Extractor analisa uma startup por invocação; a recuperação "
                f"trouxe {len(candidatas)} empresas e nenhuma foi selecionada"
            )
        return candidatas.pop()

    @staticmethod
    def _modo_estrito(estado: EstadoRadar) -> bool:
        """Espelha exatamente o predicado de R2, sem duplicar a regra.

        Uma extração fresca não tem perfil validado anterior e nunca é estrita.
        """
        bruto = estado.get("perfil_validado")
        if bruto is None:
            return False
        try:
            perfil = PerfilValidado.model_validate(bruto)
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroExtractor(
                "o perfil validado anterior está fora do contrato e não permite "
                f"reextração segura: {erro}"
            ) from erro
        bruto_classificacao = estado.get("classificacao")
        if bruto_classificacao is None:
            raise ErroExtractor(
                "um perfil validado anterior exige a classificação correspondente "
                "para decidir a reextração com segurança"
            )
        try:
            classificacao = Classificacao.model_validate(bruto_classificacao)
        except (ValidationError, ValueError, TypeError) as erro:
            raise ErroExtractor(
                "a classificação anterior está fora do contrato e não permite "
                f"reextração segura: {erro}"
            ) from erro
        return precisa_reextrair(perfil, classificacao)

    # ------------------------------------------------------------------
    # Fronteira do LLM
    # ------------------------------------------------------------------

    def _extrair_com_validacao(
        self,
        id_startup: int,
        empresa: EmpresaCandidata | None,
        plano: PlanoConsulta,
        documentos: list[DocumentoIntegral],
        modo_estrito: bool = False,
    ) -> PerfilExtraido:
        erro_anterior: str | None = None
        for tentativa in range(2):
            mensagens = self._montar_mensagens(
                id_startup,
                empresa,
                plano,
                documentos,
                erro_anterior,
                modo_estrito,
            )
            try:
                bruto = self.provedor.invocar(mensagens)
            except (ValidationError, ValueError, TypeError) as exc:
                # O adaptador de structured output pode validar com Pydantic antes
                # de devolver o objeto. Essa falha ainda é uma resposta fora do
                # contrato e, portanto, consome o mesmo retry corretivo.
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroExtractor(
                        "O Gemini respondeu duas vezes fora do contrato estruturado; "
                        "nenhum perfil foi gravado no estado."
                    ) from exc
                continue
            except Exception as exc:
                raise ErroExtractor(
                    "O Gemini não respondeu ao Extractor; nenhum perfil foi fabricado."
                ) from exc
            try:
                return self._validar(bruto, id_startup, documentos)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroExtractor(
                        "O Gemini respondeu duas vezes fora do contrato estruturado; "
                        "nenhum perfil foi gravado no estado."
                    ) from exc
        raise AssertionError("laço de validação terminou em estado impossível")

    @staticmethod
    def _validar(
        bruto: object, id_startup: int, documentos: list[DocumentoIntegral]
    ) -> PerfilExtraido:
        """Confere estrutura e escopo — nunca proveniência literal.

        A conferência do trecho contra o texto completo pertence exclusivamente
        ao Evidence Validator. Duplicá-la aqui faria o Extractor abortar
        primeiro, deixando ``taxa_derrubada`` estruturalmente em zero e o laço
        R2 inalcançável em produção.
        """
        perfil = PerfilExtraido.model_validate(bruto)
        if perfil.id_startup != id_startup:
            raise ValueError(
                f"id_startup {perfil.id_startup} difere da startup analisada {id_startup}"
            )
        ids_permitidos = {documento.id_documento for documento in documentos}
        for afirmacao in perfil.afirmacoes:
            if afirmacao.id_documento not in ids_permitidos:
                raise ValueError(
                    f"a afirmação {afirmacao.id_afirmacao} cita o documento "
                    f"{afirmacao.id_documento}, fora do conjunto permitido "
                    f"{sorted(ids_permitidos)}"
                )
        return perfil

    @staticmethod
    def _resumir_erro(erro: Exception) -> str:
        if isinstance(erro, ValidationError):
            # loc vazio acontece em validador de modelo; sem o nome nem a mensagem
            # a instrução de correção não diria ao modelo o que consertar.
            return "; ".join(
                f"{'.'.join(str(item) for item in falha['loc']) or 'perfil'}: "
                f"{falha['msg']}"
                for falha in erro.errors()
            )
        return str(erro)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _instrucao(
        id_startup: int,
        plano: PlanoConsulta,
        ids_permitidos: list[int],
        modo_estrito: bool = False,
    ) -> str:
        estruturais = ", ".join(sorted(CATEGORIAS_ESTRUTURAIS))
        instrucao = (
            "Você é o Extractor do NVIDIA Startup AI Radar. Produza um PerfilExtraido "
            "estritamente estruturado sobre a startup indicada, usando exclusivamente "
            "os documentos fornecidos nesta mensagem.\n"
            f"- id_startup deve ser exatamente {id_startup}.\n"
            "- resumo_produto: 2 ou 3 frases sobre o que a empresa vende, cada uma "
            "terminada em ponto.\n"
            "- afirmacoes: de 1 a 20 fatos; id_afirmacao sequencial a partir de 1, na "
            "ordem da lista.\n"
            "- texto: um único fato, em uma frase.\n"
            f"- categoria: um destes dez valores: {', '.join(CATEGORIAS_AFIRMACAO)}.\n"
            f"- nas categorias estruturais ({estruturais}), use 'presenca' para "
            "capacidade observada, 'ausencia_explicita' para gap declarado e "
            "'neutro' quando o documento citar o tema sem permitir concluir "
            "capacidade ou gap; em todas as demais categorias use 'neutro'.\n"
            f"- id_documento: um dos ids fornecidos: {ids_permitidos}.\n"
            "- trecho_citado: substring literal e contígua do documento citado, com "
            f"{MINIMO_CARACTERES_TRECHO_CITADO} a {LIMITE_TRECHO_CITADO} caracteres "
            f"e ao menos {MINIMO_PALAVRAS_TRECHO_CITADO} palavras, copiada sem "
            "reescrever, resumir, corrigir ou traduzir.\n"
            "- Não afirme nada que os documentos não sustentem literalmente.\n"
            "- Silêncio não é ausência: se os documentos não mencionam algo, não gere "
            "afirmação alguma sobre isso. Use 'ausencia_explicita' somente quando um "
            "documento afirmar explicitamente que algo não existe, não é usado ou "
            "está faltando, e cite esse trecho.\n"
            "- Não classifique a maturidade de IA da empresa; isso não é tarefa do "
            "Extractor.\n"
            f"Foco da análise: {plano.foco_analise}"
        )
        if modo_estrito:
            instrucao += (
                "\n- REEXTRAÇÃO ESTRITA: a validação anterior rejeitou a evidência "
                "produzida. Reduza as afirmações ao que possuir trecho literal "
                "inequívoco nos documentos permitidos, copiado caractere a "
                "caractere, incluindo pontuação e acentuação."
            )
        return instrucao

    @staticmethod
    def _dados(
        id_startup: int,
        empresa: EmpresaCandidata | None,
        documentos: list[DocumentoIntegral],
    ) -> str:
        identidade = (
            f"Startup analisada: {empresa.nome} (id {id_startup}); setor "
            f"{empresa.setor}; estágio {empresa.estagio}; localização "
            f"{empresa.localizacao or 'não informada'}."
            if empresa is not None
            else f"Startup analisada: id {id_startup}."
        )
        blocos = "\n\n".join(
            f"[documento {documento.id_documento} | tipo: {documento.tipo} | "
            f"título: {documento.titulo}]\n{documento.conteudo_texto}"
            for documento in documentos
        )
        return f"{identidade}\n\nDocumentos permitidos:\n\n{blocos}"

    def _montar_mensagens(
        self,
        id_startup: int,
        empresa: EmpresaCandidata | None,
        plano: PlanoConsulta,
        documentos: list[DocumentoIntegral],
        erro_anterior: str | None,
        modo_estrito: bool = False,
    ) -> list[tuple[str, str]]:
        ids_permitidos = [documento.id_documento for documento in documentos]
        mensagens = [
            (
                "system",
                self._instrucao(id_startup, plano, ids_permitidos, modo_estrito),
            ),
            ("human", self._dados(id_startup, empresa, documentos)),
        ]
        if erro_anterior:
            mensagens.append(
                (
                    "system",
                    "A resposta anterior violou o contrato. Corrija sem texto livre. "
                    f"Falha de validação: {erro_anterior}",
                )
            )
        return mensagens
