from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from radar.agentes.ancoras_sinal import (
    completar_sinais_inequivocos,
    filtrar_sinais_sem_ancora,
    sinais_possivelmente_nao_extraidos,
    sinais_possivelmente_omitidos,
)
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
from radar.provedores import ErroReservaIncompativel, ProvedorPerfilExtraido


logger = logging.getLogger(__name__)

# As seis categorias em que a polaridade não carrega informação. O contrato
# aceita nelas um único valor, e é isso que torna a correção determinística.
CATEGORIAS_NAO_ESTRUTURAIS: tuple[str, ...] = tuple(
    categoria
    for categoria in CATEGORIAS_AFIRMACAO
    if categoria not in CATEGORIAS_ESTRUTURAIS
)

# Valores que o provedor às vezes preenche por analogia com as estruturais.
POLARIDADES_REDUNDANTES: frozenset[str] = frozenset(
    {"presenca", "ausencia_explicita"}
)

POLARIDADE_CANONICA = "neutro"


def normalizar_polaridade_nao_estrutural(bruto: object) -> object:
    """Corrige a polaridade redundante das categorias sem semântica de gap.

    A §11 do contrato reserva ``presenca`` e ``ausencia_explicita`` às quatro
    dimensões estruturais; nas outras seis categorias ``neutro`` é o único
    valor válido. Quando o provedor preenche uma delas por analogia, a resposta
    está fora da forma, mas nenhum fato está errado: o texto, o trecho literal,
    a categoria e a proveniência continuam os mesmos. Trocar só esse campo é
    determinístico e preserva o significado — recusar consome o único retry e
    derruba a análise inteira por uma questão de forma.

    Nada mais é reparado. Categoria estrutural, categoria desconhecida,
    polaridade ausente, polaridade irreconhecível e objeto malformado seguem
    para a validação, o retry e a falha segura que já existem.
    """
    if not isinstance(bruto, dict):
        return bruto
    afirmacoes = bruto.get("afirmacoes")
    if not isinstance(afirmacoes, list):
        return bruto
    corrigidas = [_normalizar_afirmacao(afirmacao) for afirmacao in afirmacoes]
    alteradas = sum(
        1
        for original, corrigida in zip(afirmacoes, corrigidas)
        if original is not corrigida
    )
    if not alteradas:
        return bruto
    logger.info(
        "polaridade redundante normalizada para 'neutro' em %s afirmação(ões) "
        "de categoria não estrutural",
        alteradas,
    )
    copia = dict(bruto)
    copia["afirmacoes"] = corrigidas
    return copia


def _normalizar_afirmacao(afirmacao: object) -> object:
    if not isinstance(afirmacao, dict):
        return afirmacao
    if afirmacao.get("categoria") not in CATEGORIAS_NAO_ESTRUTURAIS:
        return afirmacao
    if afirmacao.get("polaridade") not in POLARIDADES_REDUNDANTES:
        return afirmacao
    corrigida = dict(afirmacao)
    corrigida["polaridade"] = POLARIDADE_CANONICA
    return corrigida


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
                        "O provedor de IA respondeu duas vezes fora do contrato "
                        "estruturado; nenhum perfil foi gravado no estado."
                    ) from exc
                continue
            except ErroReservaIncompativel as exc:
                # A reserva recusou o **pedido** estruturado (HTTP 400), não caiu.
                # Isso é falha de contrato na fronteira de provedores, e falha de
                # contrato é exatamente o que a tentativa corretiva deste nó existe
                # para absorver — consome a mesma, nunca uma terceira. O prompt não
                # ganha aviso de correção: o modelo não respondeu nada errado.
                if tentativa == 1:
                    raise ErroExtractor(
                        "O provedor de IA respondeu duas vezes fora do contrato "
                        "estruturado; nenhum perfil foi gravado no estado."
                    ) from exc
                continue
            except Exception as exc:
                raise ErroExtractor(
                    "O provedor de IA não respondeu ao Extractor; "
                    "nenhum perfil foi fabricado."
                ) from exc
            try:
                perfil = self._validar(bruto, id_startup, documentos)
            except (ValidationError, ValueError, TypeError) as exc:
                erro_anterior = self._resumir_erro(exc)
                if tentativa == 1:
                    raise ErroExtractor(
                        "O provedor de IA respondeu duas vezes fora do contrato "
                        "estruturado; nenhum perfil foi gravado no estado."
                    ) from exc
                continue
            omitidos = sinais_possivelmente_omitidos(perfil)
            nao_extraidos = sinais_possivelmente_nao_extraidos(perfil, documentos)
            if (omitidos or nao_extraidos) and tentativa == 0:
                detalhes_afirmacoes = "; ".join(
                    f"afirmação {id_afirmacao}: {', '.join(sinais)}"
                    for id_afirmacao, sinais in omitidos
                )
                detalhes_documentos = "; ".join(
                    f"documento {id_documento}: {', '.join(sinais)}"
                    for id_documento, sinais in nao_extraidos
                )
                detalhes = "; ".join(
                    item
                    for item in (detalhes_afirmacoes, detalhes_documentos)
                    if item
                )
                erro_anterior = (
                    "sinais técnicos possivelmente omitidos apesar de âncora "
                    f"literal inequívoca ({detalhes}). Reavalie essa afirmação "
                    "ou documento contra toda a matriz; extraia o fato quando "
                    "ele realmente qualificar e anexe somente os sinais "
                    "sustentados pelo próprio trecho_citado"
                )
                continue
            if omitidos:
                perfil = completar_sinais_inequivocos(perfil)
            return perfil
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
        # Duas correções determinísticas antes do contrato: a polaridade
        # redundante das categorias não estruturais e o sinal técnico sem
        # âncora literal. Nenhuma das duas inventa fato; ambas removem o que a
        # fonte não sustenta, preservando a afirmação inteira.
        perfil = PerfilExtraido.model_validate(
            filtrar_sinais_sem_ancora(normalizar_polaridade_nao_estrutural(bruto))
        )
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
        nao_estruturais = ", ".join(CATEGORIAS_NAO_ESTRUTURAIS)
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
            "- Quando os documentos descreverem o produto ou serviço vendido, inclua "
            "ao menos uma afirmação positiva sobre essa oferta: use "
            "'workflow_profundo' para processo ou serviço integrado entregue ao "
            "cliente, 'stack_propria' para componentes técnicos próprios e "
            "'distribuicao' para canais ou acesso comercial. Use 'outro' somente "
            "quando nenhuma categoria definida representar literalmente o fato; "
            "não invente uma afirmação de produto se a fonte não a sustentar.\n"
            "- MATRIZ DE POLARIDADE POR CATEGORIA, obrigatória e sem exceção:\n"
            f"  * categorias estruturais ({estruturais}): use 'presenca' para "
            "capacidade observada, 'ausencia_explicita' para gap declarado ou "
            "'neutro' quando o documento citar o tema sem permitir concluir "
            "capacidade ou gap.\n"
            f"  * categorias não estruturais ({nao_estruturais}): use sempre "
            "'neutro'. Nessas categorias a polaridade não tem significado e "
            "nenhum outro valor é aceito.\n"
            "- SINAIS TÉCNICOS (campo sinais_tecnicos, opcional): avalie "
            "obrigatoriamente cada afirmação contra os dez sinais abaixo e "
            "anexe todos os sinais qualificados na mesma afirmação. Não deixe "
            "a lista vazia quando o próprio trecho_citado nomear uma das cargas "
            "de trabalho definidas. Anexe um "
            "sinal a uma afirmação SOMENTE quando o trecho_citado dela nomear "
            "literalmente a carga de trabalho. Setor não cria sinal — atuar "
            "num mercado não é executar uma carga. Menção genérica do tipo "
            "'usa IA' ou 'usa inteligência artificial', sem nomear a carga, "
            "também não cria sinal. Nenhum sinal continua sendo um resultado "
            "válido depois dessa auditoria completa, e é sempre preferível a "
            "inferir uma carga que a fonte não nomeou.\n"
            "- Antes de finalizar, percorra todos os documentos permitidos e "
            "procure fatos que nomeiem alguma das dez cargas técnicas. Se um "
            "documento trouxer uma carga qualificável, inclua ao menos uma "
            "afirmação literal sobre ela; não resuma o perfil apenas ao produto "
            "geral e ao financiamento.\n"
            "  * inferencia_llm: qualifica quando o texto nomeia modelo de "
            "linguagem, LLM, IA generativa ou RAG dentro do produto. NÃO "
            "qualifica: 'assistente com IA'.\n"
            "  * treinamento_ou_finetuning: qualifica quando nomeia treino de "
            "modelo, pré-treino contínuo ou ajuste fino. NÃO qualifica: "
            "'algoritmo próprio'.\n"
            "  * voz_fala_ou_transcricao: qualifica quando nomeia voz, fala, "
            "áudio, transcrição ou call center. NÃO qualifica: 'atendimento "
            "por aplicativo'.\n"
            "  * dados_em_escala: qualifica quando declara volume de dados ou "
            "de amostras processadas. NÃO qualifica: 'dados proprietários' sem "
            "volume declarado.\n"
            "  * machine_learning_classico: qualifica quando nomeia modelo "
            "preditivo, score ou técnica de ML sobre atributos estruturados "
            "citados. NÃO qualifica: 'usa IA e machine learning' sem dado nem "
            "modelo nomeado.\n"
            "  * visao_computacional: qualifica quando nomeia imagem, vídeo, "
            "OCR, reconhecimento facial ou visão computacional. NÃO qualifica: 'inspeção de "
            "qualidade'.\n"
            "  * robotica_ou_simulacao: qualifica quando nomeia robô, veículo "
            "autônomo, simulação ou gêmeo digital. NÃO qualifica: 'automação "
            "de processos'.\n"
            "  * imagem_medica: qualifica quando nomeia exame de imagem, "
            "radiologia ou patologia digital. NÃO qualifica: 'atua em "
            "saúde'.\n"
            "  * agentes_com_acoes_ou_controles: qualifica quando nomeia "
            "sistema multiagente, agente que executa ações, verificação de "
            "citação ou human-in-the-loop. NÃO qualifica: 'chatbot'.\n"
            "  * ciberseguranca_em_escala: qualifica quando nomeia detecção de "
            "ameaça, intrusão ou análise de telemetria de segurança. NÃO "
            "qualifica: 'criptografia' ou 'HSM'.\n"
            "- MODALIDADE DA FONTE, obrigatória: preserve se o documento "
            "descreve capacidade **atual**, **requisito** ou **responsabilidade "
            "futura**. Documento de vaga descreve o que o cargo vai fazer, não "
            "o que a empresa já faz: escreva 'A vaga...', 'O cargo...' ou "
            "'A empresa busca...', nunca 'A empresa faz...'. Uma vaga pode "
            "sustentar sinal técnico quando a carga de trabalho está nomeada no "
            "trecho, mas jamais como prova de que a capacidade já opera no "
            "produto. O mesmo vale para proposta, plano ou projeto anunciado: "
            "proposta não é operação.\n"
            f"- id_documento: um dos ids fornecidos: {ids_permitidos}.\n"
            "- trecho_citado: substring literal e contígua exclusivamente do "
            "conteudo_texto do documento citado; nunca copie do título, tipo, URL "
            "ou qualquer outro metadado. O trecho deve ter "
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
            f"título informativo (proibido citar): {documento.titulo}]\n"
            f"CONTEUDO_TEXTO (única área válida para trecho_citado):\n"
            f"{documento.conteudo_texto}"
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
