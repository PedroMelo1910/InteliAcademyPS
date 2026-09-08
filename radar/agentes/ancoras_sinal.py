"""Porta determinística: sinal técnico exige âncora literal no trecho citado.

O lote ao vivo provou que a instrução de prompt não basta. O Gemini anexou
``visao_computacional`` a "manejo digital de plantas daninhas" e
``inferencia_llm`` a "a inteligência artificial da empresa" — nos dois casos a
proveniência formal fechava (o trecho existe no documento) e o lastro sobre a
carga de trabalho não existia.

Esta porta **não** é um classificador de verdade semântica. Ela só recusa a
atribuição que não traz o mínimo de evidência explícita que a matriz congelada
exige. Um sinal sem âncora é removido individualmente; a afirmação sobrevive
inteira, com texto, categoria, polaridade, documento e trecho intactos, e a
``taxa_derrubada`` do Evidence Validator não é afetada — derrubar afirmação e
remover sinal são coisas diferentes.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from radar.contratos import DocumentoIntegral, SINAIS_TECNICOS, PerfilExtraido


logger = logging.getLogger(__name__)

# Âncoras em forma normalizada (minúscula, sem acento). São condição
# **necessária**, nunca suficiente: a presença do termo apenas autoriza o sinal
# a continuar; nada aqui afirma que a startup faz aquilo bem.
ANCORAS_POR_SINAL: dict[str, tuple[str, ...]] = {
    # "usa IA" e "agente de IA" ficam de fora de propósito: nomeiam o rótulo,
    # não a carga de trabalho de modelo de linguagem.
    "inferencia_llm": (
        "modelo de linguagem",
        "modelos de linguagem",
        "llm",
        "generativ",
        "rag",
        "retrieval-augmented",
        "retrieval augmented",
    ),
    "treinamento_ou_finetuning": (
        "treina",
        "pre-treino",
        "pretreino",
        "fine-tuning",
        "finetuning",
        "ajuste fino",
    ),
    "voz_fala_ou_transcricao": (
        "voz",
        "fala",
        "audio",
        "transcri",
        "call center",
        "speech",
        "text-to-speech",
    ),
    # Escala precisa de número: "dados proprietários" sozinho não qualifica.
    "dados_em_escala": ("bilh", "milh", "terabyte", "petabyte"),
    "machine_learning_classico": (
        "machine learning",
        "aprendizado de maquina",
        "modelo predit",
        "modelos predit",
        "modelo analit",
        "modelos analit",
        "regressao",
        "classificador",
    ),
    "visao_computacional": (
        "visao computacional",
        "computer vision",
        "imagem",
        "imagens",
        "video",
        "ocr",
        "foto",
        "camera",
        "reconhecimento facial",
    ),
    "robotica_ou_simulacao": (
        "robo",
        "robotic",
        "autonom",
        "sem motorista",
        "simulac",
        "gemeo digital",
        "digital twin",
    ),
    "imagem_medica": (
        "radiolog",
        "tomograf",
        "ressonanc",
        "raio-x",
        "raio x",
        "patologia digital",
        "exame de imagem",
        "exames de imagem",
        "imagem medica",
        "dicom",
    ),
    # "agente" sozinho não basta: o que qualifica é ação executada, arquitetura
    # multiagente ou controle nomeado.
    "agentes_com_acoes_ou_controles": (
        "multiagente",
        "multi-agente",
        "human-in-the-loop",
        "human in the loop",
        "guardrail",
        "executa acao",
        "executa acoes",
        "verificac",
        "supervisao de especialistas",
        "fontes utilizadas",
    ),
    "ciberseguranca_em_escala": (
        "ameaca",
        "intrusao",
        "deteccao de incidente",
        "trafego de rede",
        "telemetria de seguranca",
        "malware",
        "defesa cibernetica",
    ),
}

# Sinais cuja âncora só vale acompanhada de um número declarado na fonte.
SINAIS_QUE_EXIGEM_NUMERO: frozenset[str] = frozenset({"dados_em_escala"})

_TERMOS_DE_VOLUME: tuple[str, ...] = (
    "dad",
    "amostr",
    "document",
    "image",
    "video",
    "registr",
    "transa",
    "event",
    "usuari",
    "pessoa",
    "client",
    "pacient",
)
_TERMOS_MONETARIOS: tuple[str, ...] = (
    "r$",
    "us$",
    "dolar",
    "reais",
    "aporte",
    "captou",
    "captacao",
    "rodada",
    "valuation",
)
# Vocabulário fechado de negação. Cada entrada é uma frase, nunca a partícula
# "não" solta, para que uma negativa alheia não apague uma afirmação válida.
# As formas de indicativo já cobrem por prefixo as flexões vizinhas ("nao
# utiliza" casa "nao utilizar" e "nao utilizam"); subjuntivo e imperativo
# precisam de entrada própria, porque a construção concessiva os exige —
# "embora não utilize", "apesar de não usar", "desde que não possua".
_NEGACOES: tuple[str, ...] = (
    "nao utiliza",
    "nao utilize",
    "nao usa",
    "nao use",
    "nao possui",
    "nao possua",
    "sem utilizar",
    "sem uso",
    "deixou de",
)
_SUJEITOS_AGENTICOS: tuple[str, ...] = ("agente", "assistente")
_ACOES_AGENTICAS: tuple[str, ...] = (
    "executa",
    "realiza",
    "automatiza",
    "analisa",
    "otimiza",
    "evolui",
    "emite",
    "paga",
    "consulta",
    "cadastra",
    "orquestra",
    "gerencia",
)
_CONTEXTO_ML_ESTRUTURADO: tuple[str, ...] = (
    "dado",
    "atributo",
    "risco",
    "fraude",
    "imposto",
    "erro",
    "inconsistencia",
    "telemetria",
    "gps",
    "sensor",
    "nota",
    "score",
    "preco",
)

# Esta segunda tabela é deliberadamente menor que ``ANCORAS_POR_SINAL``. A
# primeira é uma porta negativa (remove atribuições sem lastro mínimo); esta
# apenas detecta uma omissão muito provável para pedir ao próprio modelo uma
# única revisão. Termos ambíguos como "imagem", "vídeo", "autônomo" ou
# "machine learning" isolado ficam de fora para não transformar o mecanismo
# em um classificador oculto.
ANCORAS_INEQUIVOCAS_DE_OMISSAO: dict[str, tuple[str, ...]] = {
    "inferencia_llm": (
        "modelo de linguagem",
        "modelos de linguagem",
        "llm",
        "ia generativa",
        "inteligencia artificial generativa",
        "retrieval-augmented",
        "retrieval augmented",
    ),
    "treinamento_ou_finetuning": (
        "pre-treino",
        "pretreino",
        "fine-tuning",
        "finetuning",
        "ajuste fino",
    ),
    "voz_fala_ou_transcricao": (
        "transcri",
        "reconhecimento de fala",
        "sintese de voz",
        "text-to-speech",
        "speech-to-text",
        "call center",
    ),
    "dados_em_escala": ("bilh", "milh", "terabyte", "petabyte"),
    "machine_learning_classico": (
        "modelo predit",
        "modelos predit",
        "regressao",
        "classificador",
        "nota de direcao",
        "score de risco",
    ),
    "visao_computacional": (
        "visao computacional",
        "computer vision",
        "ocr",
        "reconhecimento de movimento",
        "reconhecimento facial",
    ),
    "robotica_ou_simulacao": (
        "robo industrial",
        "robos industriais",
        "sistema robotico",
        "plataforma robotica",
        "sem motorista",
        "gemeo digital",
        "digital twin",
    ),
    "imagem_medica": ANCORAS_POR_SINAL["imagem_medica"],
    "agentes_com_acoes_ou_controles": (
        "multiagente",
        "multi-agente",
        "human-in-the-loop",
        "human in the loop",
        "guardrail",
        "executa acao",
        "executa acoes",
        "agentops",
    ),
    "ciberseguranca_em_escala": (
        "deteccao de ameaca",
        "deteccao de intrusao",
        "deteccao de incidente",
        "telemetria de seguranca",
        "defesa cibernetica",
    ),
}


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(
        letra for letra in decomposto if not unicodedata.combining(letra)
    ).casefold()


# Fronteiras de oração: pontuação de frase mais os conectivos adversativos que
# viram a afirmação em português. Vírgula e "e" ficam de fora de propósito —
# eles ligam enumerações ("não utiliza visão computacional, robótica e imagem
# médica"), e cortar ali libertaria da negação tudo o que vem depois da
# primeira tecnologia. É uma régua de escopo, não uma análise sintática.
_SEPARADORES_DE_ORACAO = re.compile(
    r"[.;:!?]"
    r"|\bmas\b|\bporem\b|\bcontudo\b|\btodavia\b|\bentretanto\b"
    r"|\benquanto\b|\bembora\b|\bapesar\b"
)

_ESPACOS = re.compile(r"\s+")

# Conectivos que abrem uma subordinada. Quando eles iniciam a frase, quem fecha
# a subordinada e abre a principal é a vírgula — e vírgula, sozinha, não é
# fronteira de oração aqui.
_CONECTIVO_SUBORDINATIVO = re.compile(r"\b(?:embora|apesar de|apesar|enquanto)\b")

# Coordenadores que ligam itens de uma enumeração: ``e``, ``ou`` e ``nem``.
_COORDENADOR_DE_ENUMERACAO = re.compile(r"^(?:e|ou|nem)\s+")

# União das âncoras técnicas do vocabulário fechado. Ordenada só para que a
# constante seja reproduzível; ``str.startswith`` aceita a tupla inteira.
_TODAS_AS_ANCORAS: tuple[str, ...] = tuple(
    sorted({ancora for ancoras in ANCORAS_POR_SINAL.values() for ancora in ancoras})
)


def _continua_enumeracao_tecnica(segmento: str) -> bool:
    """O trecho após a vírgula é mais um item da lista, não uma oração nova.

    Quem separa os dois casos é o próprio vocabulário fechado de âncoras: um
    item de enumeração **começa** por âncora técnica ("robótica autônoma e
    imagem médica"), enquanto a oração principal começa por sujeito ou verbo
    ("a empresa usa modelos de linguagem"). Basta olhar o início — exigir só
    que a âncora apareça em algum lugar confundiria os dois, porque a principal
    afirmativa também nomeia tecnologia.
    """
    item = _COORDENADOR_DE_ENUMERACAO.sub("", segmento.strip(), count=1)
    return item.startswith(_TODAS_AS_ANCORAS)


def _fechar_subordinadas(texto: str) -> str:
    """Promove a vírgula que fecha uma subordinada a fronteira de oração.

    ``Embora não utilize X, a empresa usa Y`` põe a negação na subordinada e a
    afirmação na principal, separadas só por vírgula. Sem este passo as duas
    ficariam na mesma oração e a negação contaminaria a afirmação — ou, na
    ordem inversa, a afirmação salvaria a negação.

    A vírgula que fecha é a primeira depois do conectivo que **não** continua
    uma enumeração técnica. Uma lista negativa em português liga seus itens por
    vírgula mais ``e``, ``ou`` ou ``nem``, e todos eles seguem sob a negação:
    ``não utilize A, B e C, a empresa usa D`` mantém A, B e C negados e liberta
    só a principal. Escolher a última vírgula não serviria — a principal
    afirmativa pode ter lista própria (``usa D, E e F``). Nenhuma outra vírgula
    é tocada, e enumerações afirmativas seguem inteiras. É uma régua de escopo
    local, não uma análise sintática.
    """
    cortes: set[int] = set()
    for conectivo in _CONECTIVO_SUBORDINATIVO.finditer(texto):
        virgula = texto.find(",", conectivo.end())
        while virgula != -1:
            proxima = texto.find(",", virgula + 1)
            limite = len(texto) if proxima == -1 else proxima
            if _continua_enumeracao_tecnica(texto[virgula + 1 : limite]):
                virgula = proxima
                continue
            cortes.add(virgula)
            break
    if not cortes:
        return texto
    return "".join(
        "." if indice in cortes else caractere
        for indice, caractere in enumerate(texto)
    )


def _oracoes(texto_normalizado: str) -> tuple[str, ...]:
    """Divide o trecho já normalizado nas orações que delimitam a negação.

    O espaço em branco é colapsado antes de tudo porque o ``trecho_citado`` é
    substring literal do documento: quebra de linha e indentação atravessam o
    texto e fariam ``"nao  utiliza"`` escapar do vocabulário fechado, além de
    impedir o casamento de ``apesar de``.
    """
    texto = _fechar_subordinadas(_ESPACOS.sub(" ", texto_normalizado))
    partes = (parte.strip() for parte in _SEPARADORES_DE_ORACAO.split(texto))
    return tuple(parte for parte in partes if parte)


def _nega_explicitamente(texto_normalizado: str) -> bool:
    """Porta única de negação, compartilhada por todas as âncoras.

    O vocabulário é fechado e deliberadamente estreito: são frases que negam
    uso ("nao utiliza", "sem uso"), não a partícula "não" solta. Um trecho que
    afirma a carga de trabalho e por acaso contém outra negativa continua
    válido — a porta recusa atribuição sem lastro, não frases negativas.
    """
    return any(negacao in texto_normalizado for negacao in _NEGACOES)


def _tem_ancora_inequivoca(sinal: str, texto_normalizado: str) -> bool:
    if _nega_explicitamente(texto_normalizado):
        return False
    if sinal == "dados_em_escala":
        return (
            any(caractere.isdigit() for caractere in texto_normalizado)
            and any(
                escala in texto_normalizado
                for escala in ANCORAS_INEQUIVOCAS_DE_OMISSAO[sinal]
            )
            and any(termo in texto_normalizado for termo in _TERMOS_DE_VOLUME)
            and not any(
                monetario in texto_normalizado for monetario in _TERMOS_MONETARIOS
            )
        )
    if sinal == "agentes_com_acoes_ou_controles":
        direto = any(
            ancora in texto_normalizado
            for ancora in ANCORAS_INEQUIVOCAS_DE_OMISSAO[sinal]
        )
        composto = any(
            sujeito in texto_normalizado for sujeito in _SUJEITOS_AGENTICOS
        ) and any(acao in texto_normalizado for acao in _ACOES_AGENTICAS)
        return direto or composto
    if sinal == "machine_learning_classico":
        direto = any(
            ancora in texto_normalizado
            for ancora in ANCORAS_INEQUIVOCAS_DE_OMISSAO[sinal]
        )
        composto = (
            "machine learning" in texto_normalizado
            or "aprendizado de maquina" in texto_normalizado
        ) and any(
            contexto in texto_normalizado for contexto in _CONTEXTO_ML_ESTRUTURADO
        )
        return direto or composto
    return any(
        ancora in texto_normalizado
        for ancora in ANCORAS_INEQUIVOCAS_DE_OMISSAO[sinal]
    )


def lista_estruturalmente_valida(sinais: object) -> bool:
    """Só uma lista já válida pelo contrato pode passar pela porta de âncora.

    Duplicata, valor fora do enum e tipo errado são **violação de contrato**, não
    ruído a limpar: reparar isso em silêncio esconderia do Pydantic — e do retry
    corretivo do agente — uma resposta estruturalmente errada.
    """
    if not isinstance(sinais, (list, tuple)):
        return False
    if not all(isinstance(item, str) for item in sinais):
        return False
    if len(set(sinais)) != len(sinais):
        return False
    return all(item in SINAIS_TECNICOS for item in sinais)


def sinais_com_ancora(sinais, trecho_citado: str) -> tuple[str, ...]:
    """Subconjunto dos sinais cujo trecho traz a âncora exigida pela matriz.

    Preserva a ordem declarada: a ordenação canônica é responsabilidade do
    contrato Pydantic, não desta porta.
    """
    normalizado = _normalizar(trecho_citado)
    # A negação explícita fecha a porta por **oração**, não pelo trecho todo.
    # Um trecho real afirma uma carga de trabalho e nega outra na mesma linha
    # ("usa modelos de linguagem, mas não utiliza visão computacional"): ali a
    # negação pertence a uma tecnologia só, e derrubar as duas perderia
    # evidência legítima. O vocabulário de negação continua o mesmo — frases
    # fechadas como "nao utiliza" e "sem uso", nunca um "não" solto.
    oracoes = _oracoes(normalizado)
    tem_numero = any(caractere.isdigit() for caractere in normalizado)
    mantidos = []
    for sinal in sinais:
        if sinal in SINAIS_QUE_EXIGEM_NUMERO and not tem_numero:
            continue
        # Escala precisa de uma relação composta, não apenas de uma palavra
        # solta. Assim, "R$ 300 milhões" não vira volume de dados mesmo quando
        # o provedor envia esse sinal explicitamente. ``_tem_ancora_inequivoca``
        # já recusa oração negada, então avaliá-la por oração dá de graça o
        # mesmo escopo dos demais sinais, sem afrouxar a regra composta.
        if sinal == "dados_em_escala":
            if any(_tem_ancora_inequivoca(sinal, oracao) for oracao in oracoes):
                mantidos.append(sinal)
            continue
        # Basta uma oração afirmativa com âncora para sustentar o sinal; se as
        # âncoras só aparecem em orações negadas, o sinal cai.
        if any(
            not _nega_explicitamente(oracao)
            and any(ancora in oracao for ancora in ANCORAS_POR_SINAL[sinal])
            for oracao in oracoes
        ):
            mantidos.append(sinal)
    return tuple(mantidos)


def sinais_possivelmente_omitidos(
    perfil: PerfilExtraido,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Aponta apenas omissões fortemente ancoradas para uma revisão do LLM.

    O retorno não modifica o perfil e não cria oportunidade. Ele serve somente
    para gastar, no máximo, o retry estruturado já existente. Na segunda
    resposta a omissão continua sendo aceita: entre um falso positivo desta
    heurística e fabricar um sinal, prevalece sempre o perfil conservador.
    """
    resultado: list[tuple[int, tuple[str, ...]]] = []
    for afirmacao in perfil.afirmacoes:
        if afirmacao.polaridade == "ausencia_explicita":
            continue
        normalizado = _normalizar(afirmacao.trecho_citado)
        declarados = set(afirmacao.sinais_tecnicos)
        omitidos = []
        for sinal in SINAIS_TECNICOS:
            if sinal in declarados:
                continue
            if _tem_ancora_inequivoca(sinal, normalizado):
                omitidos.append(sinal)
        if omitidos:
            resultado.append((afirmacao.id_afirmacao, tuple(omitidos)))
    return tuple(resultado)


def sinais_possivelmente_nao_extraidos(
    perfil: PerfilExtraido,
    documentos: list[DocumentoIntegral],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Aponta carga inequívoca da fonte ausente de todo o perfil extraído.

    Diferentemente da canonicalização por afirmação, esta função nunca cria
    fato nem sinal: ela apenas permite gastar o retry existente pedindo ao LLM
    que releia o documento. Se a segunda resposta continuar conservadora, ela
    é aceita como tal.
    """
    declarados = {
        sinal
        for afirmacao in perfil.afirmacoes
        for sinal in afirmacao.sinais_tecnicos
    }
    # A guarda documental corrige somente o modo de falha observado ao vivo:
    # perfis que ignoraram *toda* a camada técnica apesar de uma âncora forte.
    # Se o perfil já extraiu algum sinal, comparar sua cobertura com cada
    # menção da fonte exigiria julgamento de completude semântico demais para
    # uma heurística determinística.
    if declarados:
        return ()
    resultado: list[tuple[int, tuple[str, ...]]] = []
    for documento in documentos:
        normalizado = _normalizar(documento.conteudo_texto)
        ausentes = tuple(
            sinal
            for sinal in SINAIS_TECNICOS
            if sinal not in declarados
            and _tem_ancora_inequivoca(sinal, normalizado)
        )
        if ausentes:
            resultado.append((documento.id_documento, ausentes))
    return tuple(resultado)


def completar_sinais_inequivocos(perfil: PerfilExtraido) -> PerfilExtraido:
    """Canonicaliza o rótulo técnico quando a própria citação já o nomeia.

    Esta é uma rede de segurança depois do único pedido de correção ao modelo.
    Ela não combina afirmações, não usa setor, não consulta fonte externa e não
    conclui capacidade a partir de silêncio. A afirmação e seu trecho continuam
    sujeitos ao Evidence Validator; apenas o vocabulário fechado do sinal é
    normalizado a partir de uma expressão inequívoca presente nos dois campos.
    """
    omitidos = dict(sinais_possivelmente_omitidos(perfil))
    if not omitidos:
        return perfil
    afirmacoes = []
    for afirmacao in perfil.afirmacoes:
        candidatos = omitidos.get(afirmacao.id_afirmacao, ())
        # A redundância no texto da afirmação impede que uma palavra incidental
        # do recorte seja promovida sem que o próprio fato extraído a descreva.
        texto_normalizado = _normalizar(afirmacao.texto)
        adicionar = tuple(
            sinal
            for sinal in candidatos
            if _tem_ancora_inequivoca(sinal, texto_normalizado)
        )
        if not adicionar:
            afirmacoes.append(afirmacao)
            continue
        declarados = set(afirmacao.sinais_tecnicos)
        declarados.update(adicionar)
        ordenados = tuple(sinal for sinal in SINAIS_TECNICOS if sinal in declarados)
        afirmacoes.append(
            afirmacao.model_copy(update={"sinais_tecnicos": ordenados})
        )
    return PerfilExtraido.model_validate(
        {
            "id_startup": perfil.id_startup,
            "resumo_produto": perfil.resumo_produto,
            "afirmacoes": [item.model_dump() for item in afirmacoes],
        }
    )


def filtrar_sinais_sem_ancora(bruto: object) -> object:
    """Remove sinais sem âncora, preservando a afirmação e a resposta original."""
    if not isinstance(bruto, dict):
        return bruto
    afirmacoes = bruto.get("afirmacoes")
    if not isinstance(afirmacoes, list):
        return bruto
    corrigidas = [_filtrar_afirmacao(afirmacao) for afirmacao in afirmacoes]
    removidos = sum(
        1
        for original, corrigida in zip(afirmacoes, corrigidas)
        if original is not corrigida
    )
    if not removidos:
        return bruto
    logger.info(
        "sinal técnico sem âncora literal removido em %s afirmação(ões); a "
        "afirmação foi preservada",
        removidos,
    )
    copia = dict(bruto)
    copia["afirmacoes"] = corrigidas
    return copia


def _filtrar_afirmacao(afirmacao: object) -> object:
    if not isinstance(afirmacao, dict):
        return afirmacao
    declarados = afirmacao.get("sinais_tecnicos")
    if not declarados:
        return afirmacao
    # Lista malformada segue intacta para o Pydantic recusar.
    if not lista_estruturalmente_valida(declarados):
        return afirmacao
    trecho = afirmacao.get("trecho_citado")
    if not isinstance(trecho, str):
        return afirmacao
    mantidos = list(sinais_com_ancora(declarados, trecho))
    if mantidos == list(declarados):
        return afirmacao
    corrigida = dict(afirmacao)
    corrigida["sinais_tecnicos"] = mantidos
    return corrigida
