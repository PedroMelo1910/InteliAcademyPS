"""Texto fixo da tela: o que o produto diz antes de qualquer dado chegar.

Nada aqui depende de startup, briefing ou consulta. São as frases que a
interface precisa dizer sozinha — propósito, perguntas de exemplo, a distinção
entre as duas medidas e os seis desfechos que a jornada pode ter. Ficam fora do
``app.py`` porque são conteúdo revisável sem tocar em orquestração, e fora de
``rotulos.py`` porque não traduzem nenhum objeto do domínio.

Todas as constantes são literais: nenhuma delas interpola consulta do usuário,
texto de fonte pública ou saída de modelo.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Identidade e propósito
# ----------------------------------------------------------------------

NOME_PRODUTO = "NVIDIA AI Radar"
PAGINA_RADAR = "Radar"
PAGINA_DASHBOARD = "Dashboard"
ROTULO_NAVEGACAO = "Navegação"
TITULO_DASHBOARD = "Visão geral da base"
LEGENDA_DASHBOARD = (
    "Panorama das 30 startups e das análises já verificadas pelo radar."
)

# ----------------------------------------------------------------------
# Busca
# ----------------------------------------------------------------------

ROTULO_CAMPO_CONSULTA = "O que você procura?"
ROTULO_BUSCAR = "Buscar candidatas"
PLACEHOLDER_CONSULTA = (
    "Ex.: empresas brasileiras com modelos de linguagem em português"
)
CONSULTA_EM_BRANCO = "Escreva uma consulta antes de buscar."
TITULO_EXEMPLOS = "Comece por uma destas perguntas"
PERGUNTAS_DE_EXEMPLO = (
    "fintechs brasileiras que treinam modelos próprios em produção",
    "startups de saúde usando visão computacional em imagem médica",
    "empresas de logística com otimização de rotas em tempo real",
)

MENSAGEM_CARREGANDO_BUSCA = "Carregando..."

# ----------------------------------------------------------------------
# As duas medidas, que nunca se misturam
# ----------------------------------------------------------------------

ROTULO_FIT_SCORE = "Fit-score NVIDIA"
ROTULO_BM25 = "Relação com a busca"
EXPLICACAO_FIT_SCORE = (
    "O fit-score mede aderência à stack NVIDIA sustentada por evidência "
    "pública — não é nota de qualidade da empresa. Ele só aparece quando a "
    "rubrica encontrou lastro, e um zero validado continua sendo um "
    "resultado da rubrica."
)

# ----------------------------------------------------------------------
# Painel do ranking
# ----------------------------------------------------------------------

TITULO_RANKING = "Candidatas priorizadas"
LEGENDA_RANKING = (
    "Escolha se a lista deve priorizar o fit-score NVIDIA ou a relação com a busca."
)
ROTULO_ORDENACAO = "Ordenar por"
OPCAO_ORDENAR_FIT_SCORE = "Maior fit-score NVIDIA"
OPCAO_ORDENAR_RELEVANCIA = "Maior relação com a busca"
ROTULO_FILTRO_CLASSE = "Filtrar perfil"
OPCAO_TODAS_CLASSES = "Todos os perfis"
SEM_RESULTADO_NO_FILTRO = "Nenhuma candidata desta busca pertence ao perfil selecionado."

# ----------------------------------------------------------------------
# Desfechos
# ----------------------------------------------------------------------

SEM_RESULTADO = (
    "Nenhuma startup da base atende a esta consulta, mesmo após o limite de "
    "duas tentativas de relaxamento. Nada foi sugerido no lugar."
)
SEM_RESULTADO_SAIDA = (
    "Reescreva a consulta com um setor, um estágio ou um sinal técnico "
    "diferente e busque de novo."
)
CANDIDATA_FORA_DA_BUSCA = (
    "A candidata selecionada não pertence à busca atual. Volte ao ranking e "
    "escolha uma das candidatas desta consulta."
)
MENSAGEM_FALHA_DESCOBERTA = (
    "A consulta não pôde ser concluída. Nenhuma candidata foi inventada; "
    "o rastreamento técnico ficou no terminal local."
)
MENSAGEM_FALHA_APROFUNDAMENTO = (
    "A análise completa não pôde ser concluída. Nenhum briefing foi gerado e "
    "nada foi preenchido no lugar dele."
)
DIAGNOSTICO_TECNICO = (
    "O rastreamento completo da falha foi escrito no terminal onde o Streamlit "
    "está rodando. Nenhuma credencial e nenhuma resposta bruta de provedor é "
    "exibida nesta tela."
)
TITULO_DIAGNOSTICO = "O que você pode fazer"
ROTULO_TENTAR_NOVAMENTE = "Tentar novamente"
ORIENTACAO_FALHA = (
    "A análise pode ser tentada novamente. Se a indisponibilidade continuar, "
    "confira a conexão e as configurações locais."
)

# ----------------------------------------------------------------------
# Tela da análise
# ----------------------------------------------------------------------

ROTULO_VOLTAR = "Voltar para as candidatas"
LEGENDA_VOLTAR = "A volta reaproveita a busca e a análise já concluídas."
ROTULO_BAIXAR = "Baixar briefing em Markdown"
LEGENDA_BAIXAR = "O arquivo carrega os mesmos fatos e as mesmas fontes da tela."

ABA_VISAO_GERAL = "Visão geral"
ABA_EVIDENCIAS = "Evidências"
ABA_RECOMENDACOES = "Recomendações NVIDIA"

TITULO_TESE = "Tese"
TITULO_SINTESE = "Síntese executiva"
TITULO_PONTOS = "Pontos de conversa"
TITULO_AVISOS = "Limites desta análise"
TITULO_FONTES = "Fontes públicas citadas"
TITULO_EVIDENCIA_STARTUP = "Evidência pública da startup"
TITULO_EVIDENCIA_NVIDIA = "Base de conhecimento NVIDIA"
LEGENDA_EVIDENCIA_STARTUP = (
    "Trechos literais dos documentos públicos da empresa, com a fonte de cada "
    "afirmação."
)
LEGENDA_EVIDENCIA_NVIDIA = (
    "Trechos recuperados da base NVIDIA que sustentam a tecnologia citada."
)
LEGENDA_RECOMENDACAO = (
    "Cada recomendação nasce de uma evidência validada da startup somada a uma "
    "citação da base NVIDIA. Sem os dois lados, ela não é exibida."
)
TITULO_SEM_ANALISE_PROFUNDA = "Não foi possível concluir esta análise"
SEM_PONTUACAO_GRAVADA = "sem pontuação gravada"
ETIQUETA_BRIEFING = "Briefing validado"
MENSAGEM_CARREGANDO_ANALISE = "Carregando..."
ROTULO_ANALISAR = "Analisar {} em profundidade"
TITULO_MOTIVO = "Motivo registrado"
ROTULO_PRIORIDADE = "Prioridade"
ROTULO_COMPLEXIDADE = "Complexidade"
ROTULO_TECNOLOGIAS = "Tecnologias NVIDIA"
ROTULO_JUSTIFICATIVA_TECNICA = "Justificativa técnica"
ROTULO_JUSTIFICATIVA_NEGOCIO = "Justificativa de negócio"
TITULO_LASTRO = "Fontes desta recomendação"
ROTULO_SITE = "Site oficial"
TITULO_POR_QUE_APARECEU = "Por que apareceu nesta busca"
TITULO_EVIDENCIAS_CONFIRMADAS = "O que as fontes confirmaram"
TITULO_EVIDENCIAS_DESCARTADAS = "O que não passou pela conferência"
TITULO_INFORMACOES_ABERTAS = "O que ainda não foi confirmado"
TITULO_NECESSIDADES = "Necessidades confirmadas"
TITULO_SCORE = "Aderência à stack NVIDIA"
TITULO_PILARES = "Como a pontuação foi formada"
TITULO_RESULTADO_ATUAL = "Resultado da análise atual"
AVISO_CACHE_ATUAL = (
    "O ranking usa uma análise salva anteriormente. Esta página executou uma "
    "nova leitura das fontes, por isso o resultado atual pode ser diferente."
)
AVISO_CACHE_DIVERGENTE = (
    "A análise atual não confirmou a mesma classificação ou pontuação mostrada "
    "no ranking salvo. Considere o resultado atual para esta conversa."
)
EXPLICACAO_SEM_RECOMENDACAO = (
    "Nenhuma recomendação foi exibida porque faltou evidência suficiente para "
    "ligar uma necessidade da startup a uma tecnologia NVIDIA."
)
EXPLICACAO_NON_AI = (
    "non-AI é uma classe técnica desta análise, não uma avaliação da qualidade "
    "da empresa. Com essa classificação validada, o fit-score é zero e não há "
    "recomendação NVIDIA."
)
