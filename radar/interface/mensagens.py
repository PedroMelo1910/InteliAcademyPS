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

NOME_PRODUTO = "NVIDIA Startup AI Radar"
ETIQUETA_PRODUTO = "Triagem assistida · Startups & VCs"
PROPOSITO = (
    "O radar encontra startups brasileiras na base curada e identifica, com "
    "evidência pública rastreável, onde a stack NVIDIA tem oportunidade "
    "sustentada."
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
CONVITE_INICIAL = (
    "Descreva em linguagem natural o tipo de empresa que você quer triar. "
    "O radar planeja a consulta, recupera candidatas da base curada e mostra o "
    "que já foi analisado."
)
TITULO_EXEMPLOS = "Comece por uma destas perguntas"
LEGENDA_EXEMPLOS = (
    "Clicar preenche o campo acima; a busca só roda quando você confirmar."
)
PERGUNTAS_DE_EXEMPLO = (
    "fintechs brasileiras que treinam modelos próprios em produção",
    "startups de saúde usando visão computacional em imagem médica",
    "empresas de logística com otimização de rotas em tempo real",
)

MENSAGEM_CARREGANDO_BUSCA = "Planejando a consulta e recuperando candidatas..."

# ----------------------------------------------------------------------
# As duas medidas, que nunca se misturam
# ----------------------------------------------------------------------

ROTULO_FIT_SCORE = "Fit-score NVIDIA"
ROTULO_BM25 = "Relevância lexical (BM25)"
EXPLICACAO_BM25 = (
    "O BM25 mede a relevância textual da busca lexical no SQLite: valores "
    "menores indicam maior aderência ao texto da consulta. É informação de "
    "recuperação e não entra no fit-score NVIDIA."
)
EXPLICACAO_FIT_SCORE = (
    "O fit-score mede aderência à stack NVIDIA sustentada por evidência "
    "pública — não é nota de qualidade da empresa. Ele só aparece quando a "
    "rubrica encontrou lastro, e um zero validado continua sendo um "
    "resultado da rubrica."
)
TITULO_COMO_LER = "Como ler as duas medidas desta tela"

# ----------------------------------------------------------------------
# Painel do ranking
# ----------------------------------------------------------------------

ROTULO_TOTAL_CANDIDATAS = "Candidatas"
ROTULO_TOTAL_ANALISADAS = "Com análise gravada"
ROTULO_TOTAL_SEM_LASTRO = "Sem lastro suficiente"
ROTULO_TOTAL_PENDENTES = "Ainda sem análise"
TITULO_RANKING = "Candidatas priorizadas"
LEGENDA_RANKING = (
    "Primeiro as empresas já analisadas, do maior para o menor fit-score "
    "NVIDIA. A relevância lexical entra só como desempate entre empresas de "
    "mesma pontuação. As empresas sem análise concluída aparecem depois. A "
    "ordem é a que a aplicação calculou: a tela não reordena, não recalcula "
    "pontuação e não completa lacuna."
)
TITULO_INTERPRETACAO = "Como esta consulta foi interpretada"

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
TITULO_DIAGNOSTICO = "Diagnóstico técnico"

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
ABA_RASTRO = "Rastro técnico"

TITULO_TESE = "Tese"
TITULO_SINTESE = "Síntese executiva"
TITULO_PONTOS = "Pontos de conversa"
TITULO_AVISOS = "Avisos de honestidade operacional"
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
TITULO_AUDITORIA = "Auditoria da execução"
TITULO_SEM_ANALISE_PROFUNDA = "Nada foi preenchido no lugar do briefing"
SEM_PONTUACAO_GRAVADA = "sem pontuação gravada"
ETIQUETA_BRIEFING = "Briefing validado"
MENSAGEM_CARREGANDO_ANALISE = (
    "Extração, classificação, validação de evidências, RAG NVIDIA e "
    "briefing..."
)
TITULO_DOCUMENTOS = "Documentos recuperados desta candidata"
ROTULO_ANALISAR = "Analisar {} em profundidade"
TITULO_JUSTIFICATIVA = "Por que essa pontuação"
TITULO_MOTIVO = "Motivo registrado"
ROTULO_PRIORIDADE = "Prioridade"
ROTULO_COMPLEXIDADE = "Complexidade"
ROTULO_TECNOLOGIAS = "Tecnologias NVIDIA"
ROTULO_JUSTIFICATIVA_TECNICA = "Justificativa técnica"
ROTULO_JUSTIFICATIVA_NEGOCIO = "Justificativa de negócio"
TITULO_LASTRO = "Lastro desta recomendação"
ROTULO_SITE = "Site oficial"
