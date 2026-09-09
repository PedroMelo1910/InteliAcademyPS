# Notas técnicas — NVIDIA AI Radar

Este documento complementa o [README principal](../README.md). O README apresenta o produto, o fluxo, a instalação e o uso; aqui ficam as invariantes que ajudam a compreender e defender as decisões de arquitetura.

## 1. Fronteiras do sistema

### Estado e contratos

`radar/contratos.py` define o estado do LangGraph e os objetos trocados entre as etapas. Os modelos Pydantic proíbem campos extras e mantêm identificadores de startup, documento, afirmação e chunk até a saída final.

Texto integral de documentos não circula desnecessariamente no estado. Agentes recebem somente a projeção permitida pela sua responsabilidade.

### Dados

`radar/base_startups.py` é a fronteira do SQLite. Ela valida os JSONs curados, cria as tabelas, executa SQL parametrizado, mantém os índices FTS5 e persiste o cache de análises.

`dados/radar.db` guarda dados de negócio e índices. `dados/checkpoints.db` guarda checkpoints do LangGraph. Ambos são artefatos locais regeneráveis e ficam fora do Git.

### Provedores

`radar/provedores.py` adapta os serviços externos aos contratos do domínio. Gemini é o LLM primário; Groq pode atuar como reserva operacional. Embedding e reranking NVIDIA permanecem em fronteiras próprias e seus clientes são inicializados somente no primeiro uso do RAG, para que uma indisponibilidade técnica não bloqueie a descoberta textual. Testes substituem todos eles por implementações controladas.

## 2. Invariantes de evidência

- `classe_referencia` é gabarito de avaliação offline e nunca é entregue aos agentes.
- O Extractor lê somente documentos recuperados da startup selecionada.
- Cada afirmação mantém trecho literal e identificador de documento.
- O Evidence Validator é determinístico e não chama LLM nem rede.
- Confirmação de proveniência significa que o trecho existe na fonte armazenada; não certifica verdade objetiva externa.
- Informação desconhecida não é convertida em ausência.
- Conflitos permanecem desconhecidos e preservam os identificadores envolvidos.
- Evidência derrubada não classifica, não pontua e não recomenda.
- Falha operacional não é apresentada como evidência insuficiente.

## 3. Invariantes do ranking

O ranking combina informações sem fundir escalas diferentes:

- BM25 indica relação lexical entre a pergunta e os documentos recuperados;
- fit-score indica aderência NVIDIA sustentada pela análise persistida.

Análises concluídas são ordenadas antes das insuficientes ou ausentes. Em seguida vêm fit-score decrescente, BM25, nome e identificador. Uma análise concluída sem `FitScore` é tratada como cache inválido; o sistema não inventa zero.

O grafo de lote reutiliza Extractor, Classifier, Evidence Validator, R2 e R3. Ele não chama Query Planner, NVIDIA RAG, Recommendation ou Briefing.

## 4. Invariantes do RAG NVIDIA

- O corpus de startups e o corpus NVIDIA são separados.
- SQLite é a fonte de verdade dos metadados e textos indexados.
- FTS5 faz recuperação lexical; sqlite-vec faz recuperação vetorial.
- RRF combina as listas; ele não substitui o reranking.
- O reranker atribui scores aos candidatos já recuperados; a ordenação e a guarda de cobertura determinísticas selecionam os seis trechos finais.
- Cada chunk mantém URL, título, origem e tecnologia ou tópico.
- Uma recomendação precisa de evidência pública da startup e citação NVIDIA resolvida.
- Material conceitual pode contextualizar, mas não sustenta sozinho uma tecnologia.

## 5. Invariantes da recomendação

O LLM produz apenas um rascunho limitado. O programa valida o fundamento, resolve evidências e citações e calcula prioridade, complexidade e fit-score.

O fit-score usa quatro pilares e uma soma bruta máxima de 36. Somente afirmações confirmadas pontuam. `gap_confirmado` é diferente de `desconhecido`, e o gate `non-AI` produz zero explicitamente.

Uma recomendação pode partir de:

- uma lacuna confirmada, como dependência de API externa ou dor de escala;
- uma oportunidade técnica confirmada, como inferência de LLM, visão computacional, voz, dados em escala ou robótica.

Nos dois casos, a tecnologia precisa pertencer ao catálogo permitido para aquele fundamento e aparecer no contexto técnico recuperado.

## 6. Saídas possíveis

O Briefing é o único payload final do aprofundamento:

1. **normal:** classificação e evidências válidas, contexto NVIDIA e resultado permitido pelas regras de recomendação;
2. **non-AI:** classe validada, fit-score zero e nenhuma recomendação NVIDIA;
3. **evidência insuficiente:** explica o que faltou sem produzir classe, score ou recomendação fictícia.

A versão Markdown contém os mesmos fatos e fontes exibidos na interface.

## 7. Mapa de leitura do código

Para acompanhar uma busca completa:

1. comece em `radar/aplicacao.py`;
2. veja a montagem em `radar/grafo.py`;
3. acompanhe os nós em `radar/agentes/`;
4. confira as decisões em `radar/agentes/roteadores.py`;
5. veja os schemas em `radar/contratos.py`;
6. acompanhe a recuperação NVIDIA em `radar/conhecimento_nvidia/`;
7. veja score e regras em `radar/recomendacao.py` e `radar/regras_recomendacao.py`;
8. termine em `app.py` e `radar/interface/` para entender a apresentação.

Os comandos de instalação, preparação da base, testes e execução estão centralizados no [README](../README.md#instalação-do-zero).
