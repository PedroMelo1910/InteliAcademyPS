# NVIDIA Startup AI Radar — documentação de trabalho

Este arquivo reúne a documentação do projeto em um só lugar. Ele é um material de construção e compreensão; o README de entrega será escrito apenas quando o sistema estiver mais completo.

## 1. Arquitetura e fluxo atual

O sistema possui uma primeira fatia vertical executável. O `app.py` atual é somente
uma bancada provisória para acionar e observar o código; a interface final será
construída depois que os contratos e as saídas do núcleo estiverem estáveis.

```text
Consulta de entrada (temporariamente pelo Streamlit)
        │
        ▼
Query Planner — Gemini transforma a frase em PlanoConsulta
        │
        ▼
Retriever — filtros estruturados no SQLite
        │
        ▼
Retriever — FTS5/BM25 ordena os documentos por relevância
        │
        ▼
R1
  ├─ analisar ───────────────► aprofundamento futuro
  ├─ candidatas_prontas ─────► ranking interno da aplicação
  ├─ relaxar ────────────────► volta ao Query Planner
  └─ sem_resultado ──────────► encerra honestamente
```

O ranking atual mede relevância lexical para a pergunta. Ele ainda não é o fit-score NVIDIA.

Quando não há candidatas, existem no máximo duas tentativas de relaxamento. A primeira remove estágio, localização e porte; a segunda remove setor. Termos de busca, sinais de IA e uma possível classe já produzida pelo sistema são preservados.

## 2. Componentes e contratos

### Query Planner

Recebe a pergunta do usuário e os vocabulários existentes no banco. O Gemini devolve um `PlanoConsulta` por structured output. O Pydantic rejeita campos extras, listas inválidas e filtros fora do vocabulário. Uma resposta inválida recebe apenas uma nova tentativa; se falhar novamente, a execução termina sem criar um plano falso.

### Retriever

Recebe um `PlanoConsulta` validado e funciona em duas camadas. Primeiro, aplica filtros estruturados por SQL parametrizado. Depois, executa `FTS5 MATCH` e ordena os documentos com `bm25()` crescente. Entradas com hífen e outros caracteres da sintaxe FTS5 são transformadas em frases escapadas.

### R1 e estado

R1 é uma função pura que lê o estado e devolve exatamente `analisar`, `candidatas_prontas`, `relaxar` ou `sem_resultado`. O state é tipado e guarda somente o necessário para reconstruir o caminho. `trajeto` e `criterios_relaxados` usam reducers para acumular os passos.

### StateGraph e persistência

O grafo real já contém a volta condicional do Retriever para o Query Planner. O `SqliteSaver` persiste checkpoints em `dados/checkpoints.db`, separado do banco de negócio `dados/radar.db`. Os dois arquivos são artefatos locais ignorados pelo Git.

## 3. Base de startups e documentos

A base começa com três empresas reais e três documentos por empresa:

- Maritaca AI — domínios `maritaca.ai`, `arxiv.org` e `sbtnews.sbt.com.br`;
- Alice — domínios `alice.com.br`, `latinamericafund.com` e `globalprivatecapital.org`;
- Caju — domínios `caju.com.br`, `bloomberglinea.com.br` e `onevc.vc`.

Cada arquivo em `dados/base/` representa uma startup. Dentro dele ficam os campos estruturados e seus documentos. Cada documento é uma síntese factual escrita para este projeto, não uma cópia da página nem resultado de scraping.

Formato obrigatório de cada documento:

- `tipo` dentro dos seis valores aceitos pelo TAPI;
- `titulo`;
- `conteudo_texto` resumido manualmente;
- `url_fonte` real e pública;
- `dominio_fonte`, que precisa corresponder à URL;
- `data_publicacao`, quando conhecida;
- `data_acesso` obrigatória.

O seed valida no mínimo três documentos e três domínios distintos por startup, URLs únicas e os vocabulários fechados. Depois atualiza as tabelas e reconstrói o FTS5. Rodar novamente com os mesmos JSONs não duplica os registros.

`classe_referencia` existe somente para uma avaliação futura. O Retriever e o ranking nunca leem esse campo, pois isso entregaria ao sistema uma resposta que ele deveria produzir sozinho mais adiante.

## 4. Execução, testes e limites atuais

Inicializar ou atualizar o banco:

```powershell
.venv\Scripts\python.exe -m scripts.inicializar_base
```

Executar os testes offline:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Iniciar a bancada provisória de validação manual:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

A chave deve existir somente no `.env` local como `GOOGLE_API_KEY`, sem necessidade de aspas. Se a chave estiver ausente, o núcleo gera um erro de configuração claro e seguro. Se o Gemini falhar, o sistema interrompe a consulta e não fabrica candidatas.

Os testes atuais cobrem inicialização repetível, filtros estruturados, SQL parametrizado, hífen no FTS5, ordem do BM25, Retriever offline, quatro saídas de R1, dois degraus de relaxamento, structured output inválido, chave ausente, falha do Gemini, checkpoints e integração do resultado real com o ranking.

Limites deliberados deste momento:

- apenas três startups, suficientes para provar a fatia;
- recuperação lexical, ainda sem busca por significado ou sinônimos;
- ausência de Extractor, Classifier, Evidence Validator, RAG NVIDIA, recomendação, fit-score e briefing;
- rota `analisar` alcançável, mas sem produzir uma análise fictícia enquanto os agentes seguintes não existirem.
- interface atual propositalmente provisória; layout e componentes finais serão feitos somente depois do núcleo funcional completo.
