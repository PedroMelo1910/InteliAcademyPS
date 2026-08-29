---
topico: nemo_guardrails
origem: tecnologia
tecnologia: NeMo Guardrails
fonte_url: https://github.com/NVIDIA/NeMo-Guardrails
titulo: NeMo Guardrails
data_acesso: 2026-08-25
---

# Visão geral

NeMo Guardrails é um toolkit de código aberto para adicionar trilhos programáveis a assistentes e agentes baseados em LLM: regras que controlam o que o sistema pode falar, quais tópicos deve recusar, como deve se comportar em diálogo e quais ferramentas pode acionar. Os trilhos são declarados em uma linguagem própria (Colang) e aplicados em tempo de execução, independentemente do modelo usado.

# Problemas que resolve

Chatbots e agentes em produção sem controle de comportamento geram risco real: alucinação apresentada como fato, vazamento de instruções internas, respostas fora de escopo em domínios regulados e injeção de prompt. O Guardrails intercepta entrada e saída do modelo e aplica as regras do negócio antes que o usuário veja a resposta.

# Quando recomendar a uma startup

Prioritário para qualquer empresa com assistente conversacional voltado ao cliente, especialmente em setores regulados (saúde, financeiro, jurídico) ou com agentes que executam ações. É frequentemente a recomendação que acompanha NIM: servir o modelo é metade do problema; governar o comportamento é a outra metade.

# Adoção e integração

Biblioteca Python de código aberto que envolve o LLM existente — funciona com APIs externas e com modelos auto-hospedados. Não exige GPU própria; complexidade média, dominada pela definição das regras de negócio, não pela infraestrutura.
