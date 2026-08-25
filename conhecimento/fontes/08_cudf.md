---
topico: cudf
origem: tecnologia
tecnologia: cuDF
fonte_url: https://docs.rapids.ai/api/cudf/stable/
titulo: cuDF
data_acesso: 2026-08-25
---

# Visão geral

cuDF é a biblioteca de dataframes do RAPIDS: carrega, filtra, junta e agrega dados na GPU com API compatível com pandas. Desde as versões recentes, o modo acelerador (`cudf.pandas`) executa código pandas existente sem alteração, caindo para CPU quando uma operação não tem suporte.

# Problemas que resolve

O pandas em CPU é o gargalo silencioso de muitos produtos de dados: joins e groupbys de dezenas de gigabytes que duram horas. O cuDF executa essas mesmas operações em minutos, preservando o código e o conhecimento do time.

# Quando recomendar a uma startup

Recomendação pontual e de baixa fricção quando a evidência mostra pipelines pandas lentos em produção ou notebooks analíticos que limitam a operação. Frequentemente entra como primeiro passo concreto do pacote RAPIDS, por ser demonstrável em um dia.

# Adoção e integração

Requer GPU disponível; o modo compatível com pandas torna o piloto quase imediato. Complexidade média pela dependência de infraestrutura, não pela API.
