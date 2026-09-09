"""Folha de estilo estática do radar: um literal, sem nenhuma interpolação.

A direção visual tem uma ideia só, e ela é funcional: o verde NVIDIA pertence
ao eixo da decisão — fit-score, ação primária e análise concluída. A relevância
textual continua na ordenação, mas o seu índice interno não ocupa os cartões.

Âmbar e cinza cobrem os desfechos honestos (sem lastro, sem análise), e vermelho
fica reservado à falha operacional. O fundo é claro e levemente esverdeado,
com cartões brancos e texto escuro para leitura confortável na apresentação.

Este módulo não importa Streamlit e não recebe dado nenhum. ``CSS_TEMA`` é um
literal justamente para que a fronteira de HTML bruto da aplicação seja
auditável por leitura: nenhuma consulta de usuário, nenhum texto de fonte
pública e nenhuma saída de modelo consegue chegar até aqui.
"""

from __future__ import annotations


CSS_TEMA = """
<style>
:root{
  --radar-verde:#76B900;
  --radar-verde-claro:#96D329;
  --radar-verde-escuro:#315B08;
  --radar-verde-vidro:rgba(118,185,0,.09);
  --radar-azul:#356F9C;
  --radar-ambar:#B27600;
  --radar-cinza:#75806F;
  --radar-fundo:#F6F8F4;
  --radar-superficie:#FFFFFF;
  --radar-superficie-alta:#FBFDF9;
  --radar-borda:#DDE5D8;
  --radar-texto:#182017;
  --radar-texto-suave:#5E695A;
  --radar-sombra:0 10px 28px rgba(37,56,28,.07);
  --radar-passo:1.15rem;
}

/* Coluna de leitura: larga o bastante para o cartão, estreita o bastante
   para a linha não virar uma faixa ilegível num monitor de 1440. */
[data-testid="stMainBlockContainer"]{
  max-width:1160px;
  padding-top:1.9rem;
  padding-bottom:4.5rem;
}
[data-testid="stAppViewContainer"]{background:var(--radar-fundo);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{
  background:#FFFFFF;
  border-right:1px solid var(--radar-borda);
}
[data-testid="stSidebar"] h3{color:var(--radar-verde);}
[data-testid="stSidebar"] [role="radiogroup"] label{
  border-radius:10px;
  padding:.35rem .55rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:var(--radar-verde-vidro);
}
[data-testid="stSidebar"] button[kind="headerNoPadding"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]{
  font-size:0;
}
[data-testid="stSidebar"] button[kind="headerNoPadding"] [data-testid="stIconMaterial"]::after,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after{
  content:"☰";
  color:var(--radar-verde-escuro);
  font-family:Arial,sans-serif;
  font-size:1.25rem;
}
/* O botão de deploy não pertence a uma demonstração de sete minutos. */
[data-testid="stAppDeployButton"]{display:none;}

h1,h2,h3,h4,h5{letter-spacing:-.015em;}
h1{line-height:1.08;letter-spacing:-.03em;}
hr{border-color:var(--radar-borda);}
a{color:var(--radar-verde-escuro);text-underline-offset:2px;}
:focus-visible{outline:3px solid rgba(118,185,0,.38)!important;outline-offset:2px;}

/* --------------------------------------------------------------
   Faixa de abertura
   -------------------------------------------------------------- */
.st-key-topo_radar{
  border-bottom:1px solid var(--radar-borda);
  padding-bottom:var(--radar-passo);
  margin-bottom:1.5rem;
}
.st-key-topo_radar h1{
  color:var(--radar-verde);
  font-size:clamp(2.2rem,4vw,3.45rem);
  letter-spacing:-.035em;
  margin:.35rem 0 1rem;
}
.st-key-etiqueta_produto p,
.st-key-etiqueta_briefing p{
  font-size:.72rem;
  font-weight:700;
  letter-spacing:.19em;
  text-transform:uppercase;
  color:var(--radar-verde);
  margin-bottom:.15rem;
}
.st-key-proposito_produto p{
  font-size:1.08rem;
  line-height:1.55;
  color:var(--radar-texto-suave);
  max-width:68ch;
}
.st-key-como_funciona{
  margin:.8rem 0 1.5rem;
}
[class*="st-key-passo_"]{
  min-height:128px;
  background:var(--radar-superficie);
  border:1px solid var(--radar-borda);
  border-radius:16px;
  padding:1rem 1.05rem;
  box-shadow:0 6px 18px rgba(37,56,28,.045);
}
[class*="st-key-passo_"] p{color:var(--radar-texto-suave);line-height:1.45;}
[class*="st-key-numero_passo_"] p{
  color:var(--radar-verde-escuro);font-weight:800;font-size:.78rem;
  letter-spacing:.12em;text-transform:uppercase;
}

/* --------------------------------------------------------------
   Busca: o campo é a ação principal da tela
   -------------------------------------------------------------- */
.st-key-bloco_busca [data-testid="stTextInput"] input{
  font-size:1.02rem;
  padding-top:.72rem;
  padding-bottom:.72rem;
}
.st-key-bloco_busca [data-testid="stForm"]{
  border:1px solid var(--radar-borda);
  background:var(--radar-superficie);
  border-radius:14px;
  padding:1.15rem 1.15rem .95rem;
  box-shadow:var(--radar-sombra);
}

.st-key-exemplos_consulta{
  gap:.55rem;
  flex-wrap:wrap;
}
.st-key-exemplos_consulta button{
  border:1px solid var(--radar-borda);
  border-radius:999px;
  padding:.34rem .9rem;
  font-weight:500;
  color:var(--radar-texto-suave);
}
.st-key-exemplos_consulta button:hover{
  border-color:var(--radar-verde);
  background:var(--radar-verde-vidro);
  color:var(--radar-texto);
}

/* --------------------------------------------------------------
   Painel de contagens
   -------------------------------------------------------------- */
.st-key-painel_ranking [data-testid="stMetricValue"]{
  font-variant-numeric:tabular-nums;
}
.st-key-painel_ranking [data-testid="stMetricLabel"] p{
  font-size:.74rem;
  letter-spacing:.11em;
  text-transform:uppercase;
  color:var(--radar-texto-suave);
}

/* --------------------------------------------------------------
   Cartões: uma camada só, com uma marca de acento à esquerda.
   A chave do container cai no próprio bloco vertical, então o cartão
   é ele — estilizar os filhos transformaria cada parágrafo numa caixa.
   -------------------------------------------------------------- */
[class*="st-key-cartao_"],
[class*="st-key-bloco_recomendacao_"]{
  position:relative;
  background:var(--radar-superficie);
  border-color:var(--radar-borda);
  border-radius:14px;
  overflow:hidden;
  padding-left:1.3rem;
  transition:border-color .18s ease, background .18s ease;
  box-shadow:var(--radar-sombra);
  margin-bottom:.85rem;
}
[class*="st-key-cartao_"]::before,
[class*="st-key-bloco_recomendacao_"]::before{
  content:"";
  position:absolute;
  left:0;
  top:0;
  bottom:0;
  width:3px;
  background:linear-gradient(180deg,var(--radar-verde),transparent 85%);
  opacity:.6;
}
[class*="st-key-cartao_"]:hover{
  border-color:var(--radar-verde);
  background:var(--radar-superficie-alta);
}
[class*="st-key-posicao_"] p{
  font-family:inherit;
  font-size:1.2rem;
  font-weight:600;
  line-height:1;
  color:var(--radar-verde);
  font-variant-numeric:tabular-nums;
  margin:0;
}

.st-key-metricas_dashboard{
  background:var(--radar-superficie);
  border:1px solid var(--radar-borda);
  border-radius:14px;
  padding:1rem 1.1rem .75rem;
  box-shadow:var(--radar-sombra);
  margin:1rem 0 1.6rem;
}
[class*="st-key-nome_"] p{
  font-size:1.26rem;
  font-weight:700;
  line-height:1.25;
  margin-bottom:.15rem;
}
[class*="st-key-meta_"] p{
  font-size:.86rem;
  color:var(--radar-texto-suave);
}

/* Eixo azul: tudo que fala de recuperação textual, nunca de aderência. */
[class*="st-key-lexical_"] p,
.st-key-nota_lexical p{
  color:var(--radar-azul);
  font-variant-numeric:tabular-nums;
  font-size:.82rem;
}

/* --------------------------------------------------------------
   Cabeçalho da análise
   -------------------------------------------------------------- */
.st-key-cabecalho_analise{
  border:1px solid var(--radar-borda);
  border-left:3px solid var(--radar-verde);
  border-radius:14px;
  background:linear-gradient(135deg,#FFFFFF 0%,#F4FAEA 100%);
  box-shadow:var(--radar-sombra);
  padding:1.3rem 1.35rem 1.1rem;
  margin-bottom:1.4rem;
}
.st-key-cabecalho_analise h2{margin-top:0;}
.st-key-tese_principal p{
  font-size:1.06rem;
  line-height:1.6;
  border-left:2px solid var(--radar-borda);
  padding-left:.95rem;
  margin-top:.35rem;
}

/* --------------------------------------------------------------
   Abas
   -------------------------------------------------------------- */
[data-testid="stTabs"] [role="tab"]{
  font-size:.93rem;
  letter-spacing:.005em;
  padding-left:.15rem;
  padding-right:.15rem;
}
[data-testid="stTabs"] [role="tablist"]{
  border-bottom:1px solid var(--radar-borda);
  gap:1.6rem;
}

/* --------------------------------------------------------------
   Estados e detalhes
   -------------------------------------------------------------- */
[data-testid="stExpander"] details{
  border-color:var(--radar-borda);
  background:var(--radar-superficie);
  border-radius:12px;
}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums;}
[data-testid="stProgressBarTrack"]{
  background:#E7EDE2;
}
[data-testid="stProgressBarTrack"] > div{
  background-image:linear-gradient(90deg,var(--radar-verde),var(--radar-verde-claro));
}

[class*="st-key-pilar_"],
[class*="st-key-evidencia_"],
[class*="st-key-necessidade_"]{
  background:var(--radar-superficie);
  border:1px solid var(--radar-borda);
  border-radius:12px;
  padding:.75rem .9rem;
}

[data-testid="stAlert"]{border-radius:12px;}
[data-testid="stDownloadButton"] button{width:100%;}

@media (max-width:1200px){
  [data-testid="stMainBlockContainer"]{padding-left:1.4rem;padding-right:1.4rem;}
}
@media (max-width:700px){
  [data-testid="stMainBlockContainer"]{padding:1.05rem .9rem 3rem;}
  h1{font-size:2rem!important;}
  [class*="st-key-cartao_"],[class*="st-key-bloco_recomendacao_"]{padding-left:.9rem;}
  .st-key-cabecalho_analise{padding:1rem;}
  [data-testid="stTabs"] [role="tablist"]{gap:.65rem;overflow-x:auto;}
  [data-testid="stTabs"] [role="tab"]{white-space:nowrap;}
}
@media (prefers-reduced-motion:reduce){
  [class*="st-key-cartao_"] > div{transition:none;}
}
</style>
"""
