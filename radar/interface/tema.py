"""Folha de estilo estática do radar: um literal, sem nenhuma interpolação.

A direção visual tem uma ideia só, e ela é funcional: **dois eixos de cor que
nunca se encontram**. Verde NVIDIA pertence ao eixo da decisão — fit-score,
ação primária, análise concluída. Azul pertence ao eixo da recuperação —
relevância lexical BM25. Quem olha a tela numa gravação de sete minutos
consegue separar "o quanto isso aderiu à stack NVIDIA" de "o quanto isso casou
com o texto da busca" sem ler uma legenda.

Âmbar e cinza cobrem os desfechos honestos (sem lastro, sem análise) e vermelho
fica reservado à falha operacional. O fundo é carvão, não preto: contraste alto
sem o brilho duro que cansa numa apresentação projetada.

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
  --radar-verde-claro:#9BDD22;
  --radar-verde-vidro:rgba(118,185,0,.10);
  --radar-azul:#5B9DF9;
  --radar-ambar:#E0A32E;
  --radar-cinza:#7C889B;
  --radar-fundo:#0D1015;
  --radar-superficie:#151A21;
  --radar-superficie-alta:#1B222C;
  --radar-borda:#28313E;
  --radar-texto:#E9EDF3;
  --radar-texto-suave:#9AA5B7;
  --radar-passo:1.15rem;
}

/* Coluna de leitura: larga o bastante para o cartão, estreita o bastante
   para a linha não virar uma faixa ilegível num monitor de 1440. */
[data-testid="stMainBlockContainer"]{
  max-width:1160px;
  padding-top:1.9rem;
  padding-bottom:4.5rem;
}
[data-testid="stHeader"]{background:transparent;}
/* O botão de deploy não pertence a uma demonstração de sete minutos. */
[data-testid="stAppDeployButton"]{display:none;}

h1,h2,h3,h4,h5{letter-spacing:-.015em;}
h1{line-height:1.08;letter-spacing:-.03em;}
hr{border-color:var(--radar-borda);}

/* --------------------------------------------------------------
   Faixa de abertura
   -------------------------------------------------------------- */
.st-key-topo_radar{
  border-bottom:1px solid var(--radar-borda);
  padding-bottom:var(--radar-passo);
  margin-bottom:1.5rem;
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
  font-size:1.02rem;
  line-height:1.55;
  color:var(--radar-texto-suave);
  max-width:68ch;
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
  font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
  font-size:1.65rem;
  font-weight:700;
  line-height:1;
  color:var(--radar-verde);
  font-variant-numeric:tabular-nums;
  margin:0;
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
  background:linear-gradient(180deg,var(--radar-verde-vidro),transparent 60%);
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
[data-testid="stProgress"] div[role="progressbar"] > div{
  background-image:linear-gradient(90deg,var(--radar-verde),var(--radar-verde-claro));
}

@media (max-width:1200px){
  [data-testid="stMainBlockContainer"]{padding-left:1.4rem;padding-right:1.4rem;}
}
@media (prefers-reduced-motion:reduce){
  [class*="st-key-cartao_"] > div{transition:none;}
}
</style>
"""
