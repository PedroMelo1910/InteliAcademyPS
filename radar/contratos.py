from __future__ import annotations

import math
import operator
import re
from datetime import date
from typing import Annotated, Any, Literal, TypedDict, get_args
from urllib.parse import urlparse

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


ClasseStartup = Literal["AI-native", "AI-enabled", "non-AI"]
TipoDocumento = Literal[
    "site institucional",
    "blog",
    "notícia",
    "vaga",
    "perfil de founder",
    "release",
]
ResultadoR1 = Literal["analisar", "candidatas_prontas", "relaxar", "sem_resultado"]
ResultadoR2 = Literal["reextrair", "evidencia_pronta"]
ResultadoR3 = Literal["evidencia_insuficiente", "nao_aderente", "prosseguir"]

CategoriaAfirmacao = Literal[
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
    "stack_propria",
    "dependencia_api_externa",
    "escala_e_dor_operacional",
    "momento_e_financiamento",
    "equipe_e_contratacao",
    "outro",
]

CATEGORIAS_AFIRMACAO: tuple[str, ...] = get_args(CategoriaAfirmacao)

# Só as quatro dimensões estruturais distinguem capacidade observada de gap
# declarado; nas demais categorias a polaridade não carrega informação.
DimensaoGap = Literal[
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
]

# A ordem desta tupla é o contrato: todo relatório de gap percorre as quatro
# dimensões sempre na mesma sequência, para que duas execuções iguais produzam
# saídas comparáveis campo a campo.
DIMENSOES_GAP: tuple[str, ...] = get_args(DimensaoGap)

CATEGORIAS_ESTRUTURAIS: frozenset[str] = frozenset(DIMENSOES_GAP)

PolaridadeAfirmacao = Literal["presenca", "ausencia_explicita", "neutro"]

SituacaoAfirmacao = Literal["confirmada", "derrubada"]
EstadoGap = Literal["capacidade_confirmada", "gap_confirmado", "desconhecido"]
ConfiancaPerfil = Literal["normal", "baixa"]

GapEnderecado = Literal[
    "dados_proprietarios",
    "workflow_profundo",
    "distribuicao",
    "otimizacao_tecnica",
    "dependencia_api_externa",
    "escala_e_dor_operacional",
]

TipoAcao = Literal[
    "convite_inception",
    "call_tecnica_descoberta",
    "benchmark_custo_latencia",
    "poc_nim",
    "workshop_guardrails",
    "intro_comunidade_evento",
]

PrioridadeRecomendacao = Literal["alta", "media", "baixa"]
ComplexidadeRecomendacao = Literal["baixa", "media", "alta"]

PilarFit = Literal[
    "centralidade_ia",
    "gap_enderecavel",
    "momento",
    "alinhamento_setorial",
]
PILARES_FIT: tuple[str, ...] = get_args(PilarFit)

FaixaFit = Literal["baixa", "media", "alta"]
TravaFit = Literal["gate_evidencia", "teto_corrobacao", "gate_non_ai"]
TRAVAS_FIT: tuple[str, ...] = get_args(TravaFit)

MAXIMO_CARACTERES_MOTIVO = 200

MINIMO_FRASES_JUSTIFICATIVA = 2
MAXIMO_FRASES_JUSTIFICATIVA = 4

LIMITE_TRECHO_CITADO = 300
MINIMO_CARACTERES_TRECHO_CITADO = 12
MINIMO_PALAVRAS_TRECHO_CITADO = 3


def normalizar_dominio(valor: str) -> str:
    """Ponto fixo por construção: normalizar duas vezes dá o mesmo resultado.

    Um prefixo ``www.`` repetido não pode sobreviver a uma passagem, senão o
    contrato que exige host normalizado rejeita o próprio valor normalizado.
    """
    dominio = valor.strip().lower()
    while dominio.startswith("www."):
        dominio = dominio[4:]
    return dominio


def normalizar_texto_citavel(valor: str) -> str:
    """Normalização de proveniência: whitespace colapsado e casefold, nada mais.

    Qualquer outra diferença entre citação e fonte é diferença real, e precisa
    derrubar a afirmação em vez de ser absorvida em silêncio.
    """
    return " ".join(valor.split()).casefold()


def contar_frases(texto: str) -> int:
    """Conta frases sem confundir abreviações, decimais e listas com término.

    Não pretende fazer análise linguística completa; é uma fronteira tolerante
    para o formato curto produzido pelo Extractor.
    """
    limpo = texto.strip()
    if not limpo:
        return 0
    marcador = "∯"
    limpo = re.sub(r"(?<=\d)\.(?=\d)", marcador, limpo)
    limpo = re.sub(r"(?<=\d)\.(?=\s+[a-zà-ÿ])", marcador, limpo)
    padroes_nao_terminais = (
        r"\b(?:Dr|Dra|Eng|Ex|Prof|Profa|Sr|Sra|Srta)\.(?=\s+\S)",
        r"\b(?:[A-ZÀ-Ú]\.){2,}(?=\s+[a-zà-ÿ])",
        r"\betc\.(?=\s+[a-zà-ÿ])",
        r"\.{3,}(?=\s+[a-zà-ÿ])",
    )
    for padrao in padroes_nao_terminais:
        limpo = re.sub(
            padrao,
            lambda ocorrencia: ocorrencia.group(0).replace(".", marcador),
            limpo,
            flags=re.IGNORECASE,
        )
    return len(re.findall(r"[.!?]+(?:[\"'»”\)\]}]+)?(?=\s|$)", limpo))


class FiltrosEstruturados(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setor: str | None = None
    estagio: list[str] | None = None
    localizacao: str | None = None
    tamanho_time: list[str] | None = None
    classe_analisada: list[ClasseStartup] | None = None

    @field_validator("estagio", "tamanho_time", "classe_analisada")
    @classmethod
    def listas_nao_vazias(cls, valor: list[str] | None) -> list[str] | None:
        if valor == []:
            raise ValueError("use null quando não houver filtro")
        return valor


class PlanoConsulta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filtros: FiltrosEstruturados = Field(default_factory=FiltrosEstruturados)
    termos_busca: list[str] = Field(min_length=1, max_length=8)
    sinais_ia: list[str] = Field(default_factory=list, max_length=6)
    foco_analise: str = Field(min_length=1)

    @field_validator("termos_busca", "sinais_ia")
    @classmethod
    def limpar_termos(cls, valores: list[str]) -> list[str]:
        limpos: list[str] = []
        for valor in valores:
            termo = valor.strip()
            if termo and termo.casefold() not in {item.casefold() for item in limpos}:
                limpos.append(termo)
        if not limpos and valores:
            raise ValueError("os termos não podem conter apenas espaços")
        return limpos


class EmpresaCandidata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_startup: int
    nome: str
    setor: str
    estagio: str
    localizacao: str | None
    descricao_curta: str | None


class DocumentoRecuperado(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_documento: int
    id_startup: int
    tipo: TipoDocumento
    titulo: str
    url_fonte: str
    dominio_fonte: str
    data_acesso: date
    score_bm25: float


class ResultadoRecuperacao(BaseModel):
    model_config = ConfigDict(extra="forbid")

    empresas: list[EmpresaCandidata]
    documentos: list[DocumentoRecuperado]
    filtros_aplicados: FiltrosEstruturados


class DocumentoIntegral(BaseModel):
    """Documento completo lido do SQLite para os nós que precisam do texto.

    Nunca entra no ``EstadoRadar``: o checkpoint carrega ids, não documentos.
    """

    model_config = ConfigDict(extra="forbid")

    id_documento: int
    id_startup: int
    tipo: TipoDocumento
    titulo: str
    conteudo_texto: str


class DocumentoVerificavel(BaseModel):
    """Projeção mínima usada para conferir proveniência: texto e domínio.

    É deliberadamente mais estreita que ``DocumentoIntegral``: o Evidence
    Validator precisa do domínio para contar hosts, mas não precisa de título
    nem de tipo — e jamais de ``classe_referencia``, que continua invisível
    para todo o núcleo. Como ``DocumentoIntegral``, nunca entra no
    ``EstadoRadar``.
    """

    model_config = ConfigDict(extra="forbid")

    id_documento: int
    id_startup: int
    conteudo_texto: str
    dominio_fonte: str


class Afirmacao(BaseModel):
    """A unidade de evidência do sistema: um fato com proveniência literal."""

    model_config = ConfigDict(extra="forbid")

    id_afirmacao: int = Field(ge=1)
    texto: str = Field(min_length=1)
    categoria: CategoriaAfirmacao
    polaridade: PolaridadeAfirmacao
    id_documento: int
    trecho_citado: str = Field(
        min_length=MINIMO_CARACTERES_TRECHO_CITADO,
        max_length=LIMITE_TRECHO_CITADO,
    )

    @field_validator("texto", "trecho_citado")
    @classmethod
    def sem_conteudo_em_branco(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("o campo não pode conter apenas espaços")
        # O trecho citado não é normalizado aqui: ele precisa continuar literal.
        return valor

    @field_validator("texto")
    @classmethod
    def texto_e_uma_unica_frase(cls, valor: str) -> str:
        if contar_frases(valor) != 1:
            raise ValueError(
                "texto deve conter um único fato, em uma frase terminada em pontuação"
            )
        return valor

    @field_validator("trecho_citado")
    @classmethod
    def trecho_tem_conteudo_significativo(cls, valor: str) -> str:
        palavras = re.findall(r"\w+", normalizar_texto_citavel(valor))
        if len(palavras) < MINIMO_PALAVRAS_TRECHO_CITADO:
            raise ValueError(
                "trecho_citado precisa ter ao menos "
                f"{MINIMO_PALAVRAS_TRECHO_CITADO} palavras"
            )
        return valor

    @model_validator(mode="after")
    def polaridade_compativel_com_categoria(self) -> Afirmacao:
        if self.categoria not in CATEGORIAS_ESTRUTURAIS and self.polaridade != "neutro":
            raise ValueError(
                "presenca e ausencia_explicita só valem nas quatro categorias "
                "estruturais; as demais categorias usam polaridade 'neutro'"
            )
        return self


class PerfilExtraido(BaseModel):
    """Saída do Extractor: o que a empresa vende e os fatos que sustentam isso."""

    model_config = ConfigDict(extra="forbid")

    id_startup: int
    resumo_produto: str = Field(min_length=1)
    afirmacoes: list[Afirmacao] = Field(min_length=1, max_length=20)

    @field_validator("resumo_produto")
    @classmethod
    def resumo_tem_duas_ou_tres_frases(cls, valor: str) -> str:
        if contar_frases(valor) not in (2, 3):
            raise ValueError(
                "resumo_produto deve ter 2 ou 3 frases terminadas em pontuação"
            )
        return valor

    @model_validator(mode="after")
    def ids_de_afirmacao_sao_sequenciais(self) -> PerfilExtraido:
        observados = [afirmacao.id_afirmacao for afirmacao in self.afirmacoes]
        if observados != list(range(1, len(observados) + 1)):
            raise ValueError(
                "id_afirmacao deve ser sequencial a partir de 1, na ordem da lista"
            )
        return self


class Classificacao(BaseModel):
    """Saída do Classifier: a classe, o porquê e as afirmações que o sustentam.

    O contrato garante a forma; a existência dos ids de suporte dentro do
    perfil analisado é verificada na fronteira do nó, que é quem conhece o
    ``PerfilExtraido`` correspondente.
    """

    model_config = ConfigDict(extra="forbid")

    classe: ClasseStartup
    justificativa: str = Field(min_length=1)
    ids_afirmacoes_suporte: list[int] = Field(min_length=1)

    @field_validator("justificativa")
    @classmethod
    def justificativa_tem_de_duas_a_quatro_frases(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("justificativa não pode conter apenas espaços")
        if not (
            MINIMO_FRASES_JUSTIFICATIVA
            <= contar_frases(valor)
            <= MAXIMO_FRASES_JUSTIFICATIVA
        ):
            raise ValueError(
                "justificativa deve ter de "
                f"{MINIMO_FRASES_JUSTIFICATIVA} a {MAXIMO_FRASES_JUSTIFICATIVA} "
                "frases terminadas em pontuação"
            )
        return valor

    @field_validator("ids_afirmacoes_suporte")
    @classmethod
    def suporte_sem_repeticao_e_com_ids_validos(cls, valores: list[int]) -> list[int]:
        if any(valor < 1 for valor in valores):
            raise ValueError("id_afirmacao começa em 1")
        if len(set(valores)) != len(valores):
            raise ValueError("ids_afirmacoes_suporte não pode repetir ids duplicados")
        return valores


class AfirmacaoValidada(Afirmacao):
    """A afirmação original acrescida do veredito de proveniência.

    Herdar de ``Afirmacao`` preserva todos os campos e todas as regras da
    unidade de evidência; a validação de evidência acrescenta o veredito sem
    ter permissão para reescrever o fato citado.
    """

    model_config = ConfigDict(extra="forbid")

    situacao: SituacaoAfirmacao
    motivo: str | None = Field(default=None, max_length=MAXIMO_CARACTERES_MOTIVO)

    @model_validator(mode="after")
    def motivo_corresponde_a_situacao(self) -> AfirmacaoValidada:
        if self.situacao == "confirmada" and self.motivo is not None:
            raise ValueError(
                "afirmação confirmada não carrega motivo; motivo deve ser nulo"
            )
        if self.situacao == "derrubada" and not (self.motivo or "").strip():
            raise ValueError(
                "afirmação derrubada exige um motivo conciso e não vazio"
            )
        return self


class EstadoDimensaoGap(BaseModel):
    """O que a evidência confirmada permite dizer sobre uma dimensão estrutural."""

    model_config = ConfigDict(extra="forbid")

    dimensao: DimensaoGap
    estado: EstadoGap
    ids_evidencias: list[int] = Field(default_factory=list)

    @field_validator("ids_evidencias")
    @classmethod
    def evidencias_sem_repeticao_e_com_ids_validos(cls, valores: list[int]) -> list[int]:
        if any(valor < 1 for valor in valores):
            raise ValueError("id_afirmacao começa em 1")
        if len(set(valores)) != len(valores):
            raise ValueError("ids_evidencias não pode repetir ids duplicados")
        if valores != sorted(valores):
            raise ValueError("ids_evidencias precisa vir em ordem determinística")
        return valores

    @model_validator(mode="after")
    def estado_decisivo_exige_evidencia(self) -> EstadoDimensaoGap:
        if self.estado != "desconhecido" and not self.ids_evidencias:
            raise ValueError(
                f"o estado {self.estado} afirma uma conclusão sobre "
                f"{self.dimensao} e exige ao menos uma evidência que a sustente"
            )
        return self


class PerfilValidado(BaseModel):
    """Saída do Evidence Validator: o perfil depois da conferência de proveniência."""

    model_config = ConfigDict(extra="forbid")

    afirmacoes_validadas: list[AfirmacaoValidada] = Field(min_length=1)
    taxa_derrubada: float = Field(ge=0.0, le=1.0)
    hosts_distintos: list[str] = Field(default_factory=list)
    estado_dimensoes_gap: list[EstadoDimensaoGap]

    @field_validator("hosts_distintos")
    @classmethod
    def hosts_normalizados_unicos_e_ordenados(cls, valores: list[str]) -> list[str]:
        for host in valores:
            if not host.strip():
                raise ValueError("host não pode ser vazio")
            if host != normalizar_dominio(host):
                raise ValueError(
                    f"host {host!r} precisa vir normalizado em minúsculas e sem 'www.'"
                )
        if len(set(valores)) != len(valores):
            raise ValueError("hosts_distintos não pode repetir hosts")
        if valores != sorted(valores):
            raise ValueError("hosts_distintos precisa vir em ordem determinística")
        return valores

    @model_validator(mode="after")
    def ids_de_afirmacao_sao_sequenciais(self) -> PerfilValidado:
        observados = [item.id_afirmacao for item in self.afirmacoes_validadas]
        if observados != list(range(1, len(observados) + 1)):
            raise ValueError(
                "id_afirmacao deve ser sequencial a partir de 1, na ordem da lista"
            )
        return self

    @model_validator(mode="after")
    def taxa_corresponde_as_situacoes(self) -> PerfilValidado:
        derrubadas = sum(
            1 for item in self.afirmacoes_validadas if item.situacao == "derrubada"
        )
        esperada = derrubadas / len(self.afirmacoes_validadas)
        if not math.isclose(
            self.taxa_derrubada, esperada, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"taxa_derrubada declarada {self.taxa_derrubada} não corresponde a "
                f"{derrubadas} derrubadas em {len(self.afirmacoes_validadas)} "
                f"afirmações ({esperada})"
            )
        return self

    @model_validator(mode="after")
    def dimensoes_completas_na_ordem_do_contrato(self) -> PerfilValidado:
        observadas = tuple(item.dimensao for item in self.estado_dimensoes_gap)
        if observadas != DIMENSOES_GAP:
            raise ValueError(
                "estado_dimensoes_gap deve trazer exatamente as quatro dimensões "
                f"estruturais, uma vez cada, nesta ordem: {list(DIMENSOES_GAP)}"
            )
        return self

    @model_validator(mode="after")
    def dimensoes_derivam_das_afirmacoes_confirmadas(self) -> PerfilValidado:
        """Cada dimensão precisa ser exatamente o que a evidência confirmada diz.

        O artefato não pode afirmar capacidade sem afirmação de presença
        confirmada, nem descartar um dos lados de um conflito, nem citar
        evidência derrubada, de outra polaridade ou de outra dimensão.
        """
        for item in self.estado_dimensoes_gap:
            presencas = self._ids_confirmados(item.dimensao, "presenca")
            ausencias = self._ids_confirmados(item.dimensao, "ausencia_explicita")
            if presencas and ausencias:
                esperado = ("desconhecido", sorted(presencas + ausencias))
            elif presencas:
                esperado = ("capacidade_confirmada", presencas)
            elif ausencias:
                esperado = ("gap_confirmado", ausencias)
            else:
                esperado = ("desconhecido", [])
            if (item.estado, item.ids_evidencias) != esperado:
                raise ValueError(
                    f"a dimensão {item.dimensao} declara "
                    f"{item.estado} com evidências {item.ids_evidencias}, mas as "
                    f"afirmações confirmadas sustentam {esperado[0]} com "
                    f"{esperado[1]}"
                )
        return self

    def _ids_confirmados(self, dimensao: str, polaridade: str) -> list[int]:
        return sorted(
            item.id_afirmacao
            for item in self.afirmacoes_validadas
            if item.situacao == "confirmada"
            and item.categoria == dimensao
            and item.polaridade == polaridade
        )


class DocumentoCurado(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: TipoDocumento
    titulo: str = Field(min_length=3)
    conteudo_texto: str = Field(min_length=80)
    url_fonte: AnyHttpUrl
    dominio_fonte: str
    data_publicacao: date | None = None
    data_acesso: date

    @model_validator(mode="after")
    def dominio_corresponde_a_url(self) -> DocumentoCurado:
        host = urlparse(str(self.url_fonte)).hostname or ""
        if normalizar_dominio(host) != normalizar_dominio(self.dominio_fonte):
            raise ValueError("dominio_fonte deve corresponder ao host de url_fonte")
        self.dominio_fonte = normalizar_dominio(self.dominio_fonte)
        return self


class StartupCurada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nome: str = Field(min_length=2)
    site: AnyHttpUrl
    setor: str
    estagio: str
    localizacao: str | None = None
    descricao_curta: str
    ano_fundacao: int | None = Field(default=None, ge=1900, le=2100)
    tamanho_time: str
    classe_referencia: ClasseStartup
    documentos: list[DocumentoCurado] = Field(min_length=3)

    @model_validator(mode="after")
    def tres_dominios_distintos(self) -> StartupCurada:
        dominios = {doc.dominio_fonte for doc in self.documentos}
        if len(dominios) < 3:
            raise ValueError("cada startup precisa de pelo menos três domínios distintos")
        if len({str(doc.url_fonte) for doc in self.documentos}) != len(self.documentos):
            raise ValueError("as URLs de documentos devem ser únicas")
        return self


TECNOLOGIAS_NVIDIA: tuple[str, ...] = (
    "NVIDIA Inception",
    "NVIDIA NIM",
    "NVIDIA NeMo",
    "NeMo Guardrails",
    "NVIDIA Triton Inference Server",
    "TensorRT-LLM",
    "NVIDIA RAPIDS",
    "cuDF",
    "cuML",
    "CUDA",
    "NVIDIA Riva",
    "NVIDIA Omniverse",
    "NVIDIA Isaac",
    "NVIDIA Clara",
    "NVIDIA Morpheus",
    "NVIDIA AI Enterprise",
)

TecnologiaNvidia = Literal[
    "NVIDIA Inception",
    "NVIDIA NIM",
    "NVIDIA NeMo",
    "NeMo Guardrails",
    "NVIDIA Triton Inference Server",
    "TensorRT-LLM",
    "NVIDIA RAPIDS",
    "cuDF",
    "cuML",
    "CUDA",
    "NVIDIA Riva",
    "NVIDIA Omniverse",
    "NVIDIA Isaac",
    "NVIDIA Clara",
    "NVIDIA Morpheus",
    "NVIDIA AI Enterprise",
]

OrigemChunk = Literal["tecnologia", "conceitual"]


class ItemCorpusNvidia(BaseModel):
    """Regra comum do corpus NVIDIA: origem e tecnologia andam juntas."""

    model_config = ConfigDict(extra="forbid")

    topico: str = Field(min_length=2)
    origem: OrigemChunk
    tecnologia: TecnologiaNvidia | None = None

    @model_validator(mode="after")
    def origem_compativel_com_tecnologia(self) -> ItemCorpusNvidia:
        if (self.origem == "tecnologia") != (self.tecnologia is not None):
            raise ValueError(
                "origem 'tecnologia' exige uma tecnologia do TAPI; "
                "origem 'conceitual' exige tecnologia nula"
            )
        return self


class FonteNvidia(ItemCorpusNvidia):
    """Metadados obrigatórios de um arquivo curado da base de conhecimento."""

    fonte_url: AnyHttpUrl
    titulo: str = Field(min_length=2)
    data_acesso: date


class ChunkNvidia(ItemCorpusNvidia):
    """Unidade ingerível da base de conhecimento, antes do id do banco."""

    fonte_url: AnyHttpUrl
    breadcrumb: str = Field(min_length=1)
    texto_limpo: str = Field(min_length=1)
    indice_parte: int = Field(ge=1)
    hash_texto: str = Field(min_length=64, max_length=64)


class TrechoNvidia(ItemCorpusNvidia):
    """Trecho recuperado, pronto para citação com rastreabilidade completa."""

    id_chunk: int
    breadcrumb: str = Field(min_length=1)
    texto: str = Field(min_length=1)
    fonte_url: AnyHttpUrl
    score_rerank: float


class ContextoNvidia(BaseModel):
    """Saída do RAG NVIDIA: 5 a 8 trechos reordenados pelo reranking."""

    model_config = ConfigDict(extra="forbid")

    consulta_gerada: str = Field(min_length=1)
    trechos: list[TrechoNvidia] = Field(min_length=5, max_length=8)


class ProximaAcao(BaseModel):
    """Ação operacional escolhida pelo LLM dentro de um catálogo fechado."""

    model_config = ConfigDict(extra="forbid")

    tipo_acao: TipoAcao
    detalhe: str = Field(min_length=1)

    @field_validator("detalhe")
    @classmethod
    def detalhe_nao_pode_ser_branco(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("detalhe não pode conter apenas espaços")
        if contar_frases(valor) != 1:
            raise ValueError("detalhe precisa ter exatamente uma frase")
        return valor


class RecomendacaoRascunho(BaseModel):
    """Único schema que o LLM de Recommendation tem permissão para preencher."""

    model_config = ConfigDict(extra="forbid")

    gap_enderecado: GapEnderecado
    tecnologias: list[TecnologiaNvidia] = Field(min_length=1, max_length=3)
    justificativa_tecnica: str = Field(min_length=1)
    justificativa_negocio: str = Field(min_length=1)
    proxima_acao: ProximaAcao
    ids_afirmacoes: list[int] = Field(min_length=1)
    ids_chunks: list[int] = Field(min_length=1)

    @field_validator("justificativa_tecnica", "justificativa_negocio")
    @classmethod
    def justificativa_nao_pode_ser_branca(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("justificativa não pode conter apenas espaços")
        return valor

    @field_validator("tecnologias", "ids_afirmacoes", "ids_chunks")
    @classmethod
    def listas_do_rascunho_nao_repetem_itens(cls, valores: list) -> list:
        if len(set(valores)) != len(valores):
            raise ValueError("o rascunho não pode repetir itens")
        if valores and isinstance(valores[0], int) and any(valor < 1 for valor in valores):
            raise ValueError("ids começam em 1")
        return valores


class EvidenciaStartup(BaseModel):
    """Afirmação confirmada resolvida para a fonte pública correspondente."""

    model_config = ConfigDict(extra="forbid")

    id_afirmacao: int = Field(ge=1)
    id_documento: int = Field(ge=1)
    url_fonte: AnyHttpUrl
    trecho_citado: str = Field(min_length=1, max_length=LIMITE_TRECHO_CITADO)

    @field_validator("trecho_citado")
    @classmethod
    def evidencia_nao_pode_ser_branca(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("trecho_citado não pode conter apenas espaços")
        return valor


class CitacaoNvidia(ItemCorpusNvidia):
    """Chunk NVIDIA resolvido, com os metadados necessários para auditoria."""

    model_config = ConfigDict(extra="forbid")

    id_chunk: int = Field(ge=1)
    fonte_url: AnyHttpUrl
    breadcrumb: str = Field(min_length=1)


class Recomendacao(BaseModel):
    """Pacote por gap com proveniência obrigatória nos dois lados."""

    model_config = ConfigDict(extra="forbid")

    gap_enderecado: GapEnderecado
    tecnologias: list[TecnologiaNvidia] = Field(min_length=1, max_length=3)
    justificativa_tecnica: str = Field(min_length=1)
    justificativa_negocio: str = Field(min_length=1)
    prioridade: PrioridadeRecomendacao
    complexidade: ComplexidadeRecomendacao
    proxima_acao: ProximaAcao
    evidencias_startup: list[EvidenciaStartup] = Field(min_length=1)
    citacoes_nvidia: list[CitacaoNvidia] = Field(min_length=1)

    @field_validator("justificativa_tecnica", "justificativa_negocio")
    @classmethod
    def justificativa_final_nao_pode_ser_branca(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("justificativa não pode conter apenas espaços")
        return valor

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_sem_repeticao(cls, valores: list[str]) -> list[str]:
        if len(set(valores)) != len(valores):
            raise ValueError("tecnologias não pode repetir itens")
        return valores

    @model_validator(mode="after")
    def proveniencia_e_unica_e_inclui_tecnologia(self) -> Recomendacao:
        ids_afirmacoes = [item.id_afirmacao for item in self.evidencias_startup]
        if len(set(ids_afirmacoes)) != len(ids_afirmacoes):
            raise ValueError("evidencias_startup não pode repetir id_afirmacao")

        ids_chunks = [item.id_chunk for item in self.citacoes_nvidia]
        if len(set(ids_chunks)) != len(ids_chunks):
            raise ValueError("citacoes_nvidia não pode repetir id_chunk")

        if not any(
            item.origem == "tecnologia" and item.tecnologia is not None
            for item in self.citacoes_nvidia
        ):
            raise ValueError(
                "ao menos uma citação NVIDIA precisa vir de um chunk de tecnologia"
            )
        return self


class RelatorioRecomendacoes(BaseModel):
    """Saída validada do nó; variantes terminais podem manter a lista vazia."""

    model_config = ConfigDict(extra="forbid")

    recomendacoes: list[Recomendacao] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def cada_gap_entra_uma_unica_vez(self) -> RelatorioRecomendacoes:
        """Defesa em profundidade da §6.1: um pacote coeso **por gap**.

        O nó já descarta a duplicata antes do retry; esta guarda garante que
        nenhum outro caminho de construção monte um relatório com o mesmo gap
        duas vezes. Duplicata é descartada, nunca fundida: dois pacotes para o
        mesmo gap são duas respostas concorrentes, e escolher uma é decisão do
        nó, não do contrato.
        """
        gaps = [item.gap_enderecado for item in self.recomendacoes]
        repetidos = sorted({gap for gap in gaps if gaps.count(gap) > 1})
        if repetidos:
            raise ValueError(
                "o relatório não pode trazer mais de uma recomendação para o "
                f"mesmo gap; repetidos: {repetidos}"
            )
        return self


class MetadadoDocumentoFitScore(BaseModel):
    """Metadado explícito que mantém a função de score longe do SQLite."""

    model_config = ConfigDict(extra="forbid")

    id_documento: int = Field(ge=1)
    url_fonte: AnyHttpUrl
    host_normalizado: str = Field(min_length=1)
    data_publicacao: date | None = None

    @field_validator("host_normalizado")
    @classmethod
    def host_precisa_estar_normalizado(cls, valor: str) -> str:
        if valor != normalizar_dominio(valor):
            raise ValueError("host_normalizado deve estar em minúsculas e sem 'www.'")
        return valor

    @model_validator(mode="after")
    def host_corresponde_a_url(self) -> MetadadoDocumentoFitScore:
        host_url = normalizar_dominio(urlparse(str(self.url_fonte)).hostname or "")
        if self.host_normalizado != host_url:
            raise ValueError("host_normalizado precisa corresponder a url_fonte")
        return self


class EntradaFitScore(BaseModel):
    """Todos os dados necessários para pontuar, inclusive a data de referência."""

    model_config = ConfigDict(extra="forbid")

    classe: ClasseStartup
    ids_afirmacoes_suporte_classe: list[int] = Field(min_length=1)
    perfil_validado: PerfilValidado
    setor: str = Field(min_length=1)
    estagio: str = Field(min_length=1)
    documentos: list[MetadadoDocumentoFitScore] = Field(min_length=1)
    data_referencia: date

    @field_validator("ids_afirmacoes_suporte_classe")
    @classmethod
    def suporte_da_classe_e_unico_e_ordenado(cls, valores: list[int]) -> list[int]:
        if any(valor < 1 for valor in valores):
            raise ValueError("id_afirmacao começa em 1")
        if len(set(valores)) != len(valores):
            raise ValueError("ids de suporte da classe não podem se repetir")
        if valores != sorted(valores):
            raise ValueError("ids de suporte da classe precisam estar ordenados")
        return valores

    @model_validator(mode="after")
    def referencias_do_score_sao_completas(self) -> EntradaFitScore:
        documentos_por_id = {item.id_documento: item for item in self.documentos}
        if len(documentos_por_id) != len(self.documentos):
            raise ValueError("documentos do fit-score não podem repetir id_documento")

        ids_confirmados = {
            item.id_afirmacao
            for item in self.perfil_validado.afirmacoes_validadas
            if item.situacao == "confirmada"
        }
        if not set(self.ids_afirmacoes_suporte_classe).issubset(ids_confirmados):
            raise ValueError(
                "o suporte da classe precisa referenciar apenas afirmações confirmadas"
            )

        ids_documentos_necessarios = {
            item.id_documento
            for item in self.perfil_validado.afirmacoes_validadas
        }
        ausentes = ids_documentos_necessarios - set(documentos_por_id)
        if ausentes:
            raise ValueError(
                "faltam metadados dos documentos referenciados pelo perfil: "
                f"{sorted(ausentes)}"
            )
        return self


class PilarFitScore(BaseModel):
    """Explicação auditável de um dos quatro pilares do fit-score."""

    model_config = ConfigDict(extra="forbid")

    pilar: PilarFit
    pontos: int = Field(ge=0, le=10)
    faixa: FaixaFit
    ids_evidencias: list[int] = Field(default_factory=list)
    travas_aplicadas: list[TravaFit] = Field(default_factory=list)

    @field_validator("ids_evidencias")
    @classmethod
    def ids_do_pilar_sao_unicos_e_ordenados(cls, valores: list[int]) -> list[int]:
        if any(valor < 1 for valor in valores):
            raise ValueError("id_afirmacao começa em 1")
        if len(set(valores)) != len(valores) or valores != sorted(valores):
            raise ValueError("ids_evidencias precisam ser únicos e ordenados")
        return valores

    @field_validator("travas_aplicadas")
    @classmethod
    def travas_sao_unicas_e_na_ordem_do_contrato(
        cls, valores: list[str]
    ) -> list[str]:
        if len(set(valores)) != len(valores):
            raise ValueError("travas_aplicadas não pode repetir itens")
        esperadas = [item for item in TRAVAS_FIT if item in valores]
        if valores != esperadas:
            raise ValueError("travas_aplicadas precisa seguir a ordem do contrato")
        return valores

    @model_validator(mode="after")
    def faixa_corresponde_aos_pontos(self) -> PilarFitScore:
        esperada: FaixaFit
        if self.pontos <= 3:
            esperada = "baixa"
        elif self.pontos <= 7:
            esperada = "media"
        else:
            esperada = "alta"
        if self.faixa != esperada:
            raise ValueError(
                f"faixa {self.faixa!r} não corresponde a {self.pontos} pontos"
            )
        return self


class FitScore(BaseModel):
    """Pontuação final normalizada, com os componentes que permitem auditá-la."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0, le=100)
    pilares: list[PilarFitScore]
    estado_dimensoes_gap: list[EstadoDimensaoGap]
    justificativa_curta: str = Field(min_length=1)
    versao_rubrica: Literal["rubrica-v1"]

    @model_validator(mode="after")
    def componentes_seguem_a_ordem_do_contrato(self) -> FitScore:
        pilares = tuple(item.pilar for item in self.pilares)
        if pilares != PILARES_FIT:
            raise ValueError(
                "pilares deve conter os quatro pilares uma vez, na ordem do contrato"
            )
        dimensoes = tuple(item.dimensao for item in self.estado_dimensoes_gap)
        if dimensoes != DIMENSOES_GAP:
            raise ValueError(
                "estado_dimensoes_gap deve seguir a ordem das quatro dimensões"
            )

        gates_non_ai = [
            "gate_non_ai" in item.travas_aplicadas for item in self.pilares
        ]
        if any(gates_non_ai):
            if (
                not all(gates_non_ai)
                or self.total != 0
                or any(item.pontos != 0 for item in self.pilares)
            ):
                raise ValueError(
                    "gate_non_ai precisa alcançar os quatro pilares e zerar "
                    "pilares e total"
                )
        else:
            esperado = round(
                100 * sum(item.pontos for item in self.pilares) / 36
            )
            if self.total != esperado:
                raise ValueError(
                    f"total {self.total} não corresponde à normalização {esperado}"
                )
        return self


class EstadoRadar(TypedDict, total=False):
    consulta_usuario: str
    startup_selecionada: int | None
    plano_consulta: PlanoConsulta
    tentativas_relaxamento: int
    criterios_relaxados: Annotated[list[str], operator.add]
    resultado_recuperacao: ResultadoRecuperacao
    # Os quatro campos de análise aceitam ``None`` porque o grafo os invalida
    # deliberadamente quando a extração ou a recuperação que os originou é
    # substituída; um artefato órfão é pior que a ausência dele.
    perfil_extraido: PerfilExtraido | None
    tentativas_extracao: int
    classificacao: Classificacao | None
    perfil_validado: PerfilValidado | None
    confianca_perfil: ConfiancaPerfil | None
    # ``ContextoNvidia`` e ``Recomendacao`` carregam ``AnyHttpUrl``, que o
    # ``JsonPlusSerializer`` do checkpointer não serializa. O grafo grava a
    # forma JSON do mesmo contrato — por isso a anotação é o dicionário, e não
    # o modelo — e todo consumidor reidrata com ``model_validate`` na fronteira.
    # ``FitScore`` não tem URL e por isso continua atravessando como instância.
    contexto_nvidia: dict[str, Any] | None
    recomendacoes: list[dict[str, Any]] | None
    fit_score: FitScore | None
    erros: Annotated[list[str], operator.add]
    trajeto: Annotated[list[str], operator.add]
