from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict
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


def normalizar_dominio(valor: str) -> str:
    dominio = valor.strip().lower()
    return dominio[4:] if dominio.startswith("www.") else dominio


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


class EstadoRadar(TypedDict, total=False):
    consulta_usuario: str
    startup_selecionada: int | None
    plano_consulta: PlanoConsulta
    tentativas_relaxamento: int
    criterios_relaxados: Annotated[list[str], operator.add]
    resultado_recuperacao: ResultadoRecuperacao
    erros: Annotated[list[str], operator.add]
    trajeto: Annotated[list[str], operator.add]

