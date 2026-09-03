from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import TypeAdapter

from radar.configuracao import TETO_DOCUMENTOS_DESCOBERTA
from radar.contratos import (
    DocumentoIntegral,
    FonteBriefing,
    DocumentoRecuperado,
    DocumentoVerificavel,
    EmpresaCandidata,
    FiltrosEstruturados,
    MetadadoDocumentoFitScore,
    PlanoConsulta,
    ResultadoRecuperacao,
    StartupCurada,
    normalizar_dominio,
)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS startups (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL UNIQUE,
    site TEXT NOT NULL,
    setor TEXT NOT NULL,
    estagio TEXT NOT NULL,
    localizacao TEXT,
    descricao_curta TEXT,
    ano_fundacao INTEGER,
    tamanho_time TEXT NOT NULL,
    classe_referencia TEXT NOT NULL
        CHECK (classe_referencia IN ('AI-native', 'AI-enabled', 'non-AI'))
);

CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY,
    startup_id INTEGER NOT NULL REFERENCES startups(id) ON DELETE CASCADE,
    tipo TEXT NOT NULL CHECK (
        tipo IN ('site institucional', 'blog', 'notícia', 'vaga', 'perfil de founder', 'release')
    ),
    titulo TEXT NOT NULL,
    conteudo_texto TEXT NOT NULL,
    url_fonte TEXT NOT NULL UNIQUE,
    dominio_fonte TEXT NOT NULL,
    data_publicacao TEXT,
    data_acesso TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analises (
    startup_id INTEGER PRIMARY KEY REFERENCES startups(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('concluida', 'evidencia_insuficiente')),
    classe TEXT CHECK (classe IN ('AI-native', 'AI-enabled', 'non-AI')),
    fit_score_total INTEGER CHECK (fit_score_total BETWEEN 0 AND 100),
    fit_score_json TEXT,
    perfil_validado_json TEXT NOT NULL,
    data_execucao TEXT NOT NULL,
    versao_rubrica TEXT NOT NULL,
    CHECK (
        (status = 'concluida' AND classe IS NOT NULL
            AND fit_score_total IS NOT NULL AND fit_score_json IS NOT NULL)
        OR
        (status = 'evidencia_insuficiente' AND classe IS NULL
            AND fit_score_total IS NULL AND fit_score_json IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_startups_setor ON startups(setor);
CREATE INDEX IF NOT EXISTS idx_startups_estagio ON startups(estagio);
CREATE INDEX IF NOT EXISTS idx_startups_localizacao ON startups(localizacao);
CREATE INDEX IF NOT EXISTS idx_startups_tamanho_time ON startups(tamanho_time);
CREATE INDEX IF NOT EXISTS idx_documentos_startup ON documentos(startup_id);
CREATE INDEX IF NOT EXISTS idx_analises_status_classe ON analises(status, classe);

CREATE VIRTUAL TABLE IF NOT EXISTS documentos_fts USING fts5(
    titulo,
    conteudo_texto,
    content='documentos',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
"""


class ErroDocumentosStartup(RuntimeError):
    """Os ids pedidos não descrevem os documentos daquela startup na base."""


@dataclass(frozen=True)
class ConsultaParametrizada:
    sql: str
    parametros: tuple[Any, ...]


def conectar(caminho_banco: Path) -> sqlite3.Connection:
    conexao = sqlite3.connect(caminho_banco)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA foreign_keys = ON")
    return conexao


def carregar_curadoria(diretorio: Path) -> list[StartupCurada]:
    arquivos = sorted(diretorio.glob("*.json"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo de curadoria encontrado em {diretorio}")
    adaptador = TypeAdapter(StartupCurada)
    startups: list[StartupCurada] = []
    for arquivo in arquivos:
        with arquivo.open(encoding="utf-8") as entrada:
            startups.append(adaptador.validate_python(json.load(entrada)))
    nomes = [startup.nome.casefold() for startup in startups]
    if len(nomes) != len(set(nomes)):
        raise ValueError("nomes de startup repetidos na curadoria")
    return startups


def inicializar_banco(caminho_banco: Path, diretorio_curadoria: Path) -> None:
    startups = carregar_curadoria(diretorio_curadoria)
    caminho_banco.parent.mkdir(parents=True, exist_ok=True)
    with conectar(caminho_banco) as conexao:
        conexao.executescript(SCHEMA_SQL)
        for startup in startups:
            conexao.execute(
                """
                INSERT INTO startups (
                    nome, site, setor, estagio, localizacao, descricao_curta,
                    ano_fundacao, tamanho_time, classe_referencia
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(nome) DO UPDATE SET
                    site = excluded.site,
                    setor = excluded.setor,
                    estagio = excluded.estagio,
                    localizacao = excluded.localizacao,
                    descricao_curta = excluded.descricao_curta,
                    ano_fundacao = excluded.ano_fundacao,
                    tamanho_time = excluded.tamanho_time,
                    classe_referencia = excluded.classe_referencia
                """,
                (
                    startup.nome,
                    str(startup.site),
                    startup.setor,
                    startup.estagio,
                    startup.localizacao,
                    startup.descricao_curta,
                    startup.ano_fundacao,
                    startup.tamanho_time,
                    startup.classe_referencia,
                ),
            )
            startup_id = conexao.execute(
                "SELECT id FROM startups WHERE nome = ?", (startup.nome,)
            ).fetchone()["id"]
            urls_atuais = [str(documento.url_fonte) for documento in startup.documentos]
            marcadores = ", ".join("?" for _ in urls_atuais)
            conexao.execute(
                f"DELETE FROM documentos WHERE startup_id = ? AND url_fonte NOT IN ({marcadores})",
                (startup_id, *urls_atuais),
            )
            for documento in startup.documentos:
                conexao.execute(
                    """
                    INSERT INTO documentos (
                        startup_id, tipo, titulo, conteudo_texto, url_fonte,
                        dominio_fonte, data_publicacao, data_acesso
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url_fonte) DO UPDATE SET
                        startup_id = excluded.startup_id,
                        tipo = excluded.tipo,
                        titulo = excluded.titulo,
                        conteudo_texto = excluded.conteudo_texto,
                        dominio_fonte = excluded.dominio_fonte,
                        data_publicacao = excluded.data_publicacao,
                        data_acesso = excluded.data_acesso
                    """,
                    (
                        startup_id,
                        documento.tipo,
                        documento.titulo,
                        documento.conteudo_texto,
                        str(documento.url_fonte),
                        documento.dominio_fonte,
                        documento.data_publicacao.isoformat()
                        if documento.data_publicacao
                        else None,
                        documento.data_acesso.isoformat(),
                    ),
                )
        # A base é estática durante a execução; o índice é reconstruído após o seed.
        conexao.execute("INSERT INTO documentos_fts(documentos_fts) VALUES('rebuild')")


def construir_match_fts(termos: Iterable[str]) -> str:
    frases: list[str] = []
    for termo in termos:
        limpo = termo.strip()
        if limpo:
            frases.append(f'"{limpo.replace(chr(34), chr(34) * 2)}"')
    if not frases:
        raise ValueError("a busca FTS5 exige pelo menos um termo")
    return " OR ".join(frases)


def montar_consulta_descoberta(
    plano: PlanoConsulta, limite: int = TETO_DOCUMENTOS_DESCOBERTA
) -> ConsultaParametrizada:
    filtros = plano.filtros
    condicoes = ["documentos_fts MATCH ?"]
    parametros: list[Any] = [construir_match_fts([*plano.termos_busca, *plano.sinais_ia])]

    def adicionar_igual(coluna: str, valor: str | None) -> None:
        if valor is not None:
            condicoes.append(f"LOWER({coluna}) = LOWER(?)")
            parametros.append(valor)

    def adicionar_lista(coluna: str, valores: list[str] | None) -> None:
        if valores:
            marcadores = ", ".join("?" for _ in valores)
            condicoes.append(f"LOWER({coluna}) IN ({marcadores})")
            parametros.extend(valor.lower() for valor in valores)

    adicionar_igual("s.setor", filtros.setor)
    adicionar_lista("s.estagio", filtros.estagio)
    adicionar_igual("s.localizacao", filtros.localizacao)
    adicionar_lista("s.tamanho_time", filtros.tamanho_time)
    if filtros.classe_analisada:
        marcadores = ", ".join("?" for _ in filtros.classe_analisada)
        condicoes.append(
            "EXISTS (SELECT 1 FROM analises a "
            "WHERE a.startup_id = s.id AND a.status = 'concluida' "
            f"AND a.classe IN ({marcadores}))"
        )
        parametros.extend(filtros.classe_analisada)

    sql = f"""
        SELECT
            d.id AS id_documento,
            d.startup_id AS id_startup,
            d.tipo,
            d.titulo,
            d.url_fonte,
            d.dominio_fonte,
            d.data_acesso,
            bm25(documentos_fts) AS score_bm25,
            s.nome,
            s.setor,
            s.estagio,
            s.localizacao,
            s.descricao_curta
        FROM documentos_fts
        JOIN documentos d ON d.id = documentos_fts.rowid
        JOIN startups s ON s.id = d.startup_id
        WHERE {' AND '.join(condicoes)}
        ORDER BY score_bm25 ASC, s.nome ASC, d.id ASC
        LIMIT ?
    """
    parametros.append(limite)
    return ConsultaParametrizada(sql=sql, parametros=tuple(parametros))


class BaseStartups:
    """Fronteira determinística do SQLite; não acessa rede nem modelos."""

    def __init__(self, caminho_banco: Path):
        self.caminho_banco = caminho_banco

    def vocabularios(self) -> dict[str, list[str]]:
        with conectar(self.caminho_banco) as conexao:
            resultado: dict[str, list[str]] = {}
            for campo in ("setor", "estagio", "localizacao", "tamanho_time"):
                linhas = conexao.execute(
                    f"SELECT DISTINCT {campo} AS valor FROM startups "
                    f"WHERE {campo} IS NOT NULL ORDER BY {campo}"
                ).fetchall()
                resultado[campo] = [linha["valor"] for linha in linhas]
            classes = conexao.execute(
                "SELECT DISTINCT classe AS valor FROM analises "
                "WHERE status = 'concluida' ORDER BY classe"
            ).fetchall()
            resultado["classe_analisada"] = [linha["valor"] for linha in classes]
            return resultado

    def carregar_documentos(
        self, id_startup: int, ids_documentos: Sequence[int]
    ) -> list[DocumentoIntegral]:
        """Lê os documentos completos de uma startup, na ordem pedida.

        Falha alto para id ausente, repetido ou de outra startup: devolver menos
        documentos do que o solicitado esconderia uma recuperação inconsistente.
        """
        ids = list(ids_documentos)
        if not ids:
            raise ErroDocumentosStartup("nenhum documento foi informado para a startup")
        repetidos = sorted({item for item in ids if ids.count(item) > 1}, key=repr)
        if repetidos:
            raise ErroDocumentosStartup(f"ids de documento repetidos: {repetidos}")

        marcadores = ", ".join("?" for _ in ids)
        with conectar(self.caminho_banco) as conexao:
            linhas = conexao.execute(
                f"""
                SELECT id AS id_documento, startup_id AS id_startup,
                       tipo, titulo, conteudo_texto
                FROM documentos WHERE id IN ({marcadores})
                """,
                tuple(ids),
            ).fetchall()

        por_id = {linha["id_documento"]: linha for linha in linhas}
        ausentes = [item for item in ids if item not in por_id]
        if ausentes:
            raise ErroDocumentosStartup(f"documentos não existem na base: {ausentes}")
        invasores = [item for item in ids if por_id[item]["id_startup"] != id_startup]
        if invasores:
            raise ErroDocumentosStartup(
                f"documentos {invasores} pertencem a outra startup, não à {id_startup}"
            )
        return [DocumentoIntegral.model_validate(dict(por_id[item])) for item in ids]

    def carregar_documentos_verificaveis(
        self, ids_documentos: Sequence[object]
    ) -> dict[int, DocumentoVerificavel]:
        """Releitura tolerante dos documentos citados, indexada por id.

        Diferente de ``carregar_documentos``, um id ausente não interrompe a
        execução: some do resultado. Quem chama é o Evidence Validator, que
        precisa transformar cada falha de proveniência em uma afirmação
        derrubada com motivo, e não em uma análise abortada.
        """
        ids = list(dict.fromkeys(ids_documentos))
        if not ids:
            return {}
        marcadores = ", ".join("?" for _ in ids)
        with conectar(self.caminho_banco) as conexao:
            linhas = conexao.execute(
                f"""
                SELECT id AS id_documento, startup_id AS id_startup,
                       conteudo_texto, dominio_fonte
                FROM documentos WHERE id IN ({marcadores})
                """,
                tuple(ids),
            ).fetchall()
        return {
            linha["id_documento"]: DocumentoVerificavel.model_validate(dict(linha))
            for linha in linhas
        }

    def carregar_metadados_fit_score(
        self, ids_documentos: Sequence[object]
    ) -> dict[int, MetadadoDocumentoFitScore]:
        """Metadados de proveniência e datação dos documentos citados, por id.

        O fit-score é uma função pura: ele não abre o SQLite. Esta é a fronteira
        que entrega, de forma explícita, a URL, o host normalizado e a data de
        publicação de que a rubrica precisa. Como em
        ``carregar_documentos_verificaveis``, id ausente some do resultado — a
        completude é conferida pelo contrato ``EntradaFitScore``, que é quem
        sabe quais documentos o perfil realmente referencia.
        """
        ids = list(dict.fromkeys(ids_documentos))
        if not ids:
            return {}
        marcadores = ", ".join("?" for _ in ids)
        with conectar(self.caminho_banco) as conexao:
            linhas = conexao.execute(
                f"""
                SELECT id AS id_documento, url_fonte, dominio_fonte, data_publicacao
                FROM documentos WHERE id IN ({marcadores})
                """,
                tuple(ids),
            ).fetchall()
        return {
            linha["id_documento"]: MetadadoDocumentoFitScore(
                id_documento=linha["id_documento"],
                url_fonte=linha["url_fonte"],
                host_normalizado=normalizar_dominio(linha["dominio_fonte"]),
                data_publicacao=linha["data_publicacao"],
            )
            for linha in linhas
        }

    def carregar_site_oficial(self, id_startup: int) -> str:
        """Site oficial da startup, para o cabeçalho do Briefing.

        ``EmpresaCandidata`` é o snapshot da recuperação e não carrega o site;
        esta é a menor leitura parametrizada que o cabeçalho da §11.2 exige. A
        projeção é deliberadamente de uma coluna: ``classe_referencia`` não
        entra em nenhuma consulta de execução.
        """
        with conectar(self.caminho_banco) as conexao:
            linha = conexao.execute(
                "SELECT site FROM startups WHERE id = ?", (id_startup,)
            ).fetchone()
        if linha is None:
            raise ErroDocumentosStartup(
                f"a startup {id_startup} não existe na base"
            )
        return linha["site"]

    def carregar_fontes_briefing(
        self, id_startup: int, ids_documentos: Sequence[object]
    ) -> dict[int, FonteBriefing]:
        """Projeção pública dos documentos citados, isolada por startup.

        O ``AND startup_id = ?`` é a fronteira: um id de documento de outra
        empresa simplesmente não volta, e quem chama descobre a ausência em vez
        de exibir a fonte errada sob o nome desta startup.
        """
        ids = list(dict.fromkeys(ids_documentos))
        if not ids:
            return {}
        marcadores = ", ".join("?" for _ in ids)
        with conectar(self.caminho_banco) as conexao:
            linhas = conexao.execute(
                f"""
                SELECT id AS id_documento, tipo, titulo, url_fonte,
                       dominio_fonte, data_publicacao
                FROM documentos
                WHERE id IN ({marcadores}) AND startup_id = ?
                """,
                (*ids, id_startup),
            ).fetchall()
        return {
            linha["id_documento"]: FonteBriefing(
                url_fonte=linha["url_fonte"],
                host_normalizado=normalizar_dominio(linha["dominio_fonte"]),
                tipo=linha["tipo"],
                titulo=linha["titulo"],
                data_publicacao=linha["data_publicacao"],
            )
            for linha in linhas
        }

    def recuperar(
        self, plano: PlanoConsulta, startup_selecionada: int | None = None
    ) -> ResultadoRecuperacao:
        if startup_selecionada is not None:
            return self._recuperar_pinada(plano, startup_selecionada)
        consulta = montar_consulta_descoberta(plano)
        with conectar(self.caminho_banco) as conexao:
            linhas = conexao.execute(consulta.sql, consulta.parametros).fetchall()
        return self._montar_resultado(linhas, plano.filtros)

    def _recuperar_pinada(
        self, plano: PlanoConsulta, startup_id: int
    ) -> ResultadoRecuperacao:
        match = construir_match_fts([*plano.termos_busca, *plano.sinais_ia])
        with conectar(self.caminho_banco) as conexao:
            empresa = conexao.execute(
                """
                SELECT id AS id_startup, nome, setor, estagio, localizacao, descricao_curta
                FROM startups WHERE id = ?
                """,
                (startup_id,),
            ).fetchone()
            if empresa is None:
                return ResultadoRecuperacao(
                    empresas=[], documentos=[], filtros_aplicados=plano.filtros
                )
            pontuacoes = {
                linha["id_documento"]: float(linha["score_bm25"])
                for linha in conexao.execute(
                    """
                    SELECT d.id AS id_documento, bm25(documentos_fts) AS score_bm25
                    FROM documentos_fts
                    JOIN documentos d ON d.id = documentos_fts.rowid
                    WHERE documentos_fts MATCH ? AND d.startup_id = ?
                    ORDER BY score_bm25 ASC
                    """,
                    (match, startup_id),
                ).fetchall()
            }
            documentos = conexao.execute(
                """
                SELECT id AS id_documento, startup_id AS id_startup, tipo, titulo,
                       url_fonte, dominio_fonte, data_acesso
                FROM documentos WHERE startup_id = ? ORDER BY id
                """,
                (startup_id,),
            ).fetchall()
        ordenados = sorted(
            documentos,
            key=lambda doc: (
                doc["id_documento"] not in pontuacoes,
                pontuacoes.get(doc["id_documento"], 0.0),
                doc["id_documento"],
            ),
        )
        return ResultadoRecuperacao(
            empresas=[EmpresaCandidata.model_validate(dict(empresa))],
            documentos=[
                DocumentoRecuperado(
                    **dict(documento),
                    score_bm25=pontuacoes.get(documento["id_documento"], 0.0),
                )
                for documento in ordenados
            ],
            filtros_aplicados=plano.filtros,
        )

    @staticmethod
    def _montar_resultado(
        linhas: list[sqlite3.Row], filtros: FiltrosEstruturados
    ) -> ResultadoRecuperacao:
        empresas: dict[int, EmpresaCandidata] = {}
        documentos: list[DocumentoRecuperado] = []
        for linha in linhas:
            id_startup = linha["id_startup"]
            if id_startup not in empresas:
                empresas[id_startup] = EmpresaCandidata(
                    id_startup=id_startup,
                    nome=linha["nome"],
                    setor=linha["setor"],
                    estagio=linha["estagio"],
                    localizacao=linha["localizacao"],
                    descricao_curta=linha["descricao_curta"],
                )
            documentos.append(
                DocumentoRecuperado(
                    id_documento=linha["id_documento"],
                    id_startup=id_startup,
                    tipo=linha["tipo"],
                    titulo=linha["titulo"],
                    url_fonte=linha["url_fonte"],
                    dominio_fonte=linha["dominio_fonte"],
                    data_acesso=linha["data_acesso"],
                    score_bm25=float(linha["score_bm25"]),
                )
            )
        return ResultadoRecuperacao(
            empresas=list(empresas.values()),
            documentos=documentos,
            filtros_aplicados=filtros,
        )

