from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from radar.base_startups import BaseStartups
from radar.contratos import (
    DIMENSOES_GAP,
    AfirmacaoValidada,
    Classificacao,
    ConfiancaPerfil,
    DocumentoVerificavel,
    EstadoDimensaoGap,
    EstadoRadar,
    PerfilExtraido,
    PerfilValidado,
    ResultadoRecuperacao,
    normalizar_dominio,
    normalizar_texto_citavel,
)


MINIMO_HOSTS_CONFIANCA_NORMAL = 2

# A recuperação NVIDIA e a recomendação descendem do perfil validado. Quando
# este nó reconfere a evidência, o que foi derivado da validação anterior
# perde o lastro — inclusive num thread retomado, cujo checkpoint traria a
# recomendação de outra passagem pelo laço de reextração.
CAMPOS_DERIVADOS_DA_VALIDACAO: tuple[str, ...] = (
    "contexto_nvidia",
    "recomendacoes",
    "fit_score",
)


class ErroValidadorEvidencias(RuntimeError):
    """Falha segura: sem perfil, classificação ou recuperação não há o que conferir."""


class EvidenceValidator:
    """Confere a proveniência de cada afirmação contra o documento citado.

    O nó é inteiramente determinístico: não tem provedor de LLM nem acesso à
    rede. Ele relê os documentos referenciados pela fronteira do SQLite em vez
    de confiar em texto trafegado pelo estado, e nunca lê ``classe_referencia``.
    """

    def __init__(self, base: BaseStartups):
        self.base = base

    def __call__(self, estado: EstadoRadar) -> dict[str, Any]:
        perfil = self._perfil(estado)
        recuperacao = self._recuperacao(estado)
        self._conferir_isolamento_da_startup(estado, perfil, recuperacao)
        # A classificação é lida para garantir que seus ids de suporte ainda
        # existam no perfil conferido; a decisão sobre eles é do roteador.
        self._conferir_suporte(self._classificacao(estado), perfil)

        ids_recuperados = {
            documento.id_documento
            for documento in recuperacao.documentos
            if documento.id_startup == perfil.id_startup
        }
        ids_referenciados = {
            afirmacao.id_documento for afirmacao in perfil.afirmacoes
        }
        documentos = self.base.carregar_documentos_verificaveis(
            sorted(ids_referenciados & ids_recuperados)
        )

        validadas = [
            self._julgar(afirmacao, perfil.id_startup, documentos, ids_recuperados)
            for afirmacao in perfil.afirmacoes
        ]
        confirmadas = [item for item in validadas if item.situacao == "confirmada"]

        tentativa = int(estado.get("tentativas_extracao", 0))
        # ``tentativas_extracao`` é local: o Retriever o reinicia a cada novo
        # contexto recuperado. O número da validação vem do histórico já
        # acumulado em ``trajeto`` e é monotônico dentro do thread, sem campo
        # novo, sem relógio e sem identificador aleatório.
        validacao = list(estado.get("trajeto", [])).count("evidence_validator") + 1
        dimensoes, avisos = self._dimensoes_de_gap(confirmadas, tentativa, validacao)
        hosts = self._hosts_confirmados(confirmadas, documentos)
        taxa = (len(validadas) - len(confirmadas)) / len(validadas)

        try:
            perfil_validado = PerfilValidado(
                afirmacoes_validadas=validadas,
                taxa_derrubada=taxa,
                hosts_distintos=hosts,
                estado_dimensoes_gap=dimensoes,
            )
        except (ValidationError, ValueError, TypeError) as erro:
            # Falha segura: nenhum perfil parcial sai daqui, e a violação de
            # contrato vira o erro de domínio do nó em vez de vazar crua.
            raise ErroValidadorEvidencias(
                "o perfil validado montado violou o próprio contrato; "
                f"nenhum resultado foi gravado no estado: {erro}"
            ) from erro
        saida: dict[str, Any] = {
            "perfil_validado": perfil_validado,
            "confianca_perfil": self._confianca(confirmadas, taxa, hosts),
            "trajeto": ["evidence_validator"],
        }
        for campo in CAMPOS_DERIVADOS_DA_VALIDACAO:
            saida[campo] = None
        # Os canais acumulados recebem apenas o que este nó produziu; devolver a
        # lista inteira duplicaria o histórico ao passar pelo reducer.
        if avisos:
            saida["erros"] = avisos
        return saida

    # ------------------------------------------------------------------
    # Pré-condições do nó
    # ------------------------------------------------------------------

    @staticmethod
    def _perfil(estado: EstadoRadar) -> PerfilExtraido:
        return _validar_entrada(
            estado.get("perfil_extraido"),
            PerfilExtraido,
            "o Evidence Validator exige um PerfilExtraido no estado",
        )

    @staticmethod
    def _classificacao(estado: EstadoRadar) -> Classificacao:
        return _validar_entrada(
            estado.get("classificacao"),
            Classificacao,
            "o Evidence Validator exige uma Classificacao no estado",
        )

    @staticmethod
    def _recuperacao(estado: EstadoRadar) -> ResultadoRecuperacao:
        return _validar_entrada(
            estado.get("resultado_recuperacao"),
            ResultadoRecuperacao,
            "o Evidence Validator exige um ResultadoRecuperacao no estado",
        )

    @staticmethod
    def _conferir_suporte(
        classificacao: Classificacao, perfil: PerfilExtraido
    ) -> None:
        disponiveis = {afirmacao.id_afirmacao for afirmacao in perfil.afirmacoes}
        estranhos = sorted(set(classificacao.ids_afirmacoes_suporte) - disponiveis)
        if estranhos:
            raise ErroValidadorEvidencias(
                "a classificação no estado cita afirmações ausentes do perfil "
                f"conferido: {estranhos}"
            )

    @staticmethod
    def _conferir_isolamento_da_startup(
        estado: EstadoRadar,
        perfil: PerfilExtraido,
        recuperacao: ResultadoRecuperacao,
    ) -> None:
        selecionada = estado.get("startup_selecionada")
        if selecionada is not None and int(selecionada) != perfil.id_startup:
            raise ErroValidadorEvidencias(
                "o PerfilExtraido pertence a uma startup diferente da selecionada"
            )
        ids_recuperados = {empresa.id_startup for empresa in recuperacao.empresas}
        if perfil.id_startup not in ids_recuperados:
            raise ErroValidadorEvidencias(
                "a startup do PerfilExtraido não pertence ao resultado recuperado"
            )

    # ------------------------------------------------------------------
    # Veredito por afirmação
    # ------------------------------------------------------------------

    @staticmethod
    def _julgar(
        afirmacao,
        id_startup: int,
        documentos: dict[int, DocumentoVerificavel],
        ids_recuperados: set[int],
    ) -> AfirmacaoValidada:
        campos = afirmacao.model_dump()
        motivo = EvidenceValidator._motivo_de_derrubada(
            afirmacao, id_startup, documentos, ids_recuperados
        )
        if motivo is None:
            return AfirmacaoValidada(**campos, situacao="confirmada", motivo=None)
        return AfirmacaoValidada(**campos, situacao="derrubada", motivo=motivo)

    @staticmethod
    def _motivo_de_derrubada(
        afirmacao,
        id_startup: int,
        documentos: dict[int, DocumentoVerificavel],
        ids_recuperados: set[int],
    ) -> str | None:
        """A ordem das conferências é o contrato: da falha mais grave à mais sutil."""
        id_documento = afirmacao.id_documento
        if id_documento not in ids_recuperados:
            return (
                f"o documento {id_documento} não estava no conjunto recuperado "
                "desta análise"
            )
        documento = documentos.get(id_documento)
        if documento is None:
            return f"o documento {id_documento} não existe na base"
        if documento.id_startup != id_startup:
            return f"o documento {id_documento} pertence a outra startup"
        if normalizar_texto_citavel(
            afirmacao.trecho_citado
        ) not in normalizar_texto_citavel(documento.conteudo_texto):
            return (
                f"o trecho citado não ocorre literalmente no documento {id_documento}"
            )
        return None

    # ------------------------------------------------------------------
    # Agregados
    # ------------------------------------------------------------------

    @staticmethod
    def _hosts_confirmados(
        confirmadas: list[AfirmacaoValidada],
        documentos: dict[int, DocumentoVerificavel],
    ) -> list[str]:
        return sorted(
            {
                normalizar_dominio(documentos[afirmacao.id_documento].dominio_fonte)
                for afirmacao in confirmadas
            }
        )

    @staticmethod
    def _dimensoes_de_gap(
        confirmadas: list[AfirmacaoValidada],
        tentativa: int = 0,
        validacao: int = 1,
    ) -> tuple[list[EstadoDimensaoGap], list[str]]:
        """Só evidência confirmada decide; silêncio permanece desconhecido."""
        dimensoes: list[EstadoDimensaoGap] = []
        avisos: list[str] = []
        for dimensao in DIMENSOES_GAP:
            presencas = [
                afirmacao.id_afirmacao
                for afirmacao in confirmadas
                if afirmacao.categoria == dimensao and afirmacao.polaridade == "presenca"
            ]
            ausencias = [
                afirmacao.id_afirmacao
                for afirmacao in confirmadas
                if afirmacao.categoria == dimensao
                and afirmacao.polaridade == "ausencia_explicita"
            ]
            if presencas and ausencias:
                # Evidência confirmada em sentidos opostos não vira certeza: os
                # dois lados ficam registrados e a dimensão continua desconhecida.
                estado, ids = "desconhecido", sorted(presencas + ausencias)
                # O canal ``erros`` é acumulado e não pode ser reescrito: o aviso
                # se identifica como aviso e carrega a extração que o produziu,
                # para que uma reextração bem-sucedida não pareça manter conflito.
                avisos.append(
                    f"aviso [validação {validacao}; extração {tentativa}]: "
                    f"evidência confirmada em conflito na dimensão {dimensao}: "
                    f"presença {presencas} e ausência explícita {ausencias}"
                )
            elif presencas:
                estado, ids = "capacidade_confirmada", sorted(presencas)
            elif ausencias:
                estado, ids = "gap_confirmado", sorted(ausencias)
            else:
                estado, ids = "desconhecido", []
            dimensoes.append(
                EstadoDimensaoGap(dimensao=dimensao, estado=estado, ids_evidencias=ids)
            )
        return dimensoes, avisos

    @staticmethod
    def _confianca(
        confirmadas: list[AfirmacaoValidada], taxa_derrubada: float, hosts: list[str]
    ) -> ConfiancaPerfil:
        """Fórmula fechada: conflito de evidência não a afrouxa nem a endurece."""
        if not confirmadas:
            return "baixa"
        if taxa_derrubada > 0:
            return "baixa"
        if len(hosts) < MINIMO_HOSTS_CONFIANCA_NORMAL:
            return "baixa"
        return "normal"


def _validar_entrada(bruto: object, contrato, mensagem: str):
    if bruto is None:
        raise ErroValidadorEvidencias(mensagem)
    try:
        return contrato.model_validate(bruto)
    except (ValidationError, ValueError, TypeError) as erro:
        raise ErroValidadorEvidencias(f"{mensagem}; recebido fora do contrato: {erro}") from erro
