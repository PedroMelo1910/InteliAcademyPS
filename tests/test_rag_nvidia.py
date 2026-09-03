"""Nó NVIDIA RAG: adapta a recuperação já existente para depois do R3.

O nó não reimplementa busca híbrida, fusão, reranking nem citação: ele monta a
consulta a partir do estado aprovado, delega ao consultor injetado e devolve um
``ContextoNvidia`` validado. Todos os testes são offline.
"""

from __future__ import annotations

import pytest

from radar.agentes.rag_nvidia import (
    CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA,
    ErroRagNvidia,
    NvidiaRag,
    montar_consulta_nvidia,
)
from radar.conhecimento_nvidia.consulta import ErroConsultaNvidia
from radar.contratos import ContextoNvidia, EmpresaCandidata, PerfilValidado
from tests.conftest import (
    ConsultorNvidiaFalso,
    afirmacao_validada_falsa,
    contexto_nvidia_falso,
    perfil_validado_falso,
)


def empresa_falsa(id_startup: int = 7) -> EmpresaCandidata:
    return EmpresaCandidata(
        id_startup=id_startup,
        nome="Acme IA",
        setor="Saúde",
        estagio="Seed",
        localizacao="São Paulo",
        descricao_curta="Plataforma de triagem clínica assistida por modelos.",
    )


def perfil_com_gap_e_dor():
    return perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "otimizacao_tecnica", polaridade="ausencia_explicita"
            ),
            afirmacao_validada_falsa(2, "dependencia_api_externa"),
            afirmacao_validada_falsa(3, "outro"),
        ],
        hosts=["fonte-a.example", "fonte-b.example"],
    )


def estado_pos_r3(perfil=None, empresa=None, **ajustes):
    empresa = empresa or empresa_falsa()
    estado = {
        "startup_selecionada": empresa.id_startup,
        "perfil_validado": perfil if perfil is not None else perfil_com_gap_e_dor(),
        "resultado_recuperacao": {
            "empresas": [empresa.model_dump()],
            "documentos": [],
            "filtros_aplicados": {},
        },
        "contexto_nvidia": None,
        "recomendacoes": None,
        "fit_score": None,
        "erros": [],
        "trajeto": ["extractor", "classifier", "evidence_validator"],
    }
    estado.update(ajustes)
    return estado


# ----------------------------------------------------------------------
# Consulta gerada
# ----------------------------------------------------------------------


def test_consulta_e_deterministica_para_a_mesma_entrada():
    perfil, empresa = perfil_com_gap_e_dor(), empresa_falsa()
    assert montar_consulta_nvidia(perfil, empresa) == montar_consulta_nvidia(
        perfil, empresa
    )


def test_consulta_usa_setor_gap_confirmado_e_dor_documentada():
    consulta = montar_consulta_nvidia(perfil_com_gap_e_dor(), empresa_falsa())
    assert "Saúde" in consulta
    assert "otimizacao tecnica" in consulta
    assert "dependencia api externa" in consulta


def test_consulta_ignora_dimensao_de_capacidade_confirmada():
    perfil = perfil_validado_falso(
        [afirmacao_validada_falsa(1, "distribuicao", polaridade="presenca")]
    )
    consulta = montar_consulta_nvidia(perfil, empresa_falsa())
    assert "gap" in consulta.casefold()
    assert "distribuicao" not in consulta


def test_consulta_ignora_afirmacao_derrubada():
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1,
                "escala_e_dor_operacional",
                situacao="derrubada",
                texto="A empresa relata fila de inferência insustentável.",
            ),
            afirmacao_validada_falsa(2, "outro"),
        ]
    )
    consulta = montar_consulta_nvidia(perfil, empresa_falsa())
    assert "escala e dor operacional" not in consulta
    assert "insustentável" not in consulta


def test_montagem_da_consulta_so_recebe_perfil_validado_e_empresa():
    """Estrutural: não há parâmetro por onde ``classe_referencia`` entraria."""
    import inspect

    parametros = list(inspect.signature(montar_consulta_nvidia).parameters)
    assert parametros == ["perfil", "empresa"]


def test_consulta_nao_vaza_classe_de_referencia_da_curadoria():
    consulta = montar_consulta_nvidia(perfil_com_gap_e_dor(), empresa_falsa())
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        assert classe not in consulta


# ----------------------------------------------------------------------
# Execução do nó
# ----------------------------------------------------------------------


def test_no_delega_ao_consultor_injetado_e_grava_contexto_validado():
    consultor = ConsultorNvidiaFalso()
    saida = NvidiaRag(consultor)(estado_pos_r3())

    assert consultor.chamadas == 1
    contexto = ContextoNvidia.model_validate(saida["contexto_nvidia"])
    assert len(contexto.trechos) == 6


def test_no_repassa_exatamente_a_consulta_que_montou():
    consultor = ConsultorNvidiaFalso()
    estado = estado_pos_r3()
    NvidiaRag(consultor)(estado)
    esperada = montar_consulta_nvidia(
        PerfilValidado.model_validate(estado["perfil_validado"]), empresa_falsa()
    )
    assert consultor.consultas == [esperada]


def test_no_registra_exatamente_um_item_de_trajeto():
    saida = NvidiaRag(ConsultorNvidiaFalso())(estado_pos_r3())
    assert saida["trajeto"] == ["nvidia_rag"]


def test_no_invalida_os_campos_derivados_do_contexto():
    saida = NvidiaRag(ConsultorNvidiaFalso())(
        estado_pos_r3(recomendacoes=["recomendação velha"], fit_score={"total": 99})
    )
    assert CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA == (
        "recomendacoes",
        "fit_score",
        "briefing",
    )
    for campo in CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA:
        assert saida[campo] is None


def test_no_nao_escreve_campos_que_nao_lhe_pertencem():
    saida = NvidiaRag(ConsultorNvidiaFalso())(estado_pos_r3())
    assert set(saida) == {
        "contexto_nvidia",
        "recomendacoes",
        "fit_score",
        "briefing",
        "trajeto",
    }


# ----------------------------------------------------------------------
# Falha segura
# ----------------------------------------------------------------------


def test_falha_da_recuperacao_nao_produz_contexto():
    consultor = ConsultorNvidiaFalso(
        erro=ErroConsultaNvidia("índice NVIDIA ainda não foi ingerido")
    )
    with pytest.raises(ErroRagNvidia, match="recuperação NVIDIA"):
        NvidiaRag(consultor)(estado_pos_r3())


def test_falha_operacional_do_provedor_nao_vira_passagem_fabricada():
    consultor = ConsultorNvidiaFalso(erro=RuntimeError("endpoint indisponível"))
    with pytest.raises(ErroRagNvidia) as capturado:
        NvidiaRag(consultor)(estado_pos_r3())
    assert "fabricad" in str(capturado.value)


def test_contexto_fora_do_contrato_e_recusado():
    class ConsultorForaDoContrato:
        def consultar(self, consulta):
            return {"consulta_gerada": consulta, "trechos": []}

    with pytest.raises(ErroRagNvidia, match="contrato"):
        NvidiaRag(ConsultorForaDoContrato())(estado_pos_r3())


def test_no_exige_perfil_validado_no_estado():
    with pytest.raises(ErroRagNvidia, match="PerfilValidado"):
        NvidiaRag(ConsultorNvidiaFalso())(estado_pos_r3(perfil_validado=None))


def test_no_exige_resultado_recuperacao_no_estado():
    with pytest.raises(ErroRagNvidia, match="ResultadoRecuperacao"):
        NvidiaRag(ConsultorNvidiaFalso())(estado_pos_r3(resultado_recuperacao=None))


def test_no_recusa_startup_ausente_do_conjunto_recuperado():
    estado = estado_pos_r3(startup_selecionada=999)
    with pytest.raises(ErroRagNvidia, match="startup"):
        NvidiaRag(ConsultorNvidiaFalso())(estado)


def test_contexto_de_outra_consulta_nao_e_reaproveitado_em_silencio():
    """Cada execução consulta de novo; nada é herdado do checkpoint."""
    consultor = ConsultorNvidiaFalso()
    estado = estado_pos_r3(contexto_nvidia=contexto_nvidia_falso("consulta antiga"))
    saida = NvidiaRag(consultor)(estado)
    assert consultor.chamadas == 1
    contexto = ContextoNvidia.model_validate(saida["contexto_nvidia"])
    assert contexto.consulta_gerada != "consulta antiga"


def test_contexto_gravado_no_estado_atravessa_o_checkpoint(tmp_path):
    """O msgpack do checkpointer não serializa ``AnyHttpUrl``; a forma JSON sim."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    saida = NvidiaRag(ConsultorNvidiaFalso())(estado_pos_r3())
    JsonPlusSerializer().dumps_typed(saida["contexto_nvidia"])
    assert ContextoNvidia.model_validate(saida["contexto_nvidia"]).trechos
