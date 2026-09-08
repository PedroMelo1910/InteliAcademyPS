"""Nó NVIDIA RAG: adapta a recuperação já existente para depois do R3.

O nó não reimplementa busca híbrida, fusão, reranking nem citação: ele monta a
consulta a partir do estado aprovado, delega ao consultor injetado e devolve um
``ContextoNvidia`` validado. Todos os testes são offline.

A segunda parte detalha a montagem da consulta a partir dos sinais técnicos: a
oportunidade que autorizou a recomendação precisa guiar a busca, em ordem
canônica, sob o teto de evidências, e as fronteiras que continuam proibidas.
"""

from __future__ import annotations

import inspect

import pytest

from radar.agentes.rag_nvidia import (
    CAMPOS_DERIVADOS_DO_CONTEXTO_NVIDIA,
    MAXIMO_EVIDENCIAS_NA_CONSULTA,
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
    assert len(contexto.trechos) == 8


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


# ------------------------------------------------------------------------
# Montagem da consulta a partir dos sinais técnicos
# ------------------------------------------------------------------------

def empresa(setor: str = "Saúde") -> EmpresaCandidata:
    return EmpresaCandidata(
        id_startup=1,
        nome="Sintética",
        setor=setor,
        estagio="série A",
        localizacao="Campinas, SP",
        descricao_curta="Descrição curta.",
    )


def perfil(*itens):
    return perfil_validado_falso(list(itens))


def test_um_perfil_so_com_sinal_leva_o_sinal_e_a_evidencia_para_a_consulta():
    item = afirmacao_validada_falsa(
        1, "stack_propria", sinais_tecnicos=["visao_computacional"]
    )

    consulta = montar_consulta_nvidia(perfil(item), empresa())

    assert "visao computacional" in consulta
    assert item.texto in consulta


def test_consulta_nomeia_tecnologias_candidatas_do_fundamento_confirmado():
    item = afirmacao_validada_falsa(
        1, "stack_propria", sinais_tecnicos=["inferencia_llm"]
    )

    consulta = montar_consulta_nvidia(perfil(item), empresa())

    assert "tecnologias NVIDIA candidatas pela regra" in consulta
    assert "NVIDIA NIM" in consulta
    assert "NVIDIA Triton Inference Server" in consulta
    assert "TensorRT-LLM" in consulta


def test_consulta_sem_fundamento_nao_inventa_tecnologia_candidata():
    consulta = montar_consulta_nvidia(
        perfil(afirmacao_validada_falsa(1, "outro")), empresa()
    )

    assert "tecnologias NVIDIA candidatas pela regra" not in consulta


def test_varios_sinais_saem_em_ordem_canonica_deterministica():
    itens = (
        afirmacao_validada_falsa(1, "outro", sinais_tecnicos=["visao_computacional"]),
        afirmacao_validada_falsa(2, "stack_propria", sinais_tecnicos=["inferencia_llm"]),
    )

    consulta = montar_consulta_nvidia(perfil(*itens), empresa())

    assert consulta.index("inferencia llm") < consulta.index("visao computacional")


def test_um_sinal_de_afirmacao_derrubada_nao_entra_na_consulta():
    derrubada = afirmacao_validada_falsa(
        1,
        "stack_propria",
        situacao="derrubada",
        sinais_tecnicos=["robotica_ou_simulacao"],
    )

    consulta = montar_consulta_nvidia(perfil(derrubada), empresa())

    assert "robotica" not in consulta
    assert derrubada.texto not in consulta


def test_o_setor_sozinho_nao_cria_sinal_nenhum():
    item = afirmacao_validada_falsa(1, "outro")

    consulta = montar_consulta_nvidia(perfil(item), empresa(setor="Saúde"))

    assert "Saúde" in consulta
    assert "imagem medica" not in consulta
    assert "oportunidades confirmadas: nenhuma" in consulta


def test_uma_ausencia_declarada_nunca_vira_oportunidade():
    ausencia = afirmacao_validada_falsa(
        1, "distribuicao", polaridade="ausencia_explicita"
    )

    consulta = montar_consulta_nvidia(perfil(ausencia), empresa())

    assert "oportunidades confirmadas: nenhuma" in consulta
    assert "gaps confirmados: distribuicao" in consulta


def test_gaps_e_dores_continuam_na_consulta_como_antes():
    itens = (
        afirmacao_validada_falsa(1, "distribuicao", polaridade="ausencia_explicita"),
        afirmacao_validada_falsa(2, "dependencia_api_externa"),
    )

    consulta = montar_consulta_nvidia(perfil(*itens), empresa())

    assert "gaps confirmados: distribuicao" in consulta
    assert "dependencia api externa" in consulta


def test_a_consulta_respeita_o_teto_deterministico_de_evidencias():
    itens = tuple(
        afirmacao_validada_falsa(
            indice, "stack_propria", sinais_tecnicos=["inferencia_llm"]
        )
        for indice in range(1, MAXIMO_EVIDENCIAS_NA_CONSULTA + 4)
    )

    consulta = montar_consulta_nvidia(perfil(*itens), empresa())

    presentes = sum(1 for item in itens if item.texto in consulta)
    assert presentes == MAXIMO_EVIDENCIAS_NA_CONSULTA


def test_o_rotulo_de_curadoria_e_estruturalmente_inalcancavel():
    assinatura = inspect.signature(montar_consulta_nvidia)

    assert list(assinatura.parameters) == ["perfil", "empresa"]
    # nenhum dos dois contratos de entrada expõe o rótulo de curadoria
    assert "classe_referencia" not in EmpresaCandidata.model_fields
    from radar.contratos import AfirmacaoValidada

    assert "classe_referencia" not in AfirmacaoValidada.model_fields
