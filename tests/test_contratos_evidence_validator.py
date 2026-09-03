import pytest
from pydantic import ValidationError

from radar.contratos import (
    DIMENSOES_GAP,
    AfirmacaoValidada,
    EstadoDimensaoGap,
    EstadoRadar,
    PerfilValidado,
    normalizar_dominio,
)


# --------------------------------------------------------------------------
# AfirmacaoValidada: a Afirmacao original mais o veredito de proveniência.
# --------------------------------------------------------------------------


def afirmacao(**ajustes) -> dict:
    base = {
        "id_afirmacao": 1,
        "texto": "A Acme mantém um banco proprietário de imagens rotuladas.",
        "categoria": "dados_proprietarios",
        "polaridade": "presenca",
        "id_documento": 101,
        "trecho_citado": "banco proprietário de imagens rotuladas",
        "situacao": "confirmada",
        "motivo": None,
    }
    base.update(ajustes)
    return base


def test_afirmacao_validada_preserva_todos_os_campos_da_afirmacao():
    modelo = AfirmacaoValidada.model_validate(afirmacao())
    assert modelo.id_afirmacao == 1
    assert modelo.texto.startswith("A Acme mantém")
    assert modelo.categoria == "dados_proprietarios"
    assert modelo.polaridade == "presenca"
    assert modelo.id_documento == 101
    assert modelo.trecho_citado == "banco proprietário de imagens rotuladas"


def test_situacao_fora_do_enum_e_rejeitada():
    with pytest.raises(ValidationError):
        AfirmacaoValidada.model_validate(afirmacao(situacao="parcial"))


def test_confirmada_com_motivo_e_rejeitada():
    with pytest.raises(ValidationError, match="motivo"):
        AfirmacaoValidada.model_validate(
            afirmacao(situacao="confirmada", motivo="o trecho ocorre no documento")
        )


def test_derrubada_sem_motivo_e_rejeitada():
    with pytest.raises(ValidationError, match="motivo"):
        AfirmacaoValidada.model_validate(afirmacao(situacao="derrubada", motivo=None))


def test_derrubada_com_motivo_em_branco_e_rejeitada():
    with pytest.raises(ValidationError, match="motivo"):
        AfirmacaoValidada.model_validate(afirmacao(situacao="derrubada", motivo="   "))


def test_derrubada_com_motivo_conciso_e_aceita():
    modelo = AfirmacaoValidada.model_validate(
        afirmacao(situacao="derrubada", motivo="o documento 101 não existe na base")
    )
    assert modelo.situacao == "derrubada"
    assert modelo.motivo == "o documento 101 não existe na base"


def test_afirmacao_validada_rejeita_campos_desconhecidos():
    with pytest.raises(ValidationError):
        AfirmacaoValidada.model_validate(afirmacao(classe_referencia="AI-native"))


def test_afirmacao_validada_mantem_as_regras_da_afirmacao_original():
    with pytest.raises(ValidationError):
        AfirmacaoValidada.model_validate(
            afirmacao(categoria="momento_e_financiamento", polaridade="presenca")
        )


# --------------------------------------------------------------------------
# EstadoDimensaoGap: uma das quatro dimensões estruturais e seu veredito.
# --------------------------------------------------------------------------


def test_as_quatro_dimensoes_estruturais_sao_aceitas():
    for dimensao in DIMENSOES_GAP:
        modelo = EstadoDimensaoGap(dimensao=dimensao, estado="desconhecido")
        assert modelo.dimensao == dimensao


def test_ordem_das_dimensoes_e_fixa_e_deterministica():
    assert DIMENSOES_GAP == (
        "dados_proprietarios",
        "workflow_profundo",
        "distribuicao",
        "otimizacao_tecnica",
    )


def test_dimensao_fora_das_quatro_estruturais_e_rejeitada():
    with pytest.raises(ValidationError):
        EstadoDimensaoGap(dimensao="momento_e_financiamento", estado="desconhecido")


def test_os_tres_estados_de_gap_sao_aceitos():
    casos = {
        "capacidade_confirmada": [1],
        "gap_confirmado": [2],
        "desconhecido": [],
    }
    for estado, ids in casos.items():
        modelo = EstadoDimensaoGap(
            dimensao="distribuicao", estado=estado, ids_evidencias=ids
        )
        assert modelo.estado == estado
        assert modelo.ids_evidencias == ids


def test_estado_de_gap_fora_do_enum_e_rejeitado():
    with pytest.raises(ValidationError):
        EstadoDimensaoGap(dimensao="distribuicao", estado="provavel")


def test_ids_de_evidencia_duplicados_sao_rejeitados():
    with pytest.raises(ValidationError, match="duplicad"):
        EstadoDimensaoGap(
            dimensao="distribuicao", estado="capacidade_confirmada", ids_evidencias=[2, 2]
        )


def test_id_de_evidencia_menor_que_um_e_rejeitado():
    with pytest.raises(ValidationError):
        EstadoDimensaoGap(
            dimensao="distribuicao", estado="capacidade_confirmada", ids_evidencias=[0]
        )


def test_dimensao_sem_evidencia_decisiva_aceita_lista_vazia():
    modelo = EstadoDimensaoGap(dimensao="distribuicao", estado="desconhecido")
    assert modelo.ids_evidencias == []


def test_estado_dimensao_gap_rejeita_campos_desconhecidos():
    with pytest.raises(ValidationError):
        EstadoDimensaoGap(
            dimensao="distribuicao", estado="desconhecido", classe_referencia="non-AI"
        )


# --------------------------------------------------------------------------
# PerfilValidado: o perfil depois da conferência de proveniência.
# --------------------------------------------------------------------------


def dimensoes(**ajustes) -> list[dict]:
    """Por padrão, coerente com ``perfil_validado()``: a afirmação 1 confirma."""
    estados: dict[str, tuple[str, list[int]]] = {
        "dados_proprietarios": ("capacidade_confirmada", [1]),
        "workflow_profundo": ("desconhecido", []),
        "distribuicao": ("desconhecido", []),
        "otimizacao_tecnica": ("desconhecido", []),
    }
    estados.update(ajustes)
    return [
        {
            "dimensao": dimensao,
            "estado": estados[dimensao][0],
            "ids_evidencias": estados[dimensao][1],
        }
        for dimensao in DIMENSOES_GAP
    ]


def derrubada(id_afirmacao: int, **ajustes) -> dict:
    return afirmacao(
        id_afirmacao=id_afirmacao,
        situacao="derrubada",
        motivo="o trecho citado não ocorre literalmente no documento 101",
        **ajustes,
    )


def perfil_validado(**ajustes) -> dict:
    base = {
        "afirmacoes_validadas": [afirmacao()],
        "taxa_derrubada": 0.0,
        "hosts_distintos": ["acme.example.com", "jornal.example.net"],
        "estado_dimensoes_gap": dimensoes(),
    }
    base.update(ajustes)
    return base


def test_perfil_validado_aceita_o_caso_completo():
    modelo = PerfilValidado.model_validate(perfil_validado())
    assert len(modelo.afirmacoes_validadas) == 1
    assert modelo.taxa_derrubada == 0.0
    assert modelo.hosts_distintos == ["acme.example.com", "jornal.example.net"]
    assert [item.dimensao for item in modelo.estado_dimensoes_gap] == list(DIMENSOES_GAP)


def test_taxa_de_derrubada_fora_de_zero_a_um_e_rejeitada():
    for invalida in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            PerfilValidado.model_validate(perfil_validado(taxa_derrubada=invalida))


def test_taxa_de_derrubada_nos_extremos_e_aceita():
    assert PerfilValidado.model_validate(perfil_validado(taxa_derrubada=0.0))
    assert PerfilValidado.model_validate(
        perfil_validado(
            afirmacoes_validadas=[derrubada(1)],
            taxa_derrubada=1.0,
            hosts_distintos=[],
            estado_dimensoes_gap=dimensoes(dados_proprietarios=("desconhecido", [])),
        )
    )


def test_dimensao_faltante_e_rejeitada():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(estado_dimensoes_gap=dimensoes()[:3])
        )


def test_dimensao_repetida_e_rejeitada():
    repetidas = dimensoes()
    repetidas[1] = dict(repetidas[0])
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(perfil_validado(estado_dimensoes_gap=repetidas))


def test_ordem_das_dimensoes_trocada_e_rejeitada():
    trocadas = list(reversed(dimensoes()))
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(perfil_validado(estado_dimensoes_gap=trocadas))


def test_hosts_duplicados_sao_rejeitados():
    with pytest.raises(ValidationError, match="host"):
        PerfilValidado.model_validate(
            perfil_validado(hosts_distintos=["acme.example.com", "acme.example.com"])
        )


def test_host_nao_normalizado_e_rejeitado():
    for host in ("www.acme.example.com", "ACME.example.com", "  acme.example.com  "):
        with pytest.raises(ValidationError, match="host"):
            PerfilValidado.model_validate(perfil_validado(hosts_distintos=[host]))


def test_hosts_fora_de_ordem_sao_rejeitados():
    with pytest.raises(ValidationError, match="host"):
        PerfilValidado.model_validate(
            perfil_validado(hosts_distintos=["jornal.example.net", "acme.example.com"])
        )


def test_perfil_sem_host_confirmado_aceita_lista_vazia():
    modelo = PerfilValidado.model_validate(perfil_validado(hosts_distintos=[]))
    assert modelo.hosts_distintos == []


def test_perfil_validado_rejeita_campos_desconhecidos():
    with pytest.raises(ValidationError):
        PerfilValidado.model_validate(perfil_validado(confianca_perfil="normal"))


def test_perfil_validado_nao_carrega_classe_referencia():
    with pytest.raises(ValidationError):
        PerfilValidado.model_validate(perfil_validado(classe_referencia="AI-native"))


# --------------------------------------------------------------------------
# EstadoRadar
# --------------------------------------------------------------------------


def test_estado_radar_declara_o_perfil_validado():
    assert "perfil_validado" in EstadoRadar.__annotations__


def test_estado_radar_declara_a_confianca_do_perfil():
    assert "confianca_perfil" in EstadoRadar.__annotations__


def test_estado_radar_declara_fronteira_do_proximo_marco_sem_briefing():
    for campo in ("contexto_nvidia", "recomendacoes", "fit_score"):
        assert campo in EstadoRadar.__annotations__
    for campo in ("recomendacao", "briefing"):
        assert campo not in EstadoRadar.__annotations__


# --------------------------------------------------------------------------
# C2 — O perfil validado não pode se autocontradizer
# --------------------------------------------------------------------------


def confirmada_em(id_afirmacao: int, categoria: str, polaridade: str) -> dict:
    return afirmacao(
        id_afirmacao=id_afirmacao, categoria=categoria, polaridade=polaridade
    )


def test_perfil_validado_sem_afirmacoes_e_rejeitado():
    with pytest.raises(ValidationError):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[],
                hosts_distintos=[],
                estado_dimensoes_gap=dimensoes(dados_proprietarios=("desconhecido", [])),
            )
        )


def test_ids_de_afirmacao_validada_nao_sequenciais_sao_rejeitados():
    with pytest.raises(ValidationError, match="sequencial"):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[afirmacao(), afirmacao(id_afirmacao=3)],
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("capacidade_confirmada", [1, 3])
                ),
            )
        )


def test_ids_de_afirmacao_validada_duplicados_sao_rejeitados():
    with pytest.raises(ValidationError, match="sequencial"):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[afirmacao(), afirmacao()],
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("capacidade_confirmada", [1])
                ),
            )
        )


def test_taxa_de_derrubada_incoerente_com_as_situacoes_e_rejeitada():
    with pytest.raises(ValidationError, match="taxa_derrubada"):
        PerfilValidado.model_validate(perfil_validado(taxa_derrubada=0.5))


def test_taxa_de_derrubada_com_dizima_periodica_e_aceita():
    modelo = PerfilValidado.model_validate(
        perfil_validado(
            afirmacoes_validadas=[afirmacao(), derrubada(2), confirmada_em(3, "distribuicao", "presenca")],
            taxa_derrubada=1 / 3,
            estado_dimensoes_gap=dimensoes(
                distribuicao=("capacidade_confirmada", [3])
            ),
        )
    )
    assert modelo.taxa_derrubada == 1 / 3


def test_capacidade_confirmada_sem_evidencia_e_rejeitada():
    with pytest.raises(ValidationError, match="evidência"):
        EstadoDimensaoGap(
            dimensao="distribuicao", estado="capacidade_confirmada", ids_evidencias=[]
        )


def test_gap_confirmado_sem_evidencia_e_rejeitado():
    with pytest.raises(ValidationError, match="evidência"):
        EstadoDimensaoGap(
            dimensao="distribuicao", estado="gap_confirmado", ids_evidencias=[]
        )


def test_ids_de_evidencia_fora_de_ordem_sao_rejeitados():
    with pytest.raises(ValidationError, match="ordem"):
        EstadoDimensaoGap(
            dimensao="distribuicao",
            estado="capacidade_confirmada",
            ids_evidencias=[3, 1],
        )


def test_evidencia_dimensional_precisa_apontar_afirmacao_confirmada():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[derrubada(1)],
                taxa_derrubada=1.0,
                hosts_distintos=[],
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("capacidade_confirmada", [1])
                ),
            )
        )


def test_capacidade_confirmada_nao_aceita_evidencia_de_ausencia():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[
                    confirmada_em(1, "dados_proprietarios", "ausencia_explicita")
                ],
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("capacidade_confirmada", [1])
                ),
            )
        )


def test_gap_confirmado_nao_aceita_evidencia_de_presenca():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("gap_confirmado", [1])
                )
            )
        )


def test_evidencia_de_outra_dimensao_e_rejeitada():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("desconhecido", []),
                    distribuicao=("capacidade_confirmada", [1]),
                )
            )
        )


def test_desconhecido_sem_evidencia_decisiva_exige_lista_vazia():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("desconhecido", [1])
                )
            )
        )


def test_desconhecido_em_conflito_preserva_os_ids_dos_dois_lados():
    modelo = PerfilValidado.model_validate(
        perfil_validado(
            afirmacoes_validadas=[
                afirmacao(),
                confirmada_em(2, "dados_proprietarios", "ausencia_explicita"),
            ],
            estado_dimensoes_gap=dimensoes(
                dados_proprietarios=("desconhecido", [1, 2])
            ),
        )
    )
    assert modelo.estado_dimensoes_gap[0].ids_evidencias == [1, 2]


def test_conflito_que_descarta_um_dos_lados_e_rejeitado():
    with pytest.raises(ValidationError, match="dimens"):
        PerfilValidado.model_validate(
            perfil_validado(
                afirmacoes_validadas=[
                    afirmacao(),
                    confirmada_em(2, "dados_proprietarios", "ausencia_explicita"),
                ],
                estado_dimensoes_gap=dimensoes(
                    dados_proprietarios=("capacidade_confirmada", [1])
                ),
            )
        )


# --------------------------------------------------------------------------
# C7 — normalizar_dominio precisa ser ponto fixo de si mesma
# --------------------------------------------------------------------------


def test_normalizar_dominio_remove_prefixos_www_repetidos():
    assert normalizar_dominio("www.www.www.acme.example.com") == "acme.example.com"


def test_normalizar_dominio_e_idempotente():
    for bruto in (
        "acme.example.com",
        "www.acme.example.com",
        "WWW.WWW.Acme.Example.COM",
        "  www.www.acme.example.com  ",
    ):
        uma_vez = normalizar_dominio(bruto)
        assert normalizar_dominio(uma_vez) == uma_vez
