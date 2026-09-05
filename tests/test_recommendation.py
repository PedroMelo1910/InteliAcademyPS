"""Nó Recommendation: o LLM escolhe, o nó resolve, calcula e valida.

Todos os testes são offline: o provedor de rascunhos e a fronteira de
metadados são injetados. Nenhum teste chama Gemini, endpoint NVIDIA ou URL.
"""

from __future__ import annotations

from datetime import date

import pytest

from radar.agentes.recommendation import (
    ErroRecommendation,
    Recommendation,
)
from radar.contratos import (
    Classificacao,
    ContextoNvidia,
    EmpresaCandidata,
    FitScore,
    MetadadoDocumentoFitScore,
    Recomendacao,
)
from tests.conftest import (
    ProvedorSequencialFalso,
    afirmacao_validada_falsa,
    contexto_nvidia_falso,
    perfil_validado_falso,
    trecho_nvidia_falso,
)


DATA_ACESSO = date(2026, 6, 1)


class BaseMetadadosFalsa:
    """Fronteira de leitura do SQLite, reduzida ao que o nó precisa."""

    def __init__(self, metadados=None):
        self._metadados = metadados if metadados is not None else metadados_falsos()
        self.chamadas = 0
        self.ids_pedidos: list[list[int]] = []

    def carregar_metadados_fit_score(self, ids_documentos):
        self.chamadas += 1
        ids = list(ids_documentos)
        self.ids_pedidos.append(ids)
        return {
            id_documento: self._metadados[id_documento]
            for id_documento in ids
            if id_documento in self._metadados
        }


def metadados_falsos(hosts=("fonte-a.example", "fonte-b.example", "fonte-c.example")):
    return {
        indice: MetadadoDocumentoFitScore(
            id_documento=indice,
            url_fonte=f"https://{hosts[(indice - 1) % len(hosts)]}/doc-{indice}",
            host_normalizado=hosts[(indice - 1) % len(hosts)],
            data_publicacao=date(2026, 1, 15),
        )
        for indice in range(1, 6)
    }


def empresa_falsa(**ajustes) -> EmpresaCandidata:
    campos = {
        "id_startup": 7,
        "nome": "Acme IA",
        "setor": "Saúde",
        "estagio": "Seed",
        "localizacao": "São Paulo",
        "descricao_curta": "Plataforma de triagem clínica assistida por modelos.",
    }
    campos.update(ajustes)
    return EmpresaCandidata(**campos)


def perfil_padrao():
    """Gap estrutural confirmado em otimização técnica + dor documentada."""
    return perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "otimizacao_tecnica", polaridade="ausencia_explicita"
            ),
            afirmacao_validada_falsa(2, "dependencia_api_externa", id_documento=2),
            afirmacao_validada_falsa(3, "stack_propria", id_documento=3),
        ],
        hosts=["fonte-a.example", "fonte-b.example"],
    )


def perfil_multi_gaps():
    """Cinco gaps genuinamente sustentados: quatro estruturais e uma dor."""
    return perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "dados_proprietarios", polaridade="ausencia_explicita"
            ),
            afirmacao_validada_falsa(
                2, "workflow_profundo", polaridade="ausencia_explicita",
                id_documento=2,
            ),
            afirmacao_validada_falsa(
                3, "distribuicao", polaridade="ausencia_explicita", id_documento=3,
            ),
            afirmacao_validada_falsa(
                4, "otimizacao_tecnica", polaridade="ausencia_explicita",
                id_documento=4,
            ),
            afirmacao_validada_falsa(5, "dependencia_api_externa", id_documento=5),
        ],
        hosts=["fonte-a.example", "fonte-b.example"],
    )


def classificacao_padrao():
    return Classificacao(
        classe="AI-native",
        justificativa=(
            "Os modelos próprios são o produto entregue ao cliente. "
            "A plataforma não se sustenta sem eles."
        ),
        ids_afirmacoes_suporte=[3],
    )


def documento_recuperado(id_documento: int, id_startup: int = 7):
    return {
        "id_documento": id_documento,
        "id_startup": id_startup,
        "tipo": "site institucional",
        "titulo": f"Documento {id_documento}",
        "url_fonte": f"https://fonte-a.example/doc-{id_documento}",
        "dominio_fonte": "fonte-a.example",
        "data_acesso": DATA_ACESSO.isoformat(),
        "score_bm25": -1.0,
    }


def estado_pos_rag(perfil=None, contexto=None, empresa=None, **ajustes):
    empresa = empresa or empresa_falsa()
    estado = {
        "startup_selecionada": empresa.id_startup,
        "classificacao": classificacao_padrao(),
        "perfil_validado": perfil if perfil is not None else perfil_padrao(),
        "contexto_nvidia": contexto if contexto is not None else contexto_nvidia_falso(),
        "resultado_recuperacao": {
            "empresas": [empresa.model_dump()],
            "documentos": [
                documento_recuperado(1, empresa.id_startup),
                documento_recuperado(2, empresa.id_startup),
                documento_recuperado(3, empresa.id_startup),
            ],
            "filtros_aplicados": {},
        },
        "recomendacoes": None,
        "fit_score": None,
        "erros": [],
        "trajeto": ["extractor", "classifier", "evidence_validator", "nvidia_rag"],
    }
    estado.update(ajustes)
    return estado


def rascunho(
    *,
    gap="otimizacao_tecnica",
    tecnologias=("NVIDIA Triton Inference Server",),
    ids_afirmacoes=(1, 2),
    ids_chunks=(101, 102),
):
    return {
        "gap_enderecado": gap,
        "tecnologias": list(tecnologias),
        "justificativa_tecnica": (
            "O serving dedicado remove a dependência de uma API externa de inferência."
        ),
        "justificativa_negocio": (
            "A previsibilidade de custo por chamada melhora a margem por contrato."
        ),
        "proxima_acao": {
            "tipo_acao": "benchmark_custo_latencia",
            "detalhe": "Medir latência p95 e custo por mil chamadas na carga atual.",
        },
        "ids_afirmacoes": list(ids_afirmacoes),
        "ids_chunks": list(ids_chunks),
    }


def lote(*rascunhos):
    return {"rascunhos": list(rascunhos)}


def _recomendacao(saida, posicao: int = 0) -> Recomendacao:
    """Valida o que o nó gravou no estado antes de inspecionar o conteúdo."""
    return Recomendacao.model_validate(saida["recomendacoes"][posicao])


def executar(provedor, estado=None, base=None):
    no = Recommendation(base or BaseMetadadosFalsa(), provedor)
    return no(estado if estado is not None else estado_pos_rag())


def sem_recomendacao(saida) -> str:
    """§11.3: descarte total vira estado vazio e honesto, não exceção.

    O nó deixou de derrubar o grafo quando nenhuma recomendação sobrevive à
    conferência de proveniência: a análise segue para o Briefing terminal com
    ``recomendacoes`` vazia, ``fit_score`` nulo e o motivo em ``erros``. A
    garantia sob teste é a mesma de antes — rascunho sem lastro **nunca** vira
    recomendação —, só o modo de encerrar mudou.
    """
    assert saida["recomendacoes"] == []
    assert saida["fit_score"] is None
    assert saida["erros"]
    return " | ".join(saida["erros"])


# ----------------------------------------------------------------------
# Caminho normal e construção determinística
# ----------------------------------------------------------------------


def test_caminho_normal_produz_recomendacao_validada():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    saida = executar(provedor)

    assert provedor.chamadas == 1
    recomendacoes = [
        Recomendacao.model_validate(item) for item in saida["recomendacoes"]
    ]
    assert len(recomendacoes) == 1
    assert isinstance(saida["fit_score"], FitScore)


def test_ids_de_afirmacao_sao_resolvidos_para_evidencia_completa():
    saida = executar(ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1,)))))
    evidencias = _recomendacao(saida).evidencias_startup

    assert [item.id_afirmacao for item in evidencias] == [1]
    assert evidencias[0].id_documento == 1
    assert str(evidencias[0].url_fonte) == "https://fonte-a.example/doc-1"
    esperado = perfil_padrao().afirmacoes_validadas[0]
    assert evidencias[0].trecho_citado == esperado.trecho_citado


def test_ids_de_chunk_sao_resolvidos_para_citacao_completa():
    saida = executar(ProvedorSequencialFalso(lote(rascunho(ids_chunks=(101,)))))
    citacoes = _recomendacao(saida).citacoes_nvidia

    assert [item.id_chunk for item in citacoes] == [101]
    assert citacoes[0].tecnologia == "NVIDIA NIM"
    assert citacoes[0].origem == "tecnologia"
    assert str(citacoes[0].fonte_url) == "https://nvidia.example/101"
    assert citacoes[0].breadcrumb.startswith("NVIDIA NIM")


def test_no_registra_exatamente_um_item_de_trajeto():
    saida = executar(ProvedorSequencialFalso(lote(rascunho())))
    assert saida["trajeto"] == ["recommendation"]


def test_no_escreve_apenas_os_campos_que_lhe_pertencem():
    saida = executar(ProvedorSequencialFalso(lote(rascunho())))
    assert set(saida) <= {"recomendacoes", "fit_score", "trajeto", "erros"}
    assert {"recomendacoes", "fit_score", "trajeto"} <= set(saida)


def test_produz_ate_cinco_recomendacoes_no_caminho_normal():
    """Cada um dos cinco gaps cita ao menos um id que o sustenta de fato."""
    rascunhos = [
        rascunho(gap="dados_proprietarios", tecnologias=("cuDF",), ids_afirmacoes=(1,)),
        rascunho(
            gap="workflow_profundo", tecnologias=("NVIDIA Riva",), ids_afirmacoes=(2,)
        ),
        rascunho(
            gap="distribuicao",
            tecnologias=("NVIDIA Inception",),
            ids_afirmacoes=(3,),
        ),
        rascunho(gap="otimizacao_tecnica", ids_afirmacoes=(4,)),
        rascunho(
            gap="dependencia_api_externa",
            tecnologias=("NVIDIA NIM",),
            ids_afirmacoes=(5,),
        ),
    ]
    saida = executar(
        ProvedorSequencialFalso(lote(*rascunhos)),
        estado_pos_rag(perfil=perfil_multi_gaps()),
    )
    assert len(saida["recomendacoes"]) == 5
    assert saida.get("erros", []) == []


# ----------------------------------------------------------------------
# Prioridade e complexidade determinísticas
# ----------------------------------------------------------------------


def test_prioridade_alta_vem_de_dor_citada_dentro_da_janela_de_estagio():
    saida = executar(ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1, 2)))))
    assert _recomendacao(saida).prioridade == "alta"


def test_prioridade_cai_fora_da_janela_de_estagio():
    estado = estado_pos_rag(empresa=empresa_falsa(estagio="Série C"))
    saida = executar(
        ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1, 2)))), estado
    )
    assert _recomendacao(saida).prioridade == "media"


def test_prioridade_media_para_gap_confirmado_sem_dor_citada():
    saida = executar(ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1,)))))
    assert _recomendacao(saida).prioridade == "media"


def test_complexidade_usa_a_maior_tecnologia_do_pacote():
    saida = executar(
        ProvedorSequencialFalso(
            lote(
                rascunho(
                    tecnologias=("NVIDIA Triton Inference Server", "TensorRT-LLM")
                )
            )
        )
    )
    assert _recomendacao(saida).complexidade == "alta"


def test_complexidade_sobe_um_degrau_quando_a_centralidade_de_ia_e_baixa():
    """Perfil AI-enabled sem bônus de centralidade mantém P1 em 3."""
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "otimizacao_tecnica", polaridade="ausencia_explicita"
            ),
            afirmacao_validada_falsa(2, "distribuicao", id_documento=2),
        ],
        hosts=["fonte-a.example", "fonte-b.example"],
    )
    estado = estado_pos_rag(
        perfil=perfil,
        classificacao=Classificacao(
            classe="AI-enabled",
            justificativa=(
                "O produto existe sem os modelos. A camada de IA é adicional."
            ),
            ids_afirmacoes_suporte=[2],
        ),
    )
    saida = executar(
        ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1,)))), estado
    )
    assert saida["fit_score"].pilares[0].pontos == 3
    assert _recomendacao(saida).complexidade == "alta"


# ----------------------------------------------------------------------
# Fit-score: integração com a função pura já aprovada
# ----------------------------------------------------------------------


def test_fit_score_e_calculado_pela_funcao_pura_aprovada(monkeypatch):
    """Prova que o nó chama ``calcular_fit_score`` em vez de duplicar a rubrica."""
    import radar.agentes.recommendation as modulo

    chamadas = []
    original = modulo.calcular_fit_score

    def espiao(entrada):
        chamadas.append(entrada)
        return original(entrada)

    monkeypatch.setattr(modulo, "calcular_fit_score", espiao)
    saida = executar(ProvedorSequencialFalso(lote(rascunho())))

    assert len(chamadas) == 1
    assert chamadas[0].classe == "AI-native"
    assert chamadas[0].data_referencia == DATA_ACESSO
    assert saida["fit_score"].versao_rubrica == "rubrica-v1"


def test_fit_score_recebe_metadados_de_todos_os_documentos_do_perfil():
    base = BaseMetadadosFalsa()
    executar(ProvedorSequencialFalso(lote(rascunho())), base=base)
    assert base.ids_pedidos == [[1, 2, 3]]


def test_metadado_ausente_falha_sem_gravar_recomendacao():
    base = BaseMetadadosFalsa({1: metadados_falsos()[1]})
    with pytest.raises(ErroRecommendation, match="fit-score"):
        executar(ProvedorSequencialFalso(lote(rascunho())), base=base)


# ----------------------------------------------------------------------
# Proveniência: ids inventados, de outro perfil ou de outra recuperação
# ----------------------------------------------------------------------


def test_id_de_afirmacao_inventado_e_descartado_apos_o_retry():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(99,))),
        lote(rascunho(ids_afirmacoes=(98,))),
    )
    motivo = sem_recomendacao(executar(provedor))
    assert provedor.chamadas == 2
    assert "98" in motivo


def test_id_de_chunk_inventado_e_descartado_apos_o_retry():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_chunks=(999,))),
        lote(rascunho(ids_chunks=(998,))),
    )
    motivo = sem_recomendacao(executar(provedor))
    assert provedor.chamadas == 2
    assert "998" in motivo


def test_afirmacao_derrubada_nao_serve_como_evidencia():
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(
                1, "otimizacao_tecnica", polaridade="ausencia_explicita"
            ),
            afirmacao_validada_falsa(
                2, "dependencia_api_externa", situacao="derrubada", id_documento=2
            ),
        ],
        hosts=["fonte-a.example"],
    )
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(2,))), lote(rascunho(ids_afirmacoes=(2,)))
    )
    with pytest.raises(ErroRecommendation):
        executar(provedor, estado_pos_rag(perfil=perfil))


def test_id_de_afirmacao_de_outro_perfil_nao_e_aceito():
    """O id 4 existe em outra análise, não neste perfil validado."""
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(4,))), lote(rascunho(ids_afirmacoes=(4,)))
    )
    assert "4" in sem_recomendacao(executar(provedor))


def test_id_de_chunk_fora_do_contexto_atual_nao_e_aceito():
    contexto = ContextoNvidia(
        consulta_gerada="consulta atual",
        trechos=[trecho_nvidia_falso(200 + indice) for indice in range(5)],
    )
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_chunks=(101,))), lote(rascunho(ids_chunks=(101,)))
    )
    sem_recomendacao(executar(provedor, estado_pos_rag(contexto=contexto)))


def test_startup_fora_do_conjunto_recuperado_interrompe_o_no():
    with pytest.raises(ErroRecommendation, match="startup"):
        executar(
            ProvedorSequencialFalso(lote(rascunho())),
            estado_pos_rag(startup_selecionada=999),
        )


# ----------------------------------------------------------------------
# Catálogos fechados: tecnologia e mapeamento gap → tecnologia
# ----------------------------------------------------------------------


def test_tecnologia_fora_das_dezesseis_e_recusada_pelo_contrato_do_rascunho():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(tecnologias=("Produto inventado",))),
        lote(rascunho(tecnologias=("Produto inventado",))),
    )
    with pytest.raises(ErroRecommendation):
        executar(provedor)
    assert provedor.chamadas == 2


def test_tecnologia_fora_do_conjunto_candidato_do_gap_e_recusada():
    """Riva é uma das 16, mas não é candidata para otimização técnica."""
    provedor = ProvedorSequencialFalso(
        lote(rascunho(tecnologias=("NVIDIA Riva",))),
        lote(rascunho(tecnologias=("NVIDIA Riva",))),
    )
    assert "NVIDIA Riva" in sem_recomendacao(executar(provedor))


def test_tecnologia_candidata_do_gap_escolhido_e_aceita():
    saida = executar(
        ProvedorSequencialFalso(
            lote(
                rascunho(
                    gap="workflow_profundo",
                    tecnologias=("NVIDIA Riva",),
                    ids_afirmacoes=(2,),
                )
            )
        ),
        estado_pos_rag(perfil=perfil_multi_gaps()),
    )
    assert _recomendacao(saida).tecnologias == ["NVIDIA Riva"]


# ----------------------------------------------------------------------
# Citações: exigência de chunk de tecnologia
# ----------------------------------------------------------------------


def test_citacao_apenas_conceitual_e_recusada():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_chunks=(106,))), lote(rascunho(ids_chunks=(106,)))
    )
    assert "tecnologia" in sem_recomendacao(executar(provedor))


def test_chunk_conceitual_e_aceito_como_contexto_adicional():
    saida = executar(
        ProvedorSequencialFalso(lote(rascunho(ids_chunks=(101, 106))))
    )
    citacoes = _recomendacao(saida).citacoes_nvidia
    assert {item.origem for item in citacoes} == {"tecnologia", "conceitual"}


# ----------------------------------------------------------------------
# Retry único, descarte parcial e falha segura
# ----------------------------------------------------------------------


def test_retry_unico_corrige_rascunho_invalido():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(99,))), lote(rascunho())
    )
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert len(saida["recomendacoes"]) == 1
    assert saida.get("erros", []) == []


def test_retry_recebe_o_erro_exato_da_tentativa_anterior():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(99,))), lote(rascunho())
    )
    executar(provedor)
    correcao = "\n".join(conteudo for _papel, conteudo in provedor.mensagens[1])
    assert "99" in correcao


def test_nao_existe_segundo_retry():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_afirmacoes=(99,))),
        lote(rascunho(ids_afirmacoes=(99,))),
    )
    sem_recomendacao(executar(provedor))
    assert provedor.chamadas == 2


def test_descarte_parcial_preserva_as_recomendacoes_com_lastro():
    invalido = rascunho(
        gap="dependencia_api_externa",
        tecnologias=("NVIDIA Riva",),
        ids_afirmacoes=(2,),
    )
    provedor = ProvedorSequencialFalso(
        lote(rascunho(), invalido), lote(rascunho(), invalido)
    )
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert len(saida["recomendacoes"]) == 1
    assert _recomendacao(saida).gap_enderecado == "otimizacao_tecnica"
    assert len(saida["erros"]) == 1
    assert "dependencia_api_externa" in saida["erros"][0]
    assert "NVIDIA Riva" in saida["erros"][0]


def test_falha_completa_nao_grava_recomendacao_nem_fit_score():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(ids_chunks=(999,))), lote(rascunho(ids_chunks=(999,)))
    )
    assert "nenhuma recomendação" in sem_recomendacao(executar(provedor))


def test_resposta_fora_do_contrato_estruturado_consome_o_mesmo_retry():
    provedor = ProvedorSequencialFalso({"rascunhos": []}, lote(rascunho()))
    saida = executar(provedor)
    assert provedor.chamadas == 2
    assert len(saida["recomendacoes"]) == 1


def test_duas_respostas_fora_do_contrato_falham_sem_fabricar_saida():
    provedor = ProvedorSequencialFalso({"rascunhos": []}, {"rascunhos": []})
    with pytest.raises(ErroRecommendation, match="contrato estruturado"):
        executar(provedor)
    assert provedor.chamadas == 2


def test_falha_do_provedor_interrompe_sem_consumir_retry():
    provedor = ProvedorSequencialFalso(RuntimeError("Gemini indisponível"))
    with pytest.raises(ErroRecommendation, match="não respondeu"):
        executar(provedor)
    assert provedor.chamadas == 1


# ----------------------------------------------------------------------
# Pré-condições e informação mínima enviada ao LLM
# ----------------------------------------------------------------------


def test_no_exige_contexto_nvidia_no_estado():
    with pytest.raises(ErroRecommendation, match="ContextoNvidia"):
        executar(ProvedorSequencialFalso(lote(rascunho())), estado_pos_rag(contexto_nvidia=None))


def test_no_exige_classificacao_no_estado():
    with pytest.raises(ErroRecommendation, match="Classificacao"):
        executar(ProvedorSequencialFalso(lote(rascunho())), estado_pos_rag(classificacao=None))


def test_no_exige_perfil_validado_no_estado():
    with pytest.raises(ErroRecommendation, match="PerfilValidado"):
        executar(ProvedorSequencialFalso(lote(rascunho())), estado_pos_rag(perfil_validado=None))


def test_prompt_nao_pede_prioridade_complexidade_nem_fit_score():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    executar(provedor)
    prompt = "\n".join(conteudo for _papel, conteudo in provedor.mensagens[0])
    assert "não" in prompt.casefold()
    for proibido in ("prioridade", "complexidade", "fit-score"):
        assert proibido in prompt.casefold(), (
            "o prompt precisa declarar explicitamente que estes campos não são "
            "do LLM"
        )


def test_prompt_so_oferece_ids_confirmados_e_chunks_do_contexto_atual():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    executar(provedor)
    prompt = "\n".join(conteudo for _papel, conteudo in provedor.mensagens[0])
    assert "afirmação 1" in prompt
    assert "chunk 101" in prompt
    assert "chunk 999" not in prompt


def test_prompt_nao_envia_trecho_citado_nem_classe_de_referencia():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    executar(provedor)
    prompt = "\n".join(conteudo for _papel, conteudo in provedor.mensagens[0])
    assert "Trecho público verificável" not in prompt
    assert "classe_referencia" not in prompt


# ----------------------------------------------------------------------
# Elegibilidade do gap: a evidência citada precisa sustentar o gap escolhido
# ----------------------------------------------------------------------


def _recusa(rascunhos, estado=None):
    """Executa com o mesmo rascunho nas duas tentativas e devolve o motivo."""
    provedor = ProvedorSequencialFalso(lote(*rascunhos), lote(*rascunhos))
    saida = executar(provedor, estado)
    assert provedor.chamadas == 2
    return sem_recomendacao(saida)


def test_dimensao_estrutural_desconhecida_e_recusada_como_gap():
    """'distribuicao' está desconhecida no perfil padrão."""
    mensagem = _recusa(
        [
            rascunho(
                gap="distribuicao",
                tecnologias=("NVIDIA Inception",),
                ids_afirmacoes=(3,),
            )
        ]
    )
    assert "distribuicao" in mensagem
    assert "não está sustentado" in mensagem


def test_dimensao_com_capacidade_confirmada_e_recusada_como_gap():
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(1, "distribuicao", polaridade="presenca"),
            afirmacao_validada_falsa(
                2, "otimizacao_tecnica", polaridade="ausencia_explicita",
                id_documento=2,
            ),
            afirmacao_validada_falsa(3, "stack_propria", id_documento=3),
        ],
        hosts=["fonte-a.example"],
    )
    mensagem = _recusa(
        [
            rascunho(
                gap="distribuicao",
                tecnologias=("NVIDIA Inception",),
                ids_afirmacoes=(1,),
            )
        ],
        estado_pos_rag(perfil=perfil),
    )
    assert "não está sustentado" in mensagem


def test_gap_estrutural_citando_apenas_evidencia_alheia_e_recusado():
    """otimizacao_tecnica é gap, mas o rascunho cita só a afirmação 3."""
    mensagem = _recusa([rascunho(gap="otimizacao_tecnica", ids_afirmacoes=(3,))])
    assert "não sustentam o gap" in mensagem
    assert "otimizacao_tecnica" in mensagem


def test_gap_estrutural_citando_a_propria_evidencia_e_aceito():
    saida = executar(
        ProvedorSequencialFalso(
            lote(rascunho(gap="otimizacao_tecnica", ids_afirmacoes=(1, 3)))
        )
    )
    recomendacao = _recomendacao(saida)
    assert recomendacao.gap_enderecado == "otimizacao_tecnica"
    assert 1 in [item.id_afirmacao for item in recomendacao.evidencias_startup]


def test_gap_de_dor_citando_afirmacao_da_mesma_categoria_e_aceito():
    saida = executar(
        ProvedorSequencialFalso(
            lote(
                rascunho(
                    gap="dependencia_api_externa",
                    tecnologias=("NVIDIA NIM",),
                    ids_afirmacoes=(2,),
                )
            )
        )
    )
    assert _recomendacao(saida).gap_enderecado == "dependencia_api_externa"


def test_gap_de_dor_citando_apenas_outra_categoria_e_recusado():
    mensagem = _recusa(
        [
            rascunho(
                gap="dependencia_api_externa",
                tecnologias=("NVIDIA NIM",),
                ids_afirmacoes=(1, 3),
            )
        ]
    )
    assert "não sustentam o gap" in mensagem
    assert "dependencia_api_externa" in mensagem


def test_conflito_representado_como_desconhecido_nao_gera_recomendacao():
    """Presença e ausência confirmadas na mesma dimensão viram desconhecido."""
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(1, "workflow_profundo", polaridade="presenca"),
            afirmacao_validada_falsa(
                2, "workflow_profundo", polaridade="ausencia_explicita",
                id_documento=2,
            ),
            afirmacao_validada_falsa(3, "dependencia_api_externa", id_documento=3),
        ],
        hosts=["fonte-a.example"],
    )
    mensagem = _recusa(
        [
            rascunho(
                gap="workflow_profundo",
                tecnologias=("NVIDIA Riva",),
                ids_afirmacoes=(2,),
            )
        ],
        estado_pos_rag(perfil=perfil),
    )
    assert "não está sustentado" in mensagem


def test_sem_nenhum_gap_sustentado_o_provedor_nao_e_chamado():
    perfil = perfil_validado_falso(
        [
            afirmacao_validada_falsa(1, "stack_propria"),
            afirmacao_validada_falsa(2, "outro", id_documento=2),
            afirmacao_validada_falsa(
                3, "distribuicao", polaridade="presenca", id_documento=3
            ),
        ],
        hosts=["fonte-a.example"],
    )
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    motivo = sem_recomendacao(executar(provedor, estado_pos_rag(perfil=perfil)))
    assert "nenhum gap está sustentado" in motivo
    assert provedor.chamadas == 0


def test_sem_gap_sustentado_nada_e_gravado_nem_pontuado(monkeypatch):
    """A falha segura precede até o cálculo do fit-score."""
    import radar.agentes.recommendation as modulo

    chamadas = []
    monkeypatch.setattr(
        modulo, "calcular_fit_score", lambda entrada: chamadas.append(entrada)
    )
    perfil = perfil_validado_falso(
        [afirmacao_validada_falsa(1, "outro")], hosts=["fonte-a.example"]
    )
    motivo = sem_recomendacao(
        executar(
            ProvedorSequencialFalso(lote(rascunho())),
            estado_pos_rag(perfil=perfil),
        )
    )
    assert "nenhum gap está sustentado" in motivo
    assert chamadas == []


def test_gap_invalido_participa_do_retry_unico_e_pode_ser_corrigido():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(gap="otimizacao_tecnica", ids_afirmacoes=(3,))),
        lote(rascunho(gap="otimizacao_tecnica", ids_afirmacoes=(1,))),
    )
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert len(saida["recomendacoes"]) == 1
    assert saida.get("erros", []) == []


def test_gap_invalido_nao_sobrevive_ao_lado_de_um_rascunho_valido():
    invalido = rascunho(
        gap="distribuicao", tecnologias=("NVIDIA Inception",), ids_afirmacoes=(3,)
    )
    provedor = ProvedorSequencialFalso(
        lote(rascunho(), invalido), lote(rascunho(), invalido)
    )
    saida = executar(provedor)

    assert [item["gap_enderecado"] for item in saida["recomendacoes"]] == [
        "otimizacao_tecnica"
    ]
    assert len(saida["erros"]) == 1
    assert "distribuicao" in saida["erros"][0]


def test_prompt_separa_gaps_sustentados_dores_e_dimensoes_bloqueadas():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    executar(provedor)
    prompt = "\n".join(conteudo for _papel, conteudo in provedor.mensagens[0])

    assert "Dimensões estruturais confirmadas como gap: otimizacao_tecnica" in prompt
    assert "Categorias de dor documentada por afirmação confirmada: " in prompt
    assert "dependencia_api_externa (sustentado pelas afirmações [2])" in prompt
    assert "NÃO podem ser recomendadas como gap" in prompt
    assert "distribuicao (desconhecido)" in prompt


def test_prompt_so_oferece_o_catalogo_dos_gaps_sustentados():
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    executar(provedor)
    instrucao = provedor.mensagens[0][0][1]

    assert "  - otimizacao_tecnica:" in instrucao
    assert "  - dependencia_api_externa:" in instrucao
    assert "  - distribuicao:" not in instrucao
    assert "  - workflow_profundo:" not in instrucao


# ----------------------------------------------------------------------
# §6.1 — um pacote coeso por gap: duplicata é descartada, nunca fundida
# ----------------------------------------------------------------------


def _gaps(saida) -> list[str]:
    return [
        Recomendacao.model_validate(item).gap_enderecado
        for item in saida["recomendacoes"]
    ]


def test_tres_rascunhos_do_mesmo_gap_nao_viram_tres_recomendacoes():
    """O LLM insiste no mesmo gap nas duas tentativas; sobra um pacote só."""
    lote_repetido = lote(rascunho(), rascunho(), rascunho())
    provedor = ProvedorSequencialFalso(lote_repetido, lote(rascunho(), rascunho()))
    saida = executar(provedor)

    assert _gaps(saida) == ["otimizacao_tecnica"]
    assert len(saida["recomendacoes"]) == 1


def test_duplicata_no_primeiro_lote_e_corrigida_em_exatamente_um_retry():
    """Retry único: corrigido o lote, nada do erro anterior sobra no estado."""
    provedor = ProvedorSequencialFalso(
        lote(rascunho(), rascunho()),
        lote(
            rascunho(ids_afirmacoes=(1,)),
            rascunho(
                gap="dependencia_api_externa",
                tecnologias=("NVIDIA NIM",),
                ids_afirmacoes=(2,),
            ),
        ),
    )
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert sorted(_gaps(saida)) == ["dependencia_api_externa", "otimizacao_tecnica"]
    assert saida.get("erros", []) == []


def test_duplicata_que_persiste_apos_o_retry_registra_o_descarte():
    provedor = ProvedorSequencialFalso(
        lote(rascunho(), rascunho()), lote(rascunho(), rascunho())
    )
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert _gaps(saida) == ["otimizacao_tecnica"]
    assert len(saida["erros"]) == 1
    assert "uma única vez" in saida["erros"][0]
    assert "otimizacao_tecnica" in saida["erros"][0]


def test_gaps_distintos_sustentados_entram_juntos_sem_descarte():
    """A regra de unicidade não penaliza lote legítimo com gaps diferentes."""
    provedor = ProvedorSequencialFalso(
        lote(
            rascunho(ids_afirmacoes=(1,)),
            rascunho(
                gap="dependencia_api_externa",
                tecnologias=("NVIDIA NIM",),
                ids_afirmacoes=(2,),
            ),
        )
    )
    saida = executar(provedor)

    assert provedor.chamadas == 1
    assert sorted(_gaps(saida)) == ["dependencia_api_externa", "otimizacao_tecnica"]
    assert saida.get("erros", []) == []


def test_rascunho_seguinte_do_mesmo_gap_vale_quando_o_primeiro_foi_descartado():
    """Duplicata é relativa ao que foi aceito, não ao que apenas foi tentado.

    O primeiro rascunho do gap cai pela tecnologia fora das candidatas; o
    segundo, do mesmo gap, continua sendo candidato legítimo — e o descarte
    registrado precisa ser o da tecnologia, não o da regra de unicidade.
    """
    invalido = rascunho(tecnologias=("NVIDIA Riva",))  # fora das candidatas do gap
    par = lote(invalido, rascunho())
    provedor = ProvedorSequencialFalso(par, par)
    saida = executar(provedor)

    assert provedor.chamadas == 2
    assert _gaps(saida) == ["otimizacao_tecnica"]
    assert len(saida["erros"]) == 1
    assert "NVIDIA Riva" in saida["erros"][0]
    assert "uma única vez" not in saida["erros"][0]


# ----------------------------------------------------------------------
# Fronteira do consumidor: o estado chega em forma JSON e é revalidado
# ----------------------------------------------------------------------


def test_contexto_nvidia_em_forma_json_e_aceito_pelo_consumidor():
    """O grafo grava dicionário; o nó precisa reidratar sem reclamar."""
    contexto_json = contexto_nvidia_falso().model_dump(mode="json")
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    saida = executar(provedor, estado_pos_rag(contexto=contexto_json))

    assert len(saida["recomendacoes"]) == 1
    assert all(isinstance(item, dict) for item in saida["recomendacoes"])


def test_contexto_nvidia_malformado_e_recusado_na_fronteira_do_consumidor():
    """O consumidor valida a entrada antes de usar; não confia na anotação."""
    provedor = ProvedorSequencialFalso(lote(rascunho()))
    estado = estado_pos_rag(contexto={"consulta_gerada": "consulta", "trechos": []})

    with pytest.raises(ErroRecommendation):
        executar(provedor, estado)
    assert provedor.chamadas == 0


# ----------------------------------------------------------------------
# §10.2 — a grafia inglesa do estágio não pode rebaixar a prioridade
# ----------------------------------------------------------------------


def test_prioridade_alta_para_estagio_series_a_com_dor_citada():
    estado = estado_pos_rag(empresa=empresa_falsa(estagio="Series A"))
    saida = executar(
        ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1, 2)))), estado
    )
    assert _recomendacao(saida).prioridade == "alta"


def test_prioridade_media_para_estagio_series_b_com_dor_citada():
    estado = estado_pos_rag(empresa=empresa_falsa(estagio="Series B"))
    saida = executar(
        ProvedorSequencialFalso(lote(rascunho(ids_afirmacoes=(1, 2)))), estado
    )
    assert _recomendacao(saida).prioridade == "media"
