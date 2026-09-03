import pytest
from pydantic import ValidationError

from radar.agentes.classifier import Classifier, ErroClassificador
from radar.contratos import Classificacao, PerfilExtraido


# --------------------------------------------------------------------------
# Perfis fixos: o Classifier só enxerga PerfilExtraido, nunca o banco.
# --------------------------------------------------------------------------


def perfil_ai_native(**ajustes) -> dict:
    base = {
        "id_startup": 11,
        "resumo_produto": (
            "A Acme Vision vende inspeção visual automatizada para linhas de montagem. "
            "A detecção de defeitos é o próprio produto entregue ao cliente."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A Acme Vision treina modelos próprios de detecção de defeitos.",
                "categoria": "stack_propria",
                "polaridade": "neutro",
                "id_documento": 101,
                "trecho_citado": "treina modelos próprios de detecção de defeitos",
            },
            {
                "id_afirmacao": 2,
                "texto": "O banco de imagens rotuladas da empresa é proprietário.",
                "categoria": "dados_proprietarios",
                "polaridade": "presenca",
                "id_documento": 101,
                "trecho_citado": "banco de imagens rotuladas da empresa é proprietário",
            },
            {
                "id_afirmacao": 3,
                "texto": "A plataforma inspeciona 12 fábricas clientes no Brasil.",
                "categoria": "escala_e_dor_operacional",
                "polaridade": "neutro",
                "id_documento": 102,
                "trecho_citado": "inspeciona 12 fábricas clientes no Brasil",
            },
        ],
    }
    base.update(ajustes)
    return base


def perfil_ai_enabled(**ajustes) -> dict:
    base = {
        "id_startup": 12,
        "resumo_produto": (
            "A Boreal opera uma plataforma de cartão de benefícios corporativos. "
            "O cartão é o produto contratado pelas empresas clientes."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A Boreal opera um cartão de benefícios usado por empresas clientes.",
                "categoria": "distribuicao",
                "polaridade": "presenca",
                "id_documento": 201,
                "trecho_citado": "cartão de benefícios usado por empresas clientes",
            },
            {
                "id_afirmacao": 2,
                "texto": "A empresa adicionou um assistente de atendimento sobre modelos de linguagem.",
                "categoria": "dependencia_api_externa",
                "polaridade": "neutro",
                "id_documento": 202,
                "trecho_citado": "assistente de atendimento sobre modelos de linguagem",
            },
        ],
    }
    base.update(ajustes)
    return base


def perfil_non_ai(**ajustes) -> dict:
    base = {
        "id_startup": 13,
        "resumo_produto": (
            "A Cedro Logística vende roteirização de entregas urbanas para varejistas. "
            "O planejamento é conduzido por operadores humanos em painéis de controle."
        ),
        "afirmacoes": [
            {
                "id_afirmacao": 1,
                "texto": "A Cedro entrega roteirização conduzida por planejadores humanos.",
                "categoria": "workflow_profundo",
                "polaridade": "presenca",
                "id_documento": 301,
                "trecho_citado": "roteirização conduzida por planejadores humanos",
            },
            {
                "id_afirmacao": 2,
                "texto": "A empresa mantém um treinamento interno de IA para o time comercial.",
                "categoria": "equipe_e_contratacao",
                "polaridade": "neutro",
                "id_documento": 302,
                "trecho_citado": "treinamento interno de IA para o time comercial",
            },
            {
                "id_afirmacao": 3,
                "texto": "A reportagem afirma que o produto não emprega modelos de aprendizado de máquina.",
                "categoria": "otimizacao_tecnica",
                "polaridade": "ausencia_explicita",
                "id_documento": 303,
                "trecho_citado": "o produto não emprega modelos de aprendizado de máquina",
            },
        ],
    }
    base.update(ajustes)
    return base


def classificacao(**ajustes) -> dict:
    base = {
        "classe": "AI-native",
        "justificativa": (
            "A empresa treina modelos próprios de detecção de defeitos. "
            "Sem esses modelos não resta produto para o cliente."
        ),
        "ids_afirmacoes_suporte": [1, 2],
    }
    base.update(ajustes)
    return base


def estado(perfil: dict, **ajustes) -> dict:
    base = {"perfil_extraido": PerfilExtraido.model_validate(perfil)}
    base.update(ajustes)
    return base


class ProvedorSequencial:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas: list[list[tuple[str, str]]] = []

    def invocar(self, mensagens):
        self.chamadas.append(mensagens)
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    @property
    def ultimo_prompt(self) -> str:
        return "\n".join(texto for _, texto in self.chamadas[-1])


# --------------------------------------------------------------------------
# Classificação válida e as três classes operacionais
# --------------------------------------------------------------------------


def test_classificacao_valida_entra_no_estado():
    provedor = ProvedorSequencial(classificacao())
    resultado = Classifier(provedor)(estado(perfil_ai_native()))

    assert isinstance(resultado["classificacao"], Classificacao)
    assert resultado["classificacao"].classe == "AI-native"
    assert resultado["classificacao"].ids_afirmacoes_suporte == [1, 2]
    assert resultado["trajeto"] == ["classifier"]
    assert len(provedor.chamadas) == 1


def test_as_tres_classes_operacionais_saem_de_perfis_fixos():
    casos = (
        (
            perfil_ai_native(),
            classificacao(),
        ),
        (
            perfil_ai_enabled(),
            classificacao(
                classe="AI-enabled",
                justificativa=(
                    "O cartão de benefícios existe sem qualquer modelo. "
                    "O assistente de atendimento é uma camada adicionada sobre esse produto."
                ),
                ids_afirmacoes_suporte=[1, 2],
            ),
        ),
        (
            perfil_non_ai(),
            classificacao(
                classe="non-AI",
                justificativa=(
                    "A roteirização entregue ao cliente é conduzida por planejadores humanos. "
                    "A reportagem registra que o produto não emprega modelos de aprendizado de máquina."
                ),
                ids_afirmacoes_suporte=[1, 3],
            ),
        ),
    )
    for perfil, resposta in casos:
        provedor = ProvedorSequencial(resposta)
        resultado = Classifier(provedor)(estado(perfil))
        assert resultado["classificacao"].classe == resposta["classe"]
        assert len(provedor.chamadas) == 1


# --------------------------------------------------------------------------
# Ids de suporte
# --------------------------------------------------------------------------


def test_id_de_suporte_inexistente_no_perfil_e_rejeitado():
    invalida = classificacao(ids_afirmacoes_suporte=[1, 99])
    provedor = ProvedorSequencial(invalida, invalida)
    with pytest.raises(ErroClassificador, match="duas vezes fora do contrato"):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2
    assert "99" in provedor.chamadas[1][-1][1]


def test_id_de_suporte_de_outro_perfil_e_rejeitado():
    # O perfil AI-enabled tem apenas as afirmações 1 e 2.
    invalida = classificacao(classe="AI-enabled", ids_afirmacoes_suporte=[3])
    provedor = ProvedorSequencial(invalida, invalida)
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_enabled()))
    assert len(provedor.chamadas) == 2


def test_ids_de_suporte_duplicados_sao_rejeitados():
    invalida = classificacao(ids_afirmacoes_suporte=[1, 1])
    provedor = ProvedorSequencial(invalida, invalida)
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2


def test_todos_os_ids_do_perfil_podem_sustentar_a_classe():
    valida = classificacao(ids_afirmacoes_suporte=[1, 2, 3])
    provedor = ProvedorSequencial(valida)
    resultado = Classifier(provedor)(estado(perfil_ai_native()))
    assert resultado["classificacao"].ids_afirmacoes_suporte == [1, 2, 3]


# --------------------------------------------------------------------------
# Fatos não sustentados pelo perfil
# --------------------------------------------------------------------------


def test_justificativa_com_numero_ausente_do_perfil_e_rejeitada():
    invalida = classificacao(
        justificativa=(
            "A empresa treina modelos próprios de detecção de defeitos. "
            "A plataforma atende 400 fábricas clientes."
        )
    )
    provedor = ProvedorSequencial(invalida, invalida)
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert "400" in provedor.chamadas[1][-1][1]


def test_justificativa_pode_citar_numero_presente_no_perfil():
    valida = classificacao(
        justificativa=(
            "A empresa treina modelos próprios de detecção de defeitos. "
            "A plataforma inspeciona 12 fábricas clientes no Brasil."
        ),
        ids_afirmacoes_suporte=[1, 3],
    )
    provedor = ProvedorSequencial(valida)
    resultado = Classifier(provedor)(estado(perfil_ai_native()))
    assert resultado["classificacao"].classe == "AI-native"
    assert len(provedor.chamadas) == 1


def test_numero_de_afirmacao_nao_selecionada_nao_sustenta_justificativa():
    invalida = classificacao(
        justificativa=(
            "A empresa treina modelos próprios de detecção de defeitos. "
            "A plataforma inspeciona 12 fábricas clientes no Brasil."
        ),
        ids_afirmacoes_suporte=[1, 2],
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))

    assert len(provedor.chamadas) == 2
    assert "12" in provedor.chamadas[1][-1][1]


def test_justificativa_pode_referenciar_os_ids_das_afirmacoes():
    valida = classificacao(
        justificativa=(
            "As afirmações [1 e 2] mostram modelos e base de dados próprios. "
            "Sem esses modelos não resta produto para o cliente."
        )
    )
    provedor = ProvedorSequencial(valida)
    assert Classifier(provedor)(estado(perfil_ai_native()))["classificacao"]
    assert len(provedor.chamadas) == 1


def test_id_de_afirmacao_nao_mascara_numero_factual_ausente():
    invalida = classificacao(
        justificativa=(
            "A empresa treina modelos próprios de detecção de defeitos. "
            "A plataforma atende 2 milhões de fábricas clientes."
        )
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))

    assert len(provedor.chamadas) == 2
    assert "2" in provedor.chamadas[1][-1][1]


def test_referencia_de_id_sem_delimitador_nao_mascara_grandeza_factual():
    invalida = classificacao(
        justificativa=(
            "A empresa treina modelos próprios de detecção de defeitos. "
            "Conforme afirmação 2 milhões de clientes usam a plataforma."
        ),
        ids_afirmacoes_suporte=[1, 2],
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))

    assert len(provedor.chamadas) == 2
    assert "2" in provedor.chamadas[1][-1][1]


def test_referencia_delimitada_nao_aceita_id_fora_do_suporte_selecionado():
    invalida = classificacao(
        justificativa=(
            "A afirmação [3] descreve a escala da plataforma. "
            "A empresa treina modelos próprios de detecção de defeitos."
        ),
        ids_afirmacoes_suporte=[1, 2],
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))

    assert len(provedor.chamadas) == 2
    assert "3" in provedor.chamadas[1][-1][1]


# --------------------------------------------------------------------------
# Desconhecido não é ausência declarada
# --------------------------------------------------------------------------


def test_non_ai_apoiada_apenas_em_ausencia_declarada_e_rejeitada():
    invalida = classificacao(
        classe="non-AI",
        justificativa=(
            "A reportagem registra que o produto não emprega modelos. "
            "Nada indica uso de aprendizado de máquina no produto."
        ),
        ids_afirmacoes_suporte=[3],
    )
    provedor = ProvedorSequencial(invalida, invalida)
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_non_ai()))
    assert "evidência positiva" in provedor.chamadas[1][-1][1]


def test_non_ai_nao_aceita_ausencia_com_treinamento_interno_irrelevante():
    invalida = classificacao(
        classe="non-AI",
        justificativa=(
            "A reportagem registra que o produto não emprega modelos. "
            "O time comercial participa de treinamento interno de IA."
        ),
        ids_afirmacoes_suporte=[2, 3],
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_non_ai()))

    assert len(provedor.chamadas) == 2
    assert "fatos apenas contextuais" in provedor.chamadas[1][-1][1]


def test_non_ai_nao_aceita_ausencia_com_fato_coringa_irrelevante():
    perfil = perfil_non_ai()
    perfil["afirmacoes"][1].update(
        texto="A empresa foi fundada na cidade de São Paulo.",
        categoria="outro",
        polaridade="neutro",
        trecho_citado="foi fundada na cidade de São Paulo",
    )
    invalida = classificacao(
        classe="non-AI",
        justificativa=(
            "A reportagem registra que o produto não emprega modelos. "
            "A empresa foi fundada na cidade de São Paulo."
        ),
        ids_afirmacoes_suporte=[2, 3],
    )
    provedor = ProvedorSequencial(invalida, invalida)

    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil))

    assert len(provedor.chamadas) == 2
    assert "evidência positiva" in provedor.chamadas[1][-1][1]


def test_non_ai_com_evidencia_positiva_do_produto_e_aceita():
    valida = classificacao(
        classe="non-AI",
        justificativa=(
            "A roteirização entregue ao cliente é conduzida por planejadores humanos. "
            "A reportagem registra que o produto não emprega modelos de aprendizado de máquina."
        ),
        ids_afirmacoes_suporte=[1, 3],
    )
    provedor = ProvedorSequencial(valida)
    resultado = Classifier(provedor)(estado(perfil_non_ai()))
    assert resultado["classificacao"].classe == "non-AI"
    assert len(provedor.chamadas) == 1


# --------------------------------------------------------------------------
# Prompt: definições operacionais e limites de inferência
# --------------------------------------------------------------------------


def test_prompt_traz_as_definicoes_operacionais_das_tres_classes():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(estado(perfil_ai_native()))
    prompt = provedor.ultimo_prompt
    assert "AI-native" in prompt
    assert "AI-enabled" in prompt
    assert "non-AI" in prompt
    assert "a IA é o produto" in prompt
    assert "camada funcional adicionada" in prompt
    assert "de 2 a 4 frases" in prompt


def test_prompt_distingue_desconhecido_de_ausencia_declarada():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(estado(perfil_ai_native()))
    prompt = provedor.ultimo_prompt
    assert "Desconhecido não é ausência" in prompt
    assert "evidência positiva" in prompt


def test_prompt_recusa_jargao_isolado_e_ferramenta_interna():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(estado(perfil_ai_native()))
    prompt = provedor.ultimo_prompt
    assert "jargão" in prompt
    assert "Ferramenta interna" in prompt
    assert "treinamento em IA" in prompt


def test_prompt_apresenta_as_afirmacoes_e_os_ids_disponiveis():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(estado(perfil_ai_native()))
    prompt = provedor.ultimo_prompt
    assert "treina modelos próprios de detecção de defeitos" in prompt
    assert "dados_proprietarios" in prompt
    assert "[1, 2, 3]" in prompt


# --------------------------------------------------------------------------
# Isolamento da entrada
# --------------------------------------------------------------------------


def test_o_classifier_nao_expoe_classe_referencia():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(estado(perfil_ai_native()))
    assert "classe_referencia" not in provedor.ultimo_prompt


def test_o_classifier_le_apenas_o_perfil_do_estado():
    provedor = ProvedorSequencial(classificacao())
    Classifier(provedor)(
        estado(
            perfil_ai_native(),
            consulta_usuario="SENTINELA_CONSULTA",
            resultado_recuperacao="SENTINELA_RECUPERACAO",
            plano_consulta="SENTINELA_PLANO",
            documento_integral="SENTINELA_TEXTO_INTEGRAL_DO_DOCUMENTO",
        )
    )
    prompt = provedor.ultimo_prompt
    for sentinela in (
        "SENTINELA_CONSULTA",
        "SENTINELA_RECUPERACAO",
        "SENTINELA_PLANO",
        "SENTINELA_TEXTO_INTEGRAL_DO_DOCUMENTO",
    ):
        assert sentinela not in prompt


def test_perfil_com_campo_estranho_interrompe_o_no():
    provedor = ProvedorSequencial(classificacao())
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(
            {"perfil_extraido": perfil_ai_native(classe_referencia="AI-native")}
        )
    assert provedor.chamadas == []


def test_perfil_ausente_interrompe_o_no():
    provedor = ProvedorSequencial(classificacao())
    with pytest.raises(ErroClassificador, match="PerfilExtraido"):
        Classifier(provedor)({"consulta_usuario": "sem perfil"})
    assert provedor.chamadas == []


# --------------------------------------------------------------------------
# Retry único e falha segura
# --------------------------------------------------------------------------


def test_correcao_bem_sucedida_na_unica_tentativa_permitida():
    invalida = classificacao(ids_afirmacoes_suporte=[99])
    provedor = ProvedorSequencial(invalida, classificacao())
    resultado = Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2
    assert "Falha de validação" in provedor.chamadas[1][-1][1]
    assert resultado["classificacao"].classe == "AI-native"


def test_duas_respostas_invalidas_falham_com_seguranca():
    provedor = ProvedorSequencial({"classe": "AI-native"}, {"classe": "AI-native"})
    with pytest.raises(ErroClassificador, match="nenhuma classificação"):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2


def test_saida_estruturada_malformada_e_rejeitada():
    provedor = ProvedorSequencial("texto livre", "ainda texto livre")
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2


def test_erro_pydantic_do_adaptador_tambem_usa_o_retry():
    try:
        Classificacao.model_validate({"classe": "AI-native"})
    except ValidationError as erro_validacao:
        falha_do_adaptador = erro_validacao
    else:  # pragma: no cover - proteção contra alteração acidental do contrato
        raise AssertionError("a classificação incompleta deveria produzir ValidationError")

    provedor = ProvedorSequencial(falha_do_adaptador, classificacao())
    resultado = Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2
    assert resultado["classificacao"].classe == "AI-native"


def test_falha_do_provedor_nao_fabrica_classificacao():
    provedor = ProvedorSequencial(RuntimeError("indisponível"))
    with pytest.raises(ErroClassificador, match="não respondeu"):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 1


def test_no_maximo_duas_chamadas_ao_provedor():
    invalida = classificacao(ids_afirmacoes_suporte=[99])
    provedor = ProvedorSequencial(invalida, invalida, classificacao())
    with pytest.raises(ErroClassificador):
        Classifier(provedor)(estado(perfil_ai_native()))
    assert len(provedor.chamadas) == 2
    assert len(provedor.respostas) == 1


# --------------------------------------------------------------------------
# Reexecução e estado derivado
# --------------------------------------------------------------------------


def test_reexecucao_substitui_a_classificacao_anterior():
    anterior = Classifier(ProvedorSequencial(classificacao()))(
        estado(perfil_ai_native())
    )["classificacao"]
    nova_resposta = classificacao(
        classe="AI-enabled",
        justificativa=(
            "A inspeção visual é vendida sobre uma operação industrial existente. "
            "Os modelos formam uma camada adicionada ao produto."
        ),
    )
    resultado = Classifier(ProvedorSequencial(nova_resposta))(
        estado(perfil_ai_native(), classificacao=anterior)
    )
    assert resultado["classificacao"].classe == "AI-enabled"
    assert resultado["trajeto"] == ["classifier"]


def test_campos_derivados_da_classificacao_sao_invalidados(monkeypatch):
    monkeypatch.setattr(
        "radar.agentes.classifier.CAMPOS_DERIVADOS_DA_CLASSIFICACAO",
        ("campo_derivado_futuro",),
    )
    resultado = Classifier(ProvedorSequencial(classificacao()))(
        estado(perfil_ai_native())
    )
    assert resultado["campo_derivado_futuro"] is None


def test_o_classifier_nao_recebe_base_de_startups():
    # A assinatura do nó é a garantia estrutural de que ele não lê o banco.
    provedor = ProvedorSequencial(classificacao())
    classificador = Classifier(provedor)
    assert not hasattr(classificador, "base")


def test_classificar_invalida_o_estado_derivado_da_classificacao_anterior():
    """Reclassificar sem limpar o downstream deixaria um perfil validado órfão."""
    from radar.agentes.classifier import CAMPOS_DERIVADOS_DA_CLASSIFICACAO

    assert CAMPOS_DERIVADOS_DA_CLASSIFICACAO == (
        "perfil_validado",
        "confianca_perfil",
        "contexto_nvidia",
        "recomendacoes",
        "fit_score",
    )
    saida = Classifier(ProvedorSequencial(classificacao()))(estado(perfil_ai_native()))
    for campo in CAMPOS_DERIVADOS_DA_CLASSIFICACAO:
        assert saida[campo] is None
