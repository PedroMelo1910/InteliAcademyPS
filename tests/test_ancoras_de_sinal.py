"""Um sinal técnico exige âncora literal explícita no próprio trecho citado.

O lote ao vivo provou que instrução de prompt não basta: o Gemini anexou
`visao_computacional` a "manejo digital de plantas daninhas" e `inferencia_llm`
a "a inteligência artificial da empresa". Esta é uma porta determinística e
conservadora — não classifica verdade semântica, só recusa atribuição sem o
mínimo de evidência explícita que a matriz congelada exige.
"""

from __future__ import annotations

import copy

import pytest

from radar.agentes.ancoras_sinal import (
    ANCORAS_POR_SINAL,
    completar_sinais_inequivocos,
    filtrar_sinais_sem_ancora,
    sinais_com_ancora,
    sinais_possivelmente_nao_extraidos,
    sinais_possivelmente_omitidos,
)
from radar.contratos import SINAIS_TECNICOS, PerfilExtraido
from tests.conftest import ProvedorSequencial
from tests.test_extractor import controlada  # noqa: F401 - registra a fixture


def afirmacao(sinais, trecho, **ajustes):
    campos = {
        "id_afirmacao": 1,
        "texto": "Um fato descrito na fonte pública.",
        "categoria": "stack_propria",
        "polaridade": "neutro",
        "id_documento": 1,
        "trecho_citado": trecho,
        # preserva o valor cru: o caso de tipo errado precisa chegar torto
        "sinais_tecnicos": list(sinais) if isinstance(sinais, (list, tuple)) else sinais,
    }
    campos.update(ajustes)
    return campos


def bruto(*afirmacoes):
    return {
        "id_startup": 1,
        "resumo_produto": "A empresa vende um produto. O produto chega ao cliente.",
        "afirmacoes": list(afirmacoes),
    }


def sinais_de(resultado, indice=0):
    return resultado["afirmacoes"][indice]["sinais_tecnicos"]


# ----------------------------------------------------------------------
# A tabela de âncoras cobre o vocabulário congelado
# ----------------------------------------------------------------------


def test_toda_categoria_de_sinal_tem_ancora_declarada():
    assert set(ANCORAS_POR_SINAL) == set(SINAIS_TECNICOS)


@pytest.mark.parametrize(
    "sinal,trecho",
    [
        ("inferencia_llm", "a empresa opera um modelo de linguagem próprio"),
        ("treinamento_ou_finetuning", "o processo inclui pré-treino e ajuste fino"),
        ("voz_fala_ou_transcricao", "permite interações faladas e transcrição"),
        ("dados_em_escala", "um banco de 3,5 bilhões de amostras rotuladas"),
        ("machine_learning_classico", "constrói modelos preditivos de risco"),
        ("visao_computacional", "usa visão computacional para processar imagens"),
        ("robotica_ou_simulacao", "veículos que trafegam sem motorista"),
        ("imagem_medica", "análise de exames de radiologia"),
        ("agentes_com_acoes_ou_controles", "o agente executa ações no sistema"),
        ("ciberseguranca_em_escala", "detecção de intrusão no tráfego de rede"),
    ],
)
def test_cada_sinal_sobrevive_quando_o_trecho_traz_a_ancora(sinal, trecho):
    resultado = filtrar_sinais_sem_ancora(bruto(afirmacao([sinal], trecho)))

    assert sinais_de(resultado) == [sinal]


# ----------------------------------------------------------------------
# Os quatro defeitos confirmados no lote ao vivo
# ----------------------------------------------------------------------


def test_cromai_manejo_digital_nao_sustenta_visao_computacional():
    trecho = "se descreve como líder em manejo digital de plantas daninhas"

    resultado = filtrar_sinais_sem_ancora(
        bruto(afirmacao(["visao_computacional"], trecho))
    )

    assert sinais_de(resultado) == []


def test_cromai_visao_computacional_explicita_permanece():
    trecho = (
        "usando visão computacional e inteligência artificial para processar "
        "imagens aéreas"
    )

    resultado = filtrar_sinais_sem_ancora(
        bruto(afirmacao(["visao_computacional"], trecho))
    )

    assert sinais_de(resultado) == ["visao_computacional"]


@pytest.mark.parametrize(
    "trecho",
    [
        "descreve o AIrton como a inteligência artificial da empresa",
        "o AIrton, agente de inteligência artificial integrado ao WhatsApp",
        "um assistente com IA para o time comercial",
    ],
)
def test_inteligencia_artificial_generica_nao_sustenta_inferencia_llm(trecho):
    resultado = filtrar_sinais_sem_ancora(bruto(afirmacao(["inferencia_llm"], trecho)))

    assert "inferencia_llm" not in sinais_de(resultado)


def test_wine_mantem_o_sinal_de_agente_e_perde_so_o_de_llm():
    trecho = (
        "A solução executa ações como cancelamentos, alterações de endereço e "
        "abertura de tickets"
    )

    resultado = filtrar_sinais_sem_ancora(
        bruto(
            afirmacao(
                ["agentes_com_acoes_ou_controles", "inferencia_llm"], trecho
            )
        )
    )

    assert sinais_de(resultado) == ["agentes_com_acoes_ou_controles"]


def test_colab_ia_generativa_explicita_sustenta_inferencia_llm():
    trecho = (
        "Apresenta o Colab Gov.AI, ferramenta de inteligência artificial "
        "generativa criada em parceria"
    )

    resultado = filtrar_sinais_sem_ancora(bruto(afirmacao(["inferencia_llm"], trecho)))

    assert sinais_de(resultado) == ["inferencia_llm"]


def test_mombak_imagens_de_drone_sustentam_visao_independente_da_classe():
    trecho = (
        "conecta dados de campo coletados com drones a dados tridimensionais e "
        "imagens de satélite"
    )

    resultado = filtrar_sinais_sem_ancora(
        bruto(afirmacao(["visao_computacional"], trecho))
    )

    assert sinais_de(resultado) == ["visao_computacional"]


def test_origo_modelos_analiticos_mantem_a_ancora_de_ml():
    """A âncora existe; a modalidade de vaga é responsabilidade do prompt."""
    trecho = (
        "A vaga da Órigo para Especialista de Performance atribui ao cargo a "
        "construção de modelos analíticos"
    )

    resultado = filtrar_sinais_sem_ancora(
        bruto(afirmacao(["machine_learning_classico"], trecho))
    )

    assert sinais_de(resultado) == ["machine_learning_classico"]


# ----------------------------------------------------------------------
# Invariantes da porta
# ----------------------------------------------------------------------


def test_a_afirmacao_sobrevive_inteira_quando_o_sinal_cai():
    original = afirmacao(["inferencia_llm"], "a empresa usa inteligência artificial")

    resultado = filtrar_sinais_sem_ancora(bruto(original))
    sobrevivente = resultado["afirmacoes"][0]

    assert sinais_de(resultado) == []
    for campo in ("id_afirmacao", "texto", "categoria", "polaridade", "id_documento", "trecho_citado"):
        assert sobrevivente[campo] == original[campo]


def test_a_porta_nao_muta_a_resposta_original_do_provedor():
    entrada = bruto(afirmacao(["inferencia_llm"], "apenas inteligência artificial"))
    fotografia = copy.deepcopy(entrada)

    filtrar_sinais_sem_ancora(entrada)

    assert entrada == fotografia


def test_a_ordem_declarada_e_preservada_para_o_contrato_ordenar():
    """Ordenar canonicamente é do Pydantic; a porta só filtra."""
    trecho = "o agente executa ações usando um modelo de linguagem próprio"

    resultado = filtrar_sinais_sem_ancora(
        bruto(
            afirmacao(
                ["agentes_com_acoes_ou_controles", "inferencia_llm"], trecho
            )
        )
    )

    assert sinais_de(resultado) == [
        "agentes_com_acoes_ou_controles",
        "inferencia_llm",
    ]


# ----------------------------------------------------------------------
# A porta não pode esconder violação de contrato do Pydantic
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "declarados",
    [
        ["inferencia_llm", "inferencia_llm"],
        ["sinal_inventado"],
        ["inferencia_llm", "sinal_inventado"],
        ["inferencia_llm", 3],
        [None],
        "inferencia_llm",
    ],
)
def test_lista_de_sinais_fora_do_contrato_chega_intacta_ao_pydantic(declarados):
    """Duplicata, desconhecido e tipo errado são erro de forma, não ruído."""
    trecho = "usa um modelo de linguagem próprio e processa imagens"
    entrada = bruto(afirmacao(declarados, trecho))

    resultado = filtrar_sinais_sem_ancora(entrada)

    assert resultado["afirmacoes"][0]["sinais_tecnicos"] == declarados


@pytest.mark.parametrize(
    "declarados",
    [
        ["inferencia_llm", "inferencia_llm"],
        ["sinal_inventado"],
        ["inferencia_llm", "sinal_inventado"],
    ],
)
def test_o_contrato_recusa_o_que_a_porta_deixou_passar(declarados):
    from pydantic import ValidationError

    from radar.contratos import Afirmacao

    filtrada = filtrar_sinais_sem_ancora(
        bruto(afirmacao(declarados, "usa um modelo de linguagem próprio"))
    )

    with pytest.raises(ValidationError):
        Afirmacao(**filtrada["afirmacoes"][0])


def test_uma_lista_valida_continua_passando_pela_porta_de_ancora():
    resultado = filtrar_sinais_sem_ancora(
        bruto(
            afirmacao(
                ["inferencia_llm", "visao_computacional"],
                "usa um modelo de linguagem para responder",
            )
        )
    )

    assert sinais_de(resultado) == ["inferencia_llm"]


def test_a_mistura_de_valido_e_desconhecido_nao_e_reparada_pela_metade():
    entrada = bruto(
        afirmacao(
            ["visao_computacional", "sinal_inventado"], "texto sem âncora nenhuma"
        )
    )

    resultado = filtrar_sinais_sem_ancora(entrada)

    assert resultado["afirmacoes"][0]["sinais_tecnicos"] == [
        "visao_computacional",
        "sinal_inventado",
    ]


def test_zero_sinal_sobrevivente_e_resultado_valido():
    resultado = filtrar_sinais_sem_ancora(
        bruto(afirmacao(["inferencia_llm", "visao_computacional"], "texto neutro"))
    )

    assert sinais_de(resultado) == []


def test_milhoes_de_reais_nao_viram_dados_em_escala():
    perfil = PerfilExtraido.model_validate(
        bruto(
            afirmacao(
                [],
                "A empresa captou R$ 300 milhões em uma rodada de investimento.",
                categoria="momento_e_financiamento",
                polaridade="neutro",
            )
        )
    )

    assert sinais_possivelmente_omitidos(perfil) == ()
    assert completar_sinais_inequivocos(perfil) == perfil


def test_sinal_de_escala_declarado_e_removido_de_valor_monetario():
    entrada = bruto(
        afirmacao(
            ["dados_em_escala"],
            "A empresa captou R$ 300 milhões em uma rodada de investimento.",
            categoria="momento_e_financiamento",
            polaridade="neutro",
        )
    )

    resultado = filtrar_sinais_sem_ancora(entrada)

    assert sinais_de(resultado) == []


def test_sinal_de_escala_declarado_sobre_volume_operacional_e_preservado():
    entrada = bruto(
        afirmacao(
            ["dados_em_escala"],
            "A plataforma processa 4 milhões de imagens por mês.",
        )
    )

    resultado = filtrar_sinais_sem_ancora(entrada)

    assert sinais_de(resultado) == ["dados_em_escala"]


def test_volume_operacional_com_numero_pode_ser_canonizado_como_escala():
    perfil = PerfilExtraido.model_validate(
        bruto(
            afirmacao(
                [],
                "A plataforma processa 4 milhões de imagens por mês.",
                texto="A plataforma processa 4 milhões de imagens por mês.",
            )
        )
    )

    completo = completar_sinais_inequivocos(perfil)

    assert completo.afirmacoes[0].sinais_tecnicos == ("dados_em_escala",)


def test_mencao_negada_nao_e_canonizada_como_sinal():
    perfil = PerfilExtraido.model_validate(
        bruto(
            afirmacao(
                [],
                "A empresa não utiliza visão computacional no produto.",
                texto="A empresa não utiliza visão computacional no produto.",
            )
        )
    )

    assert completar_sinais_inequivocos(perfil) == perfil


def test_uma_afirmacao_sem_sinal_nenhum_passa_intocada():
    entrada = bruto(afirmacao([], "qualquer trecho"))

    assert filtrar_sinais_sem_ancora(entrada) == entrada


@pytest.mark.parametrize(
    "malformado",
    ["texto", None, {"id_startup": 1}, {"afirmacoes": "x"}, {"afirmacoes": [None]}],
)
def test_objeto_malformado_segue_para_a_validacao_sem_reparo(malformado):
    assert filtrar_sinais_sem_ancora(malformado) == malformado


def test_o_helper_puro_devolve_apenas_os_sinais_ancorados():
    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"),
        "usa um modelo de linguagem para gerar respostas",
    ) == ("inferencia_llm",)


# ----------------------------------------------------------------------
# Integração: a porta roda dentro do nó, antes do contrato
# ----------------------------------------------------------------------


def test_o_no_remove_o_sinal_sem_ancora_sem_gastar_o_retry(controlada):  # noqa: F811
    """O sinal cai, a afirmação fica, e o provedor é chamado uma única vez."""
    from radar.agentes.extractor import Extractor
    from tests.test_extractor import estado, perfil_valido

    bruta = perfil_valido(controlada)
    # A terceira afirmação não contém uma carga inequívoca alternativa; assim
    # a remoção do sinal indevido não esconde outro sinal que mereça revisão.
    bruta["afirmacoes"][2]["sinais_tecnicos"] = ["inferencia_llm"]
    provedor = ProvedorSequencial(bruta)

    resultado = Extractor(controlada.base, provedor)(estado(controlada))
    perfil = resultado["perfil_extraido"]

    assert len(provedor.chamadas) == 1
    assert resultado["tentativas_extracao"] == 1
    assert len(perfil.afirmacoes) == len(bruta["afirmacoes"])
    assert perfil.afirmacoes[2].sinais_tecnicos == ()
    assert perfil.afirmacoes[2].trecho_citado == bruta["afirmacoes"][2]["trecho_citado"]


def test_o_no_preserva_o_sinal_ancorado_no_proprio_trecho(controlada):  # noqa: F811
    from radar.agentes.extractor import Extractor
    from tests.test_extractor import estado, perfil_valido

    bruta = perfil_valido(controlada)
    # o trecho desta afirmação cita CUDA e otimização de inferência em GPUs
    bruta["afirmacoes"][2]["sinais_tecnicos"] = ["treinamento_ou_finetuning"]
    provedor = ProvedorSequencial(bruta)

    perfil = Extractor(controlada.base, provedor)(estado(controlada))["perfil_extraido"]

    assert perfil.afirmacoes[2].sinais_tecnicos == ()


def test_o_no_pede_uma_correcao_quando_uma_ancora_explicita_foi_omitida(
    controlada,
):  # noqa: F811
    """A omissão usa o retry já existente; nenhum sinal é inventado localmente."""
    from radar.agentes.extractor import Extractor
    from tests.test_extractor import estado, perfil_valido

    omitida = perfil_valido(controlada)
    omitida["afirmacoes"][0]["trecho_citado"] = (
        "A plataforma usa visão computacional para processar imagens industriais."
    )
    omitida["afirmacoes"][0]["texto"] = (
        "A plataforma usa visão computacional em imagens industriais."
    )
    omitida["afirmacoes"][0]["sinais_tecnicos"] = []

    corrigida = copy.deepcopy(omitida)
    corrigida["afirmacoes"][0]["sinais_tecnicos"] = ["visao_computacional"]
    provedor = ProvedorSequencial(omitida, corrigida)

    perfil = Extractor(controlada.base, provedor)(estado(controlada))["perfil_extraido"]

    assert len(provedor.chamadas) == 2
    assert "possivelmente omitidos" in provedor.ultimo_prompt
    assert perfil.afirmacoes[0].sinais_tecnicos == ("visao_computacional",)


def test_segunda_omissao_canoniza_somente_o_sinal_literal_inequivoco(
    controlada,
):  # noqa: F811
    from radar.agentes.extractor import Extractor
    from tests.test_extractor import estado, perfil_valido

    omitida = perfil_valido(controlada)
    omitida["afirmacoes"][0]["trecho_citado"] = (
        "A plataforma usa visão computacional para processar imagens industriais."
    )
    omitida["afirmacoes"][0]["texto"] = (
        "A plataforma usa visão computacional em imagens industriais."
    )
    omitida["afirmacoes"][0]["sinais_tecnicos"] = []
    provedor = ProvedorSequencial(omitida, copy.deepcopy(omitida))

    perfil = Extractor(controlada.base, provedor)(estado(controlada))["perfil_extraido"]

    assert len(provedor.chamadas) == 2
    assert perfil.afirmacoes[0].sinais_tecnicos == ("visao_computacional",)


def test_detecta_ancora_forte_no_documento_que_nao_chegou_a_nenhuma_afirmacao(
    controlada,
):  # noqa: F811
    documento = controlada.base.carregar_documentos(
        controlada.id_startup, [controlada.ids["site"]]
    )[0].model_copy(
        update={
            "conteudo_texto": (
                "A plataforma usa reconhecimento facial para validar o ponto "
                "dos funcionários nas rotinas de departamento pessoal."
            )
        }
    )
    perfil = PerfilExtraido.model_validate(
        {
            "id_startup": controlada.id_startup,
            "resumo_produto": "A empresa oferece uma plataforma de RH. O produto apoia rotinas internas.",
            "afirmacoes": [
                {
                    "id_afirmacao": 1,
                    "texto": "A empresa oferece uma plataforma de RH.",
                    "categoria": "workflow_profundo",
                    "polaridade": "presenca",
                    "id_documento": documento.id_documento,
                    "trecho_citado": "plataforma usa reconhecimento facial para validar o ponto",
                    "sinais_tecnicos": [],
                }
            ],
        }
    )

    assert sinais_possivelmente_nao_extraidos(perfil, [documento]) == (
        (documento.id_documento, ("visao_computacional",)),
    )


def test_a_remocao_de_sinal_nao_altera_a_contagem_de_afirmacoes():
    """Remover sinal não é derrubar afirmação: a taxa_derrubada não vê isso."""
    entrada = bruto(
        afirmacao(["inferencia_llm"], "apenas inteligência artificial genérica"),
        afirmacao(["visao_computacional"], "processa imagens aéreas", id_afirmacao=2),
    )

    resultado = filtrar_sinais_sem_ancora(entrada)

    assert len(resultado["afirmacoes"]) == len(entrada["afirmacoes"])
    assert [a["id_afirmacao"] for a in resultado["afirmacoes"]] == [1, 2]
    assert sinais_de(resultado, 0) == []
    assert sinais_de(resultado, 1) == ["visao_computacional"]


# ----------------------------------------------------------------------
# Negação explícita fecha a porta para TODO sinal, não só para escala
# ----------------------------------------------------------------------
#
# A guarda de negação existia no módulo, mas só era executada dentro de
# ``_tem_ancora_inequivoca`` — que ``sinais_com_ancora`` só consulta para
# ``dados_em_escala``. Os demais sinais caíam no ramo genérico de substring e
# mantinham o rótulo mesmo quando o próprio trecho citado negava a carga de
# trabalho. Proveniência fechava; lastro semântico, não.


NEGACOES_POR_SINAL = [
    ("inferencia_llm", "a empresa não utiliza modelos de linguagem no produto"),
    ("treinamento_ou_finetuning", "a equipe não usa fine-tuning de modelos"),
    ("voz_fala_ou_transcricao", "o produto não possui transcrição de voz"),
    ("visao_computacional", "a operação não utiliza visão computacional"),
    ("robotica_ou_simulacao", "a fábrica não possui robótica autônoma"),
    ("imagem_medica", "a clínica não utiliza imagem médica digital"),
    ("ciberseguranca_em_escala", "o time não usa detecção de ameaças automatizada"),
    ("machine_learning_classico", "a área não utiliza machine learning"),
    ("agentes_com_acoes_ou_controles", "o sistema não possui agente que executa ações"),
    ("dados_em_escala", "não utiliza os 12 bilhões de registros de volume de dados"),
]


@pytest.mark.parametrize("sinal, trecho", NEGACOES_POR_SINAL)
def test_negacao_explicita_derruba_o_sinal(sinal, trecho):
    assert sinais_com_ancora((sinal,), trecho) == ()


AFIRMACOES_POR_SINAL = [
    ("inferencia_llm", "a empresa usa modelos de linguagem em produção"),
    ("treinamento_ou_finetuning", "a equipe faz fine-tuning de modelos próprios"),
    ("voz_fala_ou_transcricao", "o produto faz transcrição de voz em tempo real"),
    ("visao_computacional", "a operação usa visão computacional nas esteiras"),
    ("robotica_ou_simulacao", "a fábrica opera robótica autônoma em três galpões"),
    ("imagem_medica", "a clínica processa imagem médica de tomografia"),
    ("ciberseguranca_em_escala", "o time faz detecção de ameaças automatizada"),
    ("machine_learning_classico", "a área usa machine learning para previsão"),
]


@pytest.mark.parametrize("sinal, trecho", AFIRMACOES_POR_SINAL)
def test_afirmacao_positiva_preserva_o_sinal(sinal, trecho):
    assert sinais_com_ancora((sinal,), trecho) == (sinal,)


def test_negativa_nao_relacionada_nao_apaga_afirmacao_valida():
    """Só o vocabulário fechado de negação fecha a porta.

    Um "não" comum, distante da carga de trabalho, não pode apagar um sinal que
    o trecho afirma — senão a porta viraria censura de qualquer frase negativa.
    """
    trecho = (
        "a empresa usa modelos de linguagem em produção e ainda não abriu "
        "capital na bolsa"
    )

    assert sinais_com_ancora(("inferencia_llm",), trecho) == ("inferencia_llm",)


def test_escala_mantem_a_ancora_composta_mais_estrita():
    # número + escala + termo de volume, sem termo monetário: qualifica
    assert sinais_com_ancora(
        ("dados_em_escala",), "processa 4 bilhões de registros de dados por mês"
    ) == ("dados_em_escala",)
    # valor monetário não é volume de dados, mesmo com número e escala
    assert sinais_com_ancora(("dados_em_escala",), "captou R$ 300 milhões") == ()
    # sem número, não qualifica
    assert sinais_com_ancora(("dados_em_escala",), "processa bilhões de dados") == ()


def test_sinal_que_exige_numero_continua_exigindo_numero():
    assert sinais_com_ancora(("dados_em_escala",), "volume de dados em terabytes") == ()


@pytest.mark.parametrize(
    "trecho",
    [
        "A empresa NÃO UTILIZA modelos de linguagem no produto",
        "a empresa nao utiliza modelos de linguagem no produto",
        "a empresa não utiliza MODELOS DE LINGUAGEM no produto",
    ],
)
def test_negacao_e_insensivel_a_caixa_e_acento(trecho):
    assert sinais_com_ancora(("inferencia_llm",), trecho) == ()


def test_afirmacao_e_insensivel_a_caixa_e_acento():
    assert sinais_com_ancora(
        ("inferencia_llm",), "A EMPRESA USA MODELOS DE LINGUAGEM"
    ) == ("inferencia_llm",)


# ----------------------------------------------------------------------
# A negação vale para a oração do sinal, não para o trecho inteiro
# ----------------------------------------------------------------------
#
# A porta de negação passou a ser aplicada a todo sinal, mas com escopo largo
# demais: bastava uma frase negativa em qualquer ponto do trecho para derrubar
# todos os sinais. Um trecho real costuma afirmar uma carga de trabalho e negar
# outra na mesma linha — "usa modelos de linguagem, mas não utiliza visão
# computacional" — e ali a negação pertence a uma tecnologia só. A regra passa
# a ser por oração: separadores de frase e conectivos adversativos delimitam o
# alcance da negação; o vocabulário fechado de negação continua o mesmo.


def test_afirmativa_de_llm_sobrevive_a_negativa_de_visao():
    trecho = (
        "A empresa usa modelos de linguagem em produção, mas não utiliza "
        "visão computacional."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_negativa_de_llm_nao_derruba_afirmativa_de_visao():
    trecho = (
        "A empresa não utiliza modelos de linguagem, mas usa visão "
        "computacional nas esteiras."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("visao_computacional",)


def test_varios_sinais_afirmados_sobrevivem_a_uma_negativa_alheia():
    trecho = (
        "Usa modelos de linguagem e transcrição de voz; não possui robótica "
        "autônoma."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "voz_fala_ou_transcricao", "robotica_ou_simulacao"),
        trecho,
    ) == ("inferencia_llm", "voz_fala_ou_transcricao")


def test_sinal_negado_numa_oracao_e_afirmado_em_outra_permanece():
    """Uma oração afirmativa basta para sustentar o sinal."""
    trecho = (
        "No atendimento não utiliza modelos de linguagem. No backoffice usa "
        "modelos de linguagem para triagem."
    )

    assert sinais_com_ancora(("inferencia_llm",), trecho) == ("inferencia_llm",)


def test_sinal_apenas_em_oracao_negada_e_recusado():
    trecho = (
        "A operação é manual e não utiliza visão computacional; a equipe é "
        "pequena."
    )

    assert sinais_com_ancora(("visao_computacional",), trecho) == ()


def test_lista_de_tecnologias_sob_a_mesma_negacao_cai_inteira():
    """Vírgula e 'e' não abrem oração: a negação cobre a enumeração toda."""
    trecho = (
        "A empresa não utiliza visão computacional, robótica autônoma e "
        "imagem médica."
    )

    assert sinais_com_ancora(
        ("visao_computacional", "robotica_ou_simulacao", "imagem_medica"), trecho
    ) == ()


def test_negativa_comum_nao_relacionada_continua_sem_efeito():
    trecho = (
        "A empresa usa modelos de linguagem em produção e ainda não abriu "
        "capital na bolsa."
    )

    assert sinais_com_ancora(("inferencia_llm",), trecho) == ("inferencia_llm",)


@pytest.mark.parametrize(
    "separador",
    ["mas", "porém", "porem", "contudo", "todavia", "entretanto", "enquanto"],
)
def test_conectivos_adversativos_delimitam_a_negacao(separador):
    trecho = (
        f"A empresa usa modelos de linguagem em produção, {separador} não "
        "utiliza visão computacional."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


@pytest.mark.parametrize(
    "trecho",
    [
        "A EMPRESA USA MODELOS DE LINGUAGEM, MAS NÃO UTILIZA VISÃO COMPUTACIONAL",
        "a empresa usa modelos de linguagem, mas nao utiliza visao computacional",
        "A  empresa   usa  modelos de linguagem,  mas  não  utiliza  visão computacional",
    ],
)
def test_escopo_da_negacao_e_estavel_sob_caixa_acento_e_espaco(trecho):
    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_escala_mantem_ancora_composta_com_negacao_por_oracao():
    # afirmativa de escala convive com negativa de outra tecnologia
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Processa 4 bilhões de registros de dados por mês, mas não utiliza "
        "visão computacional.",
    ) == ("dados_em_escala",)
    # a própria escala negada continua caindo
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Não utiliza os 12 bilhões de registros de volume de dados disponíveis.",
    ) == ()
    # valor monetário continua não sendo volume de dados
    assert sinais_com_ancora(
        ("dados_em_escala",), "Captou R$ 300 milhões, mas usa dados próprios."
    ) == ()
    # sem número continua fora
    assert sinais_com_ancora(
        ("dados_em_escala",), "Processa bilhões de dados, mas não usa GPU."
    ) == ()


def test_ordem_de_entrada_dos_sinais_e_preservada():
    trecho = (
        "Usa visão computacional nas esteiras e modelos de linguagem no "
        "atendimento; não possui robótica autônoma."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional", "robotica_ou_simulacao"), trecho
    ) == ("inferencia_llm", "visao_computacional")
    assert sinais_com_ancora(
        ("visao_computacional", "inferencia_llm", "robotica_ou_simulacao"), trecho
    ) == ("visao_computacional", "inferencia_llm")


# ----------------------------------------------------------------------
# Ordem das orações: a negação pode vir antes da afirmação
# ----------------------------------------------------------------------
#
# A régua por oração cobria só o arranjo "afirma, mas nega". Quando o conectivo
# abre a frase — "Embora não utilize X, a empresa usa Y" —, quem separa a
# subordinada negativa da principal afirmativa é a vírgula, e vírgula não abre
# oração (senão enumerações sob a mesma negação se soltariam). Faltavam também
# as formas de subjuntivo/infinitivo que essas construções exigem.


CONECTIVO_NO_INICIO = [
    "Embora não utilize visão computacional, a empresa usa modelos de "
    "linguagem em produção.",
    "Apesar de não utilizar visão computacional, a empresa usa modelos de "
    "linguagem em produção.",
    "Enquanto não utiliza visão computacional, a empresa usa modelos de "
    "linguagem em produção.",
]


@pytest.mark.parametrize("trecho", CONECTIVO_NO_INICIO)
def test_subordinada_negativa_antes_da_principal_afirmativa(trecho):
    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_afirmativa_antes_da_negativa_continua_valendo():
    trecho = (
        "A empresa usa modelos de linguagem em produção, embora não utilize "
        "visão computacional."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_subordinada_intercalada_entre_sujeito_e_verbo():
    trecho = (
        "A empresa, embora não utilize visão computacional, usa modelos de "
        "linguagem."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


@pytest.mark.parametrize(
    "negativa",
    [
        "não utiliza visão computacional",
        "não utilize visão computacional",
        "não utilizar visão computacional",
        "não usa visão computacional",
        "não use visão computacional",
        "não possui visão computacional",
        "não possua visão computacional",
    ],
)
def test_formas_verbais_fechadas_da_negacao(negativa):
    trecho = f"Embora {negativa}, a empresa usa modelos de linguagem."

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_lista_negativa_coordenada_com_nem_permanece_inteira():
    """A vírgula que só continua a coordenação negativa não fecha a oração."""
    trecho = (
        "Embora não utilize visão computacional, robótica nem imagem médica, "
        "a empresa usa modelos de linguagem."
    )

    assert sinais_com_ancora(
        (
            "inferencia_llm",
            "visao_computacional",
            "robotica_ou_simulacao",
            "imagem_medica",
        ),
        trecho,
    ) == ("inferencia_llm",)


def test_lista_negativa_coordenada_sem_conectivo_permanece_inteira():
    trecho = (
        "A empresa não utiliza visão computacional, robótica nem imagem médica."
    )

    assert sinais_com_ancora(
        ("visao_computacional", "robotica_ou_simulacao", "imagem_medica"), trecho
    ) == ()


def test_lista_afirmativa_coordenada_sobrevive_apos_subordinada_negativa():
    trecho = (
        "Embora não utilize robótica autônoma, a empresa usa visão "
        "computacional e modelos de linguagem."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional", "robotica_ou_simulacao"), trecho
    ) == ("inferencia_llm", "visao_computacional")


def test_mesmo_sinal_negado_numa_subordinada_e_afirmado_na_principal():
    trecho = (
        "Embora não utilize modelos de linguagem no atendimento, a empresa usa "
        "modelos de linguagem no backoffice."
    )

    assert sinais_com_ancora(("inferencia_llm",), trecho) == ("inferencia_llm",)


@pytest.mark.parametrize(
    "trecho",
    [
        "EMBORA NÃO UTILIZE VISÃO COMPUTACIONAL, A EMPRESA USA MODELOS DE LINGUAGEM",
        "embora nao utilize visao computacional, a empresa usa modelos de linguagem",
        "Embora  não   utilize  visão computacional,  a  empresa  usa  modelos de linguagem",
        "Embora não utilize visão computacional; a empresa usa modelos de linguagem!",
    ],
)
def test_escopo_invertido_e_estavel_sob_pontuacao_caixa_acento_e_espaco(trecho):
    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional"), trecho
    ) == ("inferencia_llm",)


def test_escala_sobrevive_a_subordinada_negativa_anterior():
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Embora não utilize visão computacional, processa 4 bilhões de "
        "registros de dados por mês.",
    ) == ("dados_em_escala",)
    # a própria escala na subordinada negada continua caindo
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Embora não utilize os 12 bilhões de registros de volume de dados, a "
        "empresa cresce.",
    ) == ()
    # sem número continua fora, mesmo na oração afirmativa
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Embora não utilize visão computacional, processa bilhões de dados.",
    ) == ()


def test_ordem_dos_sinais_preservada_com_conectivo_no_inicio():
    trecho = (
        "Embora não possua robótica autônoma, a empresa usa visão "
        "computacional e modelos de linguagem."
    )

    assert sinais_com_ancora(
        ("inferencia_llm", "visao_computacional", "robotica_ou_simulacao"), trecho
    ) == ("inferencia_llm", "visao_computacional")
    assert sinais_com_ancora(
        ("visao_computacional", "inferencia_llm", "robotica_ou_simulacao"), trecho
    ) == ("visao_computacional", "inferencia_llm")


# ----------------------------------------------------------------------
# Enumeração negativa: a negação atravessa a lista, e só a principal escapa
# ----------------------------------------------------------------------
#
# A vírgula que fecha a subordinada era escolhida olhando só para ``nem``. Uma
# lista negativa ligada por ``e`` ou ``ou`` — a forma mais comum em português —
# era lida como fim da subordinada, e tudo a partir do segundo item escapava da
# negação. Quem distingue "mais um item da enumeração" de "começo da oração
# principal" é o próprio vocabulário fechado de âncoras: item de enumeração
# começa por âncora técnica; oração principal começa por sujeito ou verbo.
#
# Duas propriedades do vocabulário aparecem nas expectativas abaixo e são
# anteriores a este marco: ``imagem`` é âncora de ``visao_computacional`` (logo
# "imagem médica" também sustenta visão), e ``agentes_com_acoes_ou_controles``
# exige ``executa acao``/``guardrail`` — "executam ações" não casa.

QUATRO_SINAIS = (
    "inferencia_llm",
    "visao_computacional",
    "robotica_ou_simulacao",
    "imagem_medica",
)

ENUMERACAO_NEGATIVA = [
    # 1. lista negativa com vírgulas e "e"
    (
        QUATRO_SINAIS,
        "Embora não utilize visão computacional, robótica autônoma e imagem "
        "médica, a empresa usa modelos de linguagem.",
        ("inferencia_llm",),
    ),
    # 2. lista negativa com vírgulas e "ou"
    (
        QUATRO_SINAIS,
        "Embora não utilize visão computacional, robótica autônoma ou imagem "
        "médica, a empresa usa modelos de linguagem.",
        ("inferencia_llm",),
    ),
    # 3. lista negativa com vírgulas e "nem"
    (
        QUATRO_SINAIS,
        "Embora não utilize visão computacional, robótica autônoma nem imagem "
        "médica, a empresa usa modelos de linguagem.",
        ("inferencia_llm",),
    ),
    # 4. três sinais negados seguidos de um afirmativo, com "apesar de"
    (
        QUATRO_SINAIS,
        "Apesar de não utilizar visão computacional, robótica autônoma e "
        "imagem médica, a empresa usa modelos de linguagem.",
        ("inferencia_llm",),
    ),
    # 5. subordinada intercalada com lista técnica negativa
    (
        QUATRO_SINAIS,
        "A empresa, embora não utilize visão computacional, robótica autônoma "
        "e imagem médica, usa modelos de linguagem.",
        ("inferencia_llm",),
    ),
    # 6. principal afirmativa com lista própria separada por vírgula
    (
        ("inferencia_llm", "voz_fala_ou_transcricao", "visao_computacional"),
        "Embora não utilize visão computacional, a empresa usa modelos de "
        "linguagem, transcrição de voz e agentes que executam ações.",
        ("inferencia_llm", "voz_fala_ou_transcricao"),
    ),
    # 6b. mesma forma, com âncora de agentes que existe no vocabulário fechado
    (
        (
            "inferencia_llm",
            "voz_fala_ou_transcricao",
            "agentes_com_acoes_ou_controles",
            "visao_computacional",
        ),
        "Embora não utilize visão computacional, a empresa usa modelos de "
        "linguagem, transcrição de voz e agentes com guardrails.",
        (
            "inferencia_llm",
            "voz_fala_ou_transcricao",
            "agentes_com_acoes_ou_controles",
        ),
    ),
    # 7. enumeração afirmativa sem negação alguma
    (
        ("inferencia_llm", "visao_computacional", "robotica_ou_simulacao"),
        "A empresa usa visão computacional, robótica autônoma e modelos de "
        "linguagem.",
        ("inferencia_llm", "visao_computacional", "robotica_ou_simulacao"),
    ),
    # 8. lista negativa sem conectivo subordinativo
    (
        ("visao_computacional", "robotica_ou_simulacao", "imagem_medica"),
        "A empresa não utiliza visão computacional, robótica autônoma nem "
        "imagem médica.",
        (),
    ),
    # 9. sinal negado numa oração e afirmado em outra
    (
        ("inferencia_llm",),
        "Embora não utilize modelos de linguagem no atendimento, a empresa usa "
        "modelos de linguagem no backoffice.",
        ("inferencia_llm",),
    ),
]


@pytest.mark.parametrize("sinais, trecho, esperado", ENUMERACAO_NEGATIVA)
def test_escopo_da_enumeracao_negativa(sinais, trecho, esperado):
    assert sinais_com_ancora(sinais, trecho) == esperado


def test_enumeracao_negativa_preserva_ordem_e_escala():
    trecho_ordem = (
        "Embora não utilize robótica autônoma e imagem médica, a empresa usa "
        "modelos de linguagem e transcrição de voz."
    )
    assert sinais_com_ancora(
        ("inferencia_llm", "voz_fala_ou_transcricao", "robotica_ou_simulacao"),
        trecho_ordem,
    ) == ("inferencia_llm", "voz_fala_ou_transcricao")
    assert sinais_com_ancora(
        ("voz_fala_ou_transcricao", "inferencia_llm", "robotica_ou_simulacao"),
        trecho_ordem,
    ) == ("voz_fala_ou_transcricao", "inferencia_llm")

    # a regra composta e numérica de escala continua valendo dos dois lados
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Embora não utilize visão computacional, robótica autônoma e imagem "
        "médica, processa 4 bilhões de registros de dados por mês.",
    ) == ("dados_em_escala",)
    assert sinais_com_ancora(
        ("dados_em_escala",),
        "Embora não utilize os 12 bilhões de registros de volume de dados, "
        "robótica autônoma e imagem médica, a empresa cresce.",
    ) == ()
