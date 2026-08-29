import pytest
from pydantic import ValidationError

from radar.contratos import (
    CATEGORIAS_ESTRUTURAIS,
    LIMITE_TRECHO_CITADO,
    MINIMO_CARACTERES_TRECHO_CITADO,
    MINIMO_PALAVRAS_TRECHO_CITADO,
    Afirmacao,
    EstadoRadar,
    PerfilExtraido,
    contar_frases,
    normalizar_texto_citavel,
)


CATEGORIAS_ESPERADAS = (
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
)


def afirmacao(**ajustes) -> dict:
    base = {
        "id_afirmacao": 1,
        "texto": "A empresa mantém um banco proprietário de imagens rotuladas.",
        "categoria": "dados_proprietarios",
        "polaridade": "presenca",
        "id_documento": 7,
        "trecho_citado": "banco proprietário de imagens rotuladas",
    }
    base.update(ajustes)
    return base


def perfil(**ajustes) -> dict:
    base = {
        "id_startup": 3,
        "resumo_produto": (
            "A empresa vende inspeção visual automatizada. Atende fábricas no Brasil."
        ),
        "afirmacoes": [afirmacao()],
    }
    base.update(ajustes)
    return base


def test_as_dez_categorias_da_arquitetura_sao_aceitas():
    for categoria in CATEGORIAS_ESPERADAS:
        polaridade = "presenca" if categoria in CATEGORIAS_ESTRUTURAIS else "neutro"
        modelo = Afirmacao.model_validate(
            afirmacao(categoria=categoria, polaridade=polaridade)
        )
        assert modelo.categoria == categoria
    assert CATEGORIAS_ESTRUTURAIS == frozenset(CATEGORIAS_ESPERADAS[:4])


def test_categoria_fora_do_enum_e_rejeitada():
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(categoria="parceria_estrategica"))


def test_polaridade_fora_do_enum_e_rejeitada():
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(polaridade="ausencia"))


def test_categoria_nao_estrutural_exige_polaridade_neutra():
    with pytest.raises(ValidationError, match="neutro"):
        Afirmacao.model_validate(
            afirmacao(categoria="momento_e_financiamento", polaridade="ausencia_explicita")
        )
    aceita = Afirmacao.model_validate(
        afirmacao(categoria="momento_e_financiamento", polaridade="neutro")
    )
    assert aceita.polaridade == "neutro"


def test_ausencia_explicita_e_permitida_nas_categorias_estruturais():
    modelo = Afirmacao.model_validate(
        afirmacao(categoria="otimizacao_tecnica", polaridade="ausencia_explicita")
    )
    assert modelo.polaridade == "ausencia_explicita"


def test_trecho_citado_acima_do_limite_e_rejeitado():
    assert LIMITE_TRECHO_CITADO == 300
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(trecho_citado="a" * (LIMITE_TRECHO_CITADO + 1)))
    trecho_no_limite = "dados " * 50
    aceita = Afirmacao.model_validate(afirmacao(trecho_citado=trecho_no_limite))
    assert len(aceita.trecho_citado) == LIMITE_TRECHO_CITADO


def test_trecho_citado_exige_conteudo_minimamente_significativo():
    assert MINIMO_CARACTERES_TRECHO_CITADO == 12
    assert MINIMO_PALAVRAS_TRECHO_CITADO == 3
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(trecho_citado="a"))
    with pytest.raises(ValidationError, match="3 palavras"):
        Afirmacao.model_validate(afirmacao(trecho_citado="NVIDIA acelerada"))
    aceita = Afirmacao.model_validate(
        afirmacao(trecho_citado="usa NVIDIA CUDA")
    )
    assert aceita.trecho_citado == "usa NVIDIA CUDA"


def test_trecho_citado_preserva_o_texto_literal():
    literal = "  o banco   proprietário  "
    modelo = Afirmacao.model_validate(afirmacao(trecho_citado=literal))
    assert modelo.trecho_citado == literal


def test_texto_da_afirmacao_e_uma_unica_frase():
    with pytest.raises(ValidationError, match="uma frase"):
        Afirmacao.model_validate(
            afirmacao(texto="A empresa treina modelos. Também contrata engenheiros.")
        )
    with pytest.raises(ValidationError, match="pontuação"):
        Afirmacao.model_validate(afirmacao(texto="A empresa treina modelos"))


def test_contagem_de_frases_tolera_abreviacoes_decimais_listas_e_reticencias():
    assert contar_frases("A Dr. Consulta opera clínicas populares no Brasil.") == 1
    assert contar_frases("A Movile S.A. contratou engenheiros de ML.") == 1
    assert contar_frases("A empresa usa NVIDIA A100 etc. para treinar modelos.") == 1
    assert contar_frases("A versão 4.5 reduziu a latência.") == 1
    assert contar_frases(
        "A empresa atua em três frentes: 1. dados 2. modelos 3. inferência."
    ) == 1
    assert contar_frases("A empresa... ainda está expandindo.") == 1
    assert contar_frases(
        'O CEO afirmou: "não usamos API externa." A empresa mantém stack própria.'
    ) == 2


def test_campos_de_texto_em_branco_sao_rejeitados():
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(trecho_citado="   "))
    with pytest.raises(ValidationError):
        Afirmacao.model_validate(afirmacao(texto="   "))


def test_ids_de_afirmacao_devem_ser_sequenciais_a_partir_de_um():
    afirmacoes = [afirmacao(id_afirmacao=1), afirmacao(id_afirmacao=3)]
    with pytest.raises(ValidationError, match="sequencial"):
        PerfilExtraido.model_validate(perfil(afirmacoes=afirmacoes))
    fora_de_ordem = [afirmacao(id_afirmacao=2), afirmacao(id_afirmacao=1)]
    with pytest.raises(ValidationError, match="sequencial"):
        PerfilExtraido.model_validate(perfil(afirmacoes=fora_de_ordem))
    validas = [afirmacao(id_afirmacao=1), afirmacao(id_afirmacao=2)]
    assert len(PerfilExtraido.model_validate(perfil(afirmacoes=validas)).afirmacoes) == 2


def test_perfil_exige_de_uma_a_vinte_afirmacoes():
    with pytest.raises(ValidationError):
        PerfilExtraido.model_validate(perfil(afirmacoes=[]))
    excesso = [afirmacao(id_afirmacao=indice) for indice in range(1, 22)]
    with pytest.raises(ValidationError):
        PerfilExtraido.model_validate(perfil(afirmacoes=excesso))
    limite = [afirmacao(id_afirmacao=indice) for indice in range(1, 21)]
    assert len(PerfilExtraido.model_validate(perfil(afirmacoes=limite)).afirmacoes) == 20


def test_resumo_produto_exige_duas_ou_tres_frases():
    with pytest.raises(ValidationError, match="frases"):
        PerfilExtraido.model_validate(perfil(resumo_produto="A empresa vende inspeção visual."))
    with pytest.raises(ValidationError, match="frases"):
        PerfilExtraido.model_validate(
            perfil(resumo_produto="Uma. Duas. Três. Quatro.")
        )
    aceito = PerfilExtraido.model_validate(
        perfil(resumo_produto="Uma frase. Outra frase. E a terceira.")
    )
    assert aceito.resumo_produto.endswith("terceira.")
    com_abreviacao = PerfilExtraido.model_validate(
        perfil(
            resumo_produto=(
                "A Dr. Consulta opera clínicas populares. "
                "A empresa atende pacientes no Brasil. "
                "A plataforma organiza a jornada de cuidado."
            )
        )
    )
    assert contar_frases(com_abreviacao.resumo_produto) == 3


def test_perfil_rejeita_campos_desconhecidos():
    with pytest.raises(ValidationError):
        PerfilExtraido.model_validate(perfil(classe_referencia="AI-native"))


def test_estado_radar_declara_os_campos_do_extractor():
    anotacoes = EstadoRadar.__annotations__
    assert "perfil_extraido" in anotacoes
    assert "tentativas_extracao" in anotacoes


def test_normalizacao_de_proveniencia_so_ajusta_caixa_e_espacos():
    assert normalizar_texto_citavel("  A  Empresa\ntreina\tmodelos ") == "a empresa treina modelos"
    assert normalizar_texto_citavel("modelos próprios") != normalizar_texto_citavel(
        "modelos proprios"
    )
