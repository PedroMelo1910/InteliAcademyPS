import pytest
from pydantic import ValidationError

from radar.contratos import Classificacao, EstadoRadar


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


def test_as_tres_classes_operacionais_sao_aceitas():
    for classe in ("AI-native", "AI-enabled", "non-AI"):
        modelo = Classificacao.model_validate(classificacao(classe=classe))
        assert modelo.classe == classe


def test_classe_fora_do_enum_e_rejeitada():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(classe="AI-first"))


def test_justificativa_com_uma_frase_e_rejeitada():
    with pytest.raises(ValidationError, match="justificativa"):
        Classificacao.model_validate(
            classificacao(justificativa="A empresa treina modelos próprios.")
        )


def test_justificativa_com_cinco_frases_e_rejeitada():
    with pytest.raises(ValidationError, match="justificativa"):
        Classificacao.model_validate(
            classificacao(
                justificativa=(
                    "A empresa treina modelos. O produto depende deles. "
                    "O time é técnico. A base é proprietária. A escala é relevante."
                )
            )
        )


def test_justificativa_de_duas_a_quatro_frases_e_aceita():
    duas = "A empresa treina modelos próprios. O produto depende deles."
    quatro = (
        "A empresa treina modelos próprios. O produto depende deles. "
        "A base rotulada é proprietária. A entrega ao cliente é o próprio modelo."
    )
    for texto in (duas, quatro):
        assert Classificacao.model_validate(classificacao(justificativa=texto))


def test_justificativa_em_branco_e_rejeitada():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(justificativa="   "))


def test_suporte_vazio_e_rejeitado():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(ids_afirmacoes_suporte=[]))


def test_ids_de_suporte_duplicados_sao_rejeitados():
    with pytest.raises(ValidationError, match="duplicad"):
        Classificacao.model_validate(classificacao(ids_afirmacoes_suporte=[1, 1]))


def test_id_de_suporte_menor_que_um_e_rejeitado():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(ids_afirmacoes_suporte=[0]))


def test_classificacao_rejeita_campos_desconhecidos():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(classe_referencia="AI-native"))


def test_classificacao_nao_aceita_confianca_perfil():
    with pytest.raises(ValidationError):
        Classificacao.model_validate(classificacao(confianca_perfil=0.9))


def test_estado_radar_declara_a_classificacao():
    assert "classificacao" in EstadoRadar.__annotations__


def test_estado_radar_ainda_nao_declara_confianca_perfil():
    assert "confianca_perfil" not in EstadoRadar.__annotations__
