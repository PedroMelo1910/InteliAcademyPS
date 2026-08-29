from radar.agentes.roteadores import rotear_r1
from radar.contratos import (
    EmpresaCandidata,
    FiltrosEstruturados,
    ResultadoRecuperacao,
)


def resultado(empresas=None):
    return ResultadoRecuperacao(
        empresas=empresas or [],
        documentos=[],
        filtros_aplicados=FiltrosEstruturados(),
    )


def empresa():
    return EmpresaCandidata(
        id_startup=1,
        nome="Empresa",
        setor="Saúde",
        estagio="seed",
        localizacao="São Paulo, SP",
        descricao_curta="Descrição",
    )


def test_zero_candidatas_dispara_relaxar():
    assert rotear_r1(
        {"resultado_recuperacao": resultado(), "tentativas_relaxamento": 0}
    ) == "relaxar"


def test_teto_de_relaxamento_dispara_sem_resultado():
    assert rotear_r1(
        {"resultado_recuperacao": resultado(), "tentativas_relaxamento": 2}
    ) == "sem_resultado"


def test_descoberta_com_candidatas_dispara_candidatas_prontas():
    assert rotear_r1(
        {"resultado_recuperacao": resultado([empresa()]), "tentativas_relaxamento": 0}
    ) == "candidatas_prontas"


def test_startup_selecionada_tem_prioridade_e_dispara_analisar():
    assert rotear_r1(
        {
            "startup_selecionada": 1,
            "resultado_recuperacao": resultado(),
            "tentativas_relaxamento": 2,
        }
    ) == "analisar"

