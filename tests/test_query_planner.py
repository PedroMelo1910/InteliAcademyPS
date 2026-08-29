import pytest

from radar.agentes.query_planner import ErroQueryPlanner, QueryPlanner
from radar.contratos import (
    EmpresaCandidata,
    FiltrosEstruturados,
    PlanoConsulta,
    ResultadoRecuperacao,
)


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


def plano_valido() -> PlanoConsulta:
    return PlanoConsulta(
        filtros=FiltrosEstruturados(
            setor="Saúde",
            estagio=["série C"],
            localizacao="São Paulo, SP",
            tamanho_time=["não divulgado"],
        ),
        termos_busca=["saúde"],
        sinais_ia=["inteligência artificial"],
        foco_analise="uso assistivo de IA",
    )


def resultado_vazio(filtros=None) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        empresas=[],
        documentos=[],
        filtros_aplicados=filtros or FiltrosEstruturados(),
    )


def test_saida_estruturada_invalida_e_rejeitada_apos_um_retry(base):
    invalida = {
        "filtros": {},
        "termos_busca": [],
        "sinais_ia": [],
        "foco_analise": "teste",
    }
    provedor = ProvedorSequencial(invalida, invalida)
    planejador = QueryPlanner(base, provedor)
    with pytest.raises(ErroQueryPlanner, match="duas vezes fora do contrato"):
        planejador({"consulta_usuario": "startups de saúde"})
    assert len(provedor.chamadas) == 2
    assert "Falha de validação" in provedor.chamadas[1][-1][1]


def test_falha_do_gemini_nao_fabrica_resultado(base):
    provedor = ProvedorSequencial(ConnectionError("segredo que não deve aparecer"))
    planejador = QueryPlanner(base, provedor)
    estado = {"consulta_usuario": "startups de saúde"}
    with pytest.raises(ErroQueryPlanner) as falha:
        planejador(estado)
    assert "segredo" not in str(falha.value)
    assert "resultado_recuperacao" not in estado
    assert "plano_consulta" not in estado


def test_relaxamento_ocorre_somente_com_plano_e_resultado_vazio(base):
    provedor = ProvedorSequencial()
    planejador = QueryPlanner(base, provedor)
    plano = plano_valido()

    relaxado = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_vazio(plano.filtros),
            "tentativas_relaxamento": 0,
        }
    )
    assert relaxado["tentativas_relaxamento"] == 1
    assert relaxado["plano_consulta"].filtros.estagio is None
    assert relaxado["plano_consulta"].filtros.localizacao is None
    assert relaxado["plano_consulta"].filtros.tamanho_time is None
    assert relaxado["plano_consulta"].filtros.setor == "Saúde"
    assert relaxado["plano_consulta"].termos_busca == plano.termos_busca
    assert relaxado["plano_consulta"].sinais_ia == plano.sinais_ia

    reutilizado = planejador({"plano_consulta": plano})
    assert "tentativas_relaxamento" not in reutilizado

    resultado_com_empresa = ResultadoRecuperacao(
        empresas=[
            EmpresaCandidata(
                id_startup=1,
                nome="Exemplo",
                setor="Saúde",
                estagio="série C",
                localizacao="São Paulo, SP",
                descricao_curta="exemplo",
            )
        ],
        documentos=[],
        filtros_aplicados=plano.filtros,
    )
    nao_relaxado = planejador(
        {"plano_consulta": plano, "resultado_recuperacao": resultado_com_empresa}
    )
    assert "tentativas_relaxamento" not in nao_relaxado
    assert provedor.chamadas == []


def test_segundo_degrau_remove_apenas_setor(base):
    provedor = ProvedorSequencial()
    planejador = QueryPlanner(base, provedor)
    plano = plano_valido()
    saida = planejador(
        {
            "plano_consulta": plano,
            "resultado_recuperacao": resultado_vazio(plano.filtros),
            "tentativas_relaxamento": 1,
        }
    )
    assert saida["tentativas_relaxamento"] == 2
    assert saida["criterios_relaxados"] == ["setor"]
    assert saida["plano_consulta"].filtros.setor is None
    assert saida["plano_consulta"].filtros.estagio == ["série C"]

