from pathlib import Path


class ErroConfiguracao(RuntimeError):
    """Configuração local inválida, sem incluir valores de credenciais."""


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_DADOS_CURADOS = RAIZ_PROJETO / "dados" / "base"
CAMINHO_BANCO = RAIZ_PROJETO / "dados" / "radar.db"
CAMINHO_CHECKPOINTS = RAIZ_PROJETO / "dados" / "checkpoints.db"

MODELO_GEMINI = "gemini-3.5-flash-lite"
TETO_DOCUMENTOS_DESCOBERTA = 20
TETO_RELAXAMENTO = 2
