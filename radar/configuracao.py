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
LIMIAR_DERRUBADA = 0.5
MAX_EXTRACOES = 2

# Base de conhecimento NVIDIA (Entregável 2).
CAMINHO_FONTES_NVIDIA = RAIZ_PROJETO / "conhecimento" / "fontes"
MODELO_EMBEDDING_NVIDIA = "nvidia/nemotron-3-embed-1b"
DIMENSAO_EMBEDDING_NVIDIA = 2048
MODELO_RERANK_NVIDIA = "nvidia/llama-nemotron-rerank-vl-1b-v2"

# Constantes iniciais do pipeline de recuperação, avaliáveis e calibráveis;
# não são afirmação de otimalidade.
K_LEXICAL_NVIDIA = 20
K_VETORIAL_NVIDIA = 20
K_RRF = 60
N_CANDIDATOS_RERANK = 20
N_TRECHOS_FINAL = 6
TETO_CARACTERES_CHUNK = 1800
TAMANHO_LOTE_EMBEDDING = 32
