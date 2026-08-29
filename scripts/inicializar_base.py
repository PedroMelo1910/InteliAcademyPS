from radar.base_startups import inicializar_banco
from radar.configuracao import CAMINHO_BANCO, CAMINHO_DADOS_CURADOS


if __name__ == "__main__":
    inicializar_banco(CAMINHO_BANCO, CAMINHO_DADOS_CURADOS)
    print(f"Base validada e inicializada em: {CAMINHO_BANCO}")

