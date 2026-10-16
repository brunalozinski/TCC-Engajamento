"""
Script de TESTE PRÁTICO da demo ao vivo.

O que ele faz:
  1) Liga a webcam e captura alguns frames reais (pede para a pessoa
     ficar de frente para a câmera por um instante).
  2) Usa exatamente a mesma função que a demo usa (`analisar_rosto`, de
     features.py) para extrair as características do rosto.
  3) Faz a predição com o modelo ATUAL (o que a demo carrega hoje:
     model_Random_Forest.pkl, já corrigido quanto ao vazamento de dados).
  4) Faz a MESMA predição com o modelo ANTIGO (guardado em
     backup_v1_2026-09-18/model_Random_Forest.pkl, o modelo contaminado
     pelo vazamento), para comparar lado a lado.

Isso confirma, com uma imagem de verdade, se a demo roda sem erros e como
a predição do modelo novo se compara com a do modelo antigo.

Este script NÃO altera nenhum arquivo do projeto.
"""

import time

import cv2
import joblib
import numpy as np

from features import RAIZ, criar_detector_predictor, analisar_rosto

CAMINHO_MODELO_ATUAL = RAIZ / "model_Random_Forest.pkl"
CAMINHO_MODELO_ANTIGO = RAIZ / "backup_v1_2026-09-18" / "model_Random_Forest.pkl"

NOMES_DAS_CLASSES = ["Muito Baixo (0)", "Baixo (1)", "Alto (2)", "Muito Alto (3)"]


def formatar_probabilidades(modelo, vetor_de_features):
    """
    Recebe um modelo já treinado e um vetor de características de um
    rosto, e devolve um dicionário {nome_da_classe: probabilidade},
    já reorganizado para sempre vir na ordem 0, 1, 2, 3 (o scikit-learn
    guarda as classes na ordem em que apareceram no treino, que pode não
    ser 0,1,2,3 exatamente nessa sequência).
    """
    probabilidades_brutas = modelo.predict_proba([vetor_de_features])[0]
    probabilidade_por_classe = {}
    for indice_da_classe, numero_da_classe in enumerate(modelo.classes_):
        probabilidade_por_classe[int(numero_da_classe)] = probabilidades_brutas[indice_da_classe]
    return probabilidade_por_classe


print("Carregando o detector de rosto, o preditor de landmarks e os dois modelos...")
detector, predictor = criar_detector_predictor()
modelo_atual = joblib.load(CAMINHO_MODELO_ATUAL)
modelo_antigo = joblib.load(CAMINHO_MODELO_ANTIGO)

print("\nLigando a webcam. Aguardando 2 segundos para a câmera estabilizar...")
captura = cv2.VideoCapture(0)
time.sleep(2.0)

vetor_de_features = None
tentativas = 0
MAXIMO_DE_TENTATIVAS = 30  # tenta por até ~30 frames até achar um rosto

while tentativas < MAXIMO_DE_TENTATIVAS:
    ret, frame = captura.read()
    if not ret:
        print("Não foi possível capturar um frame da webcam.")
        break

    vetor_de_features, face, ear = analisar_rosto(frame, detector, predictor)
    tentativas += 1

    if vetor_de_features is not None:
        # Salva o frame usado no teste, para conferência visual.
        cv2.imwrite(str(RAIZ / "teste_demo_frame_capturado.jpg"), frame)
        break

captura.release()

if vetor_de_features is None:
    print(f"\nNenhum rosto foi encontrado em {tentativas} frames capturados.")
    print("Teste encerrado sem conseguir avaliar os modelos.")
else:
    print(f"\nRosto encontrado com sucesso (tentativa {tentativas}).")
    print("Frame salvo em teste_demo_frame_capturado.jpg para conferência.")

    print("\n" + "=" * 70)
    print("PREDIÇÃO COM O MODELO ATUAL (corrigido quanto ao vazamento)")
    print("=" * 70)
    probabilidades_atual = formatar_probabilidades(modelo_atual, vetor_de_features)
    classe_prevista_atual = max(probabilidades_atual, key=probabilidades_atual.get)
    for numero_da_classe in range(4):
        print(f"  {NOMES_DAS_CLASSES[numero_da_classe]}: {probabilidades_atual[numero_da_classe] * 100:.1f}%")
    print(f"  -> Classe prevista: {NOMES_DAS_CLASSES[classe_prevista_atual]}")

    print("\n" + "=" * 70)
    print("PREDIÇÃO COM O MODELO ANTIGO (contaminado pelo vazamento)")
    print("=" * 70)
    probabilidades_antigo = formatar_probabilidades(modelo_antigo, vetor_de_features)
    classe_prevista_antiga = max(probabilidades_antigo, key=probabilidades_antigo.get)
    for numero_da_classe in range(4):
        print(f"  {NOMES_DAS_CLASSES[numero_da_classe]}: {probabilidades_antigo[numero_da_classe] * 100:.1f}%")
    print(f"  -> Classe prevista: {NOMES_DAS_CLASSES[classe_prevista_antiga]}")

    print("\n" + "=" * 70)
    print("COMPARAÇÃO")
    print("=" * 70)
    confianca_atual = probabilidades_atual[classe_prevista_atual]
    confianca_antiga = probabilidades_antigo[classe_prevista_antiga]
    print(f"  Confiança do modelo ATUAL na sua própria previsão:  {confianca_atual * 100:.1f}%")
    print(f"  Confiança do modelo ANTIGO na sua própria previsão: {confianca_antiga * 100:.1f}%")
    print("=" * 70)
