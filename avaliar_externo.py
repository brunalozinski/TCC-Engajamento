"""
Avalia os modelos JA TREINADOS (model_SVM.pkl, model_Random_Forest.pkl,
model_MLP.pkl) no conjunto de teste EXTERNO (ER_Dataset_Externo/), composto
por pessoas gravadas especificamente para este fim -- nenhuma delas aparece
no ER_Dataset original usado no treinamento.

Diferente de treinar.py, este script:
  - NAO treina nada (so carrega os modelos .pkl existentes e faz predict);
  - NAO sobrescreve X_test.npy / y_test.npy nem confusion_*.png existentes
    (esses continuam sendo os resultados do split interno do ER_Dataset,
    usados para documentar no TCC);
  - Gera arquivos com sufixo "_externo" para nao colidir com os anteriores:
      confusion_externo_SVM.png, confusion_externo_Random_Forest.png,
      confusion_externo_MLP.png, relatorio_avaliacao_externa.txt

Uso:
    python avaliar_externo.py
"""

from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import cv2

from features import RAIZ, criar_detector_predictor, extrair_features

DATASET_EXTERNO = RAIZ / "ER_Dataset_Externo"
NOMES_CLASSES = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]

MODELOS_PARA_AVALIAR = ["SVM", "Random_Forest", "MLP"]


def carregar_dataset_externo(raiz):
    """
    Mesma logica de carregar_dataset do treinar.py, mas aplicada a
    ER_Dataset_Externo/. Devolve X, y e o nome do arquivo (que aqui ja
    identifica pessoa e sessao explicitamente, ex:
    'pessoa1_sessao01_classe3_0001.jpg').
    """
    detector, predictor = criar_detector_predictor()
    X, y, nomes_arquivo = [], [], []
    for pasta_nivel in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        nivel = int(pasta_nivel.name)
        imagens = [f for f in pasta_nivel.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print(f"  Nivel {nivel}: {len(imagens)} imagens")
        for img_file in tqdm(imagens, desc=f"Nivel {nivel}"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            feat, _ = extrair_features(img, detector, predictor)
            if feat is not None:
                X.append(feat)
                y.append(nivel)
                nomes_arquivo.append(img_file.name)
    return np.array(X), np.array(y), np.array(nomes_arquivo)


def main():
    print("Carregando conjunto de teste EXTERNO (pessoas novas)...")
    X_ext, y_ext, nomes_arquivo = carregar_dataset_externo(DATASET_EXTERNO)
    print(f"\nTotal de imagens externas carregadas: {len(X_ext)}")

    pessoas = sorted(set(n.split("_")[0] for n in nomes_arquivo))
    print(f"Pessoas distintas no teste externo: {pessoas}")

    linhas_relatorio = []
    linhas_relatorio.append("RELATORIO DE AVALIACAO NO CONJUNTO DE TESTE EXTERNO")
    linhas_relatorio.append("=" * 70)
    linhas_relatorio.append(f"Total de imagens: {len(X_ext)}")
    linhas_relatorio.append(f"Pessoas distintas: {pessoas}")
    linhas_relatorio.append(
        "Nenhuma dessas pessoas aparece no ER_Dataset original (treino)."
    )
    linhas_relatorio.append("")

    for nome in MODELOS_PARA_AVALIAR:
        caminho_modelo = RAIZ / f"model_{nome}.pkl"
        print(f"\nAvaliando {nome} (modelo carregado de {caminho_modelo.name})...")
        modelo = joblib.load(caminho_modelo)

        y_pred = modelo.predict(X_ext)

        relatorio = classification_report(y_ext, y_pred, target_names=NOMES_CLASSES)
        print(relatorio)

        linhas_relatorio.append(f"--- {nome} ---")
        linhas_relatorio.append(relatorio)
        linhas_relatorio.append("")

        cm = confusion_matrix(y_ext, y_pred, labels=sorted(set(y_ext) | set(y_pred)))
        labels_presentes = [NOMES_CLASSES[i] for i in sorted(set(y_ext) | set(y_pred))]
        disp = ConfusionMatrixDisplay(cm, display_labels=labels_presentes)
        disp.plot(xticks_rotation=45)
        plt.title(f"Matriz de Confusao (teste EXTERNO) - {nome}")
        plt.tight_layout()
        caminho_png = RAIZ / f"confusion_externo_{nome}.png"
        plt.savefig(caminho_png)
        plt.close()
        print(f"  Matriz salva: {caminho_png.name}")

    caminho_txt = RAIZ / "relatorio_avaliacao_externa.txt"
    caminho_txt.write_text("\n".join(linhas_relatorio), encoding="utf-8")
    print(f"\nRelatorio completo salvo em: {caminho_txt.name}")
    print(
        "\nNOTA: os arquivos model_*.pkl, X_test.npy, y_test.npy e "
        "confusion_*.png (sem sufixo '_externo') NAO foram alterados."
    )


if __name__ == "__main__":
    main()
