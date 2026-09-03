"""
Treina os três modelos (SVM, Random Forest, MLP) para classificar o nível
de engajamento a partir dos 68 pontos faciais.

A extração de características vem de features.py, o MESMO módulo usado pela
demo ao vivo, para que treino e uso real fiquem sempre iguais.
"""

from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import cv2

from features import RAIZ, criar_detector_predictor, extrair_features

DATASET = RAIZ / "ER_Dataset"

detector, predictor = criar_detector_predictor()


def carregar_dataset(raiz):
    X, y = [], []
    for pasta_nivel in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        nivel = int(pasta_nivel.name)
        imagens = [f for f in pasta_nivel.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print(f"  Nível {nivel}: {len(imagens)} imagens")
        for img_file in tqdm(imagens, desc=f"Nível {nivel}"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            feat, _ = extrair_features(img, detector, predictor)
            if feat is not None:
                X.append(feat)
                y.append(nivel)
    return np.array(X), np.array(y)


print("Carregando dataset...")
X, y = carregar_dataset(DATASET)

print(f"\nTotal carregado: {len(X)} amostras")
print("Dividindo treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

np.save(RAIZ / "X_test.npy", X_test)
np.save(RAIZ / "y_test.npy", y_test)

modelos = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)),
    ]),
}

nomes_classes = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]

for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=nomes_classes))

    joblib.dump(modelo, RAIZ / f"model_{nome}.pkl")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=nomes_classes)
    disp.plot(xticks_rotation=45)
    plt.title(f"Matriz de Confusão — {nome}")
    plt.tight_layout()
    plt.savefig(RAIZ / f"confusion_{nome}.png")
    plt.close()
    print(f"  Modelo salvo: model_{nome}.pkl")
    print(f"  Matriz salva: confusion_{nome}.png")

print("\nTreinamento concluído!")
