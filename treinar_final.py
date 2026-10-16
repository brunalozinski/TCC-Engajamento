"""
Treina o MODELO FINAL usando 100% do ER_Dataset (sem reservar nenhuma fatia
interna para teste) e avalia esse modelo no conjunto de teste EXTERNO
(ER_Dataset_Externo/), que tem pessoas reais nunca vistas em nenhum momento
do treino.

Por que isso e diferente de treinar.py:
  - treinar.py reserva ~20% do ER_Dataset para teste interno (StratifiedGroupKFold).
    Isso serve para VALIDAR A METODOLOGIA (comparar SVM/RF/MLP, confirmar que a
    correcao do vazamento de dados funciona). Ver doc/relatorio_experimentos_vazamento_dados.md.
  - treinar_final.py usa TODO o ER_Dataset para treinar (mais dados = modelo
    potencialmente melhor), porque agora o teste de verdade e o ER_Dataset_Externo,
    nao uma fatia reservada do mesmo dataset de origem.

Este script NAO sobrescreve nada do que ja existe:
  - NAO mexe em model_SVM.pkl, model_Random_Forest.pkl, model_MLP.pkl
    (esses continuam sendo os modelos da fase de validacao, treinados com 80%)
  - NAO mexe em X_test.npy / y_test.npy / confusion_*.png (sem sufixo)
  - NAO mexe em confusion_externo_*.png / relatorio_avaliacao_externa.txt
    (resultado da fase 2: modelo-80%-testado-no-externo)

Gera, com nomes proprios:
  - model_final_SVM.pkl, model_final_Random_Forest.pkl, model_final_MLP.pkl
  - confusion_final_SVM.png, confusion_final_Random_Forest.png, confusion_final_MLP.png
  - relatorio_final.txt

Uso:
    python treinar_final.py
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import cv2

from features import RAIZ, criar_detector_predictor, extrair_features

DATASET_TREINO = RAIZ / "ER_Dataset"
DATASET_TESTE_EXTERNO = RAIZ / "ER_Dataset_Externo"
NOMES_CLASSES = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]


def carregar_dataset(raiz, detector, predictor):
    """Le todas as imagens de uma pasta (nivel 0-3) e extrai os landmarks."""
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
    detector, predictor = criar_detector_predictor()

    print("Carregando ER_Dataset INTEIRO (100%, sem reservar teste interno)...")
    X_train, y_train, _ = carregar_dataset(DATASET_TREINO, detector, predictor)
    print(f"\nTotal de imagens de treino: {len(X_train)}")

    print("\nCarregando ER_Dataset_Externo (teste final, pessoas nunca vistas)...")
    X_test, y_test, nomes_arquivo_teste = carregar_dataset(DATASET_TESTE_EXTERNO, detector, predictor)
    print(f"\nTotal de imagens de teste externo: {len(X_test)}")

    pessoas = sorted(set(n.split("_")[0] for n in nomes_arquivo_teste))
    print(f"Pessoas distintas no teste externo: {pessoas}")

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

    linhas_relatorio = []
    linhas_relatorio.append("RELATORIO DO MODELO FINAL")
    linhas_relatorio.append("=" * 70)
    linhas_relatorio.append(f"Treino: 100% do ER_Dataset ({len(X_train)} imagens)")
    linhas_relatorio.append(f"Teste:  ER_Dataset_Externo ({len(X_test)} imagens)")
    linhas_relatorio.append(f"Pessoas no teste: {pessoas}")
    linhas_relatorio.append(
        "Nenhuma pessoa do teste aparece no treino -- teste de generalizacao real."
    )
    linhas_relatorio.append("")

    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome} (100% do ER_Dataset)...")
        modelo.fit(X_train, y_train)

        print(f"Avaliando {nome} no teste externo...")
        y_pred = modelo.predict(X_test)

        relatorio = classification_report(y_test, y_pred, target_names=NOMES_CLASSES)
        print(relatorio)

        linhas_relatorio.append(f"--- {nome} ---")
        linhas_relatorio.append(relatorio)
        linhas_relatorio.append("")

        joblib.dump(modelo, RAIZ / f"model_final_{nome}.pkl")

        classes_presentes = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=classes_presentes)
        labels_presentes = [NOMES_CLASSES[i] for i in classes_presentes]
        disp = ConfusionMatrixDisplay(cm, display_labels=labels_presentes)
        disp.plot(xticks_rotation=45)
        plt.title(f"Matriz de Confusao (modelo final, teste externo) - {nome}")
        plt.tight_layout()
        plt.savefig(RAIZ / f"confusion_final_{nome}.png")
        plt.close()

        print(f"  Modelo salvo: model_final_{nome}.pkl")
        print(f"  Matriz salva: confusion_final_{nome}.png")

    caminho_txt = RAIZ / "relatorio_final.txt"
    caminho_txt.write_text("\n".join(linhas_relatorio), encoding="utf-8")
    print(f"\nRelatorio completo salvo em: {caminho_txt.name}")
    print(
        "\nNOTA: model_*.pkl (sem sufixo 'final'), X_test.npy, y_test.npy e "
        "confusion_*.png/confusion_externo_*.png NAO foram alterados."
    )


if __name__ == "__main__":
    main()
