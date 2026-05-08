import cv2, dlib, numpy as np, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib, matplotlib.pyplot as plt
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/TCC/shape_predictor_68_face_landmarks.dat")

def extrair_features(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()], dtype=float)
    coords /= faces[0].width()
    return coords.flatten()

def carregar_dataset(raiz):
    X, y = [], []
    for nivel in sorted(os.listdir(raiz)):
        pasta = os.path.join(raiz, nivel)
        if not os.path.isdir(pasta):
            continue
        imagens = [f for f in os.listdir(pasta) if f.endswith((".jpg",".png",".jpeg"))]
        print(f"  Nível {nivel}: {len(imagens)} imagens")
        for img_file in tqdm(imagens, desc=f"Nível {nivel}"):
            feat = extrair_features(os.path.join(pasta, img_file))
            if feat is not None:
                X.append(feat)
                y.append(int(nivel))
    return np.array(X), np.array(y)

print("Carregando dataset...")
X, y = carregar_dataset("C:/TCC/ER_Dataset")

print(f"\nTotal carregado: {len(X)} amostras")
print("Dividindo treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

np.save("C:/TCC/X_test.npy", X_test)
np.save("C:/TCC/y_test.npy", y_test)

modelos = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42))
    ])
}

nomes_classes = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]

for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=nomes_classes))

    joblib.dump(modelo, f"C:/TCC/model_{nome}.pkl")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=nomes_classes)
    disp.plot(xticks_rotation=45)
    plt.title(f"Matriz de Confusão — {nome}")
    plt.tight_layout()
    plt.savefig(f"C:/TCC/confusion_{nome}.png")
    plt.close()
    print(f"  Modelo salvo: model_{nome}.pkl")
    print(f"  Matriz salva: confusion_{nome}.png")

print("\nTreinamento concluído!")