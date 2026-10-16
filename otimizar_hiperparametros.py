"""
Script para BUSCAR OS MELHORES HIPERPARÂMETROS dos três modelos (SVM,
Random Forest, MLP), usando GridSearchCV.

O QUE É "HIPERPARÂMETRO"?
São os "botões de ajuste" de um modelo que a gente escolhe ANTES de
treinar (diferente dos pesos internos, que o modelo aprende sozinho).
Por exemplo, no SVM, o parâmetro C controla o quanto o modelo tolera
erros no treino: um C pequeno deixa a fronteira de decisão mais "suave"
(menos propensa a decorar o treino), um C grande deixa a fronteira mais
"apertada" em volta dos exemplos de treino.

O QUE É GridSearchCV?
É uma ferramenta do scikit-learn que testa TODAS as combinações possíveis
de uma lista de valores para cada hiperparâmetro (por isso "grid" =
grade), treinando e validando o modelo várias vezes para cada combinação,
e no final devolve a combinação que teve o melhor resultado médio.

CUIDADO COM VAZAMENTO DE NOVO:
Assim como no treinar.py, NÃO podemos deixar o GridSearchCV validar cada
combinação com uma divisão aleatória simples, porque isso reintroduziria
o mesmo problema de frames quase idênticos ficando em treino e validação
ao mesmo tempo. Por isso, passamos o mesmo StratifiedGroupKFold (a mesma
técnica usada no treinar.py) como a estratégia de validação cruzada do
GridSearchCV, através do parâmetro `cv=`.

Este script:
  1) Carrega o dataset e monta os grupos (igual ao treinar.py).
  2) Separa treino/teste por grupo (igual ao treinar.py) — o GridSearchCV
     só vai mexer DENTRO do treino, para escolher os hiperparâmetros; o
     teste final fica de fora, intocado, para dar uma avaliação honesta.
  3) Roda o GridSearchCV para cada um dos 3 modelos.
  4) Mostra a melhor combinação de cada um e a acurácia no teste.
"""

from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from features import RAIZ, criar_detector_predictor, extrair_features

DATASET = RAIZ / "ER_Dataset"

detector, predictor = criar_detector_predictor()


def carregar_dataset(raiz):
    """Mesma função do treinar.py: carrega imagens, extrai as
    características e guarda também o nome do arquivo (necessário para
    montar os grupos por vídeo/sequência)."""
    X, y, nomes_arquivo = [], [], []
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
                nomes_arquivo.append(img_file.name)
    return np.array(X), np.array(y), np.array(nomes_arquivo)


def extrair_categoria_e_numero(nome_do_arquivo):
    """Mesma lógica do treinar.py: separa 'Engaged_Confused_0020.jpg' em
    categoria='Engaged_Confused' e numero=20."""
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    ultimo_pedaco = partes_do_nome[-1]
    categoria = "_".join(partes_do_nome[:-1])
    numero = int(ultimo_pedaco) if ultimo_pedaco.isdigit() else None
    return categoria, numero


def identificar_grupo_de_cada_imagem(nomes_arquivo):
    """Mesma lógica de agrupamento do treinar.py: frames da mesma
    categoria com números próximos (salto <= 2) formam um único grupo."""
    categorias = []
    numeros = []
    for nome in nomes_arquivo:
        categoria, numero = extrair_categoria_e_numero(nome)
        categorias.append(categoria)
        numeros.append(numero)

    grupo_de_cada_indice = [None] * len(nomes_arquivo)

    categorias_unicas = sorted(set(categorias))
    for categoria_atual in categorias_unicas:
        indices_desta_categoria = [
            indice for indice, cat in enumerate(categorias) if cat == categoria_atual
        ]
        indices_desta_categoria.sort(key=lambda indice: numeros[indice])

        numero_do_grupo_atual = 0
        numero_anterior = None
        for indice in indices_desta_categoria:
            numero_atual = numeros[indice]
            if numero_anterior is not None and numero_atual is not None:
                salto = numero_atual - numero_anterior
                if salto > 2:
                    numero_do_grupo_atual += 1
            nome_do_grupo = f"{categoria_atual}_grupo{numero_do_grupo_atual}"
            grupo_de_cada_indice[indice] = nome_do_grupo
            numero_anterior = numero_atual

    return np.array(grupo_de_cada_indice)


nomes_das_classes = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]


print("Carregando dataset...")
X, y, nomes_arquivo = carregar_dataset(DATASET)
print(f"\nTotal carregado: {len(X)} amostras")

print("Identificando grupos de frames (provável mesmo vídeo/pessoa)...")
grupos = identificar_grupo_de_cada_imagem(nomes_arquivo)
print(f"  Número de grupos distintos encontrados: {len(set(grupos))}")

print("Separando treino/teste por grupo (igual ao treinar.py)...")
divisor_treino_teste = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
indices_treino, indices_teste = next(divisor_treino_teste.split(X, y, groups=grupos))

X_train, X_test = X[indices_treino], X[indices_teste]
y_train, y_test = y[indices_treino], y[indices_teste]
grupos_do_treino = grupos[indices_treino]

print(f"  Imagens de treino: {len(X_train)}")
print(f"  Imagens de teste:  {len(X_test)}")

# Esta é a estratégia de validação cruzada que o GridSearchCV vai usar
# DENTRO do treino, para comparar as combinações de hiperparâmetros.
# n_splits=3 aqui (menor que os 5 do split treino/teste) só para o
# GridSearchCV não demorar demais - cada combinação de hiperparâmetro
# ainda é testada 3 vezes com grupos diferentes.
validacao_cruzada_por_grupo = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)


def rodar_grid_search(nome_do_modelo, modelo_base, grade_de_parametros):
    """
    Função genérica que roda o GridSearchCV para um modelo, imprime a
    melhor combinação de hiperparâmetros encontrada, e avalia essa melhor
    combinação no conjunto de teste (que o GridSearchCV nunca viu).
    """
    print(f"\n{'=' * 70}")
    print(f"GridSearchCV — {nome_do_modelo}")
    print(f"{'=' * 70}")

    busca = GridSearchCV(
        estimator=modelo_base,
        param_grid=grade_de_parametros,
        cv=validacao_cruzada_por_grupo,
        scoring="accuracy",
        n_jobs=-1,
    )
    # O GroupKFold/StratifiedGroupKFold precisa saber o grupo de cada
    # amostra do treino, por isso passamos `groups=grupos_do_treino` aqui.
    busca.fit(X_train, y_train, groups=grupos_do_treino)

    print(f"Melhor combinação de hiperparâmetros: {busca.best_params_}")
    print(f"Acurácia média na validação cruzada (dentro do treino): {busca.best_score_ * 100:.2f}%")

    melhor_modelo = busca.best_estimator_
    previsoes_no_teste = melhor_modelo.predict(X_test)
    acuracia_no_teste = accuracy_score(y_test, previsoes_no_teste)

    print(f"\nAcurácia no conjunto de TESTE (nunca visto pelo GridSearchCV): {acuracia_no_teste * 100:.2f}%")
    print(classification_report(y_test, previsoes_no_teste, target_names=nomes_das_classes))

    return busca.best_params_, acuracia_no_teste


resultados = {}

# ---------------------------------------------------------------------
# SVM: testamos combinações de C (o quanto o modelo tolera erro no
# treino) e gamma (o quão "local"/curvada é a fronteira de decisão do
# kernel RBF). Grade pequena, de propósito, para não demorar demais.
# ---------------------------------------------------------------------
pipeline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True)),
])
grade_svm = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.01, 0.1],
}
resultados["SVM"] = rodar_grid_search("SVM", pipeline_svm, grade_svm)

# ---------------------------------------------------------------------
# Random Forest: testamos n_estimators (quantas árvores compõem a
# floresta - mais árvores costuma estabilizar o resultado, com custo de
# tempo) e max_depth (profundidade máxima de cada árvore - controla o
# quanto cada árvore pode "decorar" detalhes do treino).
# ---------------------------------------------------------------------
modelo_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
grade_rf = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
}
resultados["Random_Forest"] = rodar_grid_search("Random_Forest", modelo_rf, grade_rf)

# ---------------------------------------------------------------------
# MLP (rede neural simples): testamos alpha (força da regularização L2 -
# um valor maior penaliza mais pesos grandes, o que ajuda a evitar que a
# rede decore o treino) e hidden_layer_sizes (quantos "neurônios" tem
# cada camada escondida da rede).
# ---------------------------------------------------------------------
pipeline_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(max_iter=300, random_state=42)),
])
grade_mlp = {
    "clf__alpha": [0.0001, 0.001, 0.01],
    "clf__hidden_layer_sizes": [(128,), (256, 128), (128, 64)],
}
resultados["MLP"] = rodar_grid_search("MLP", pipeline_mlp, grade_mlp)

print(f"\n{'=' * 70}")
print("RESUMO FINAL — melhor combinação e acurácia no teste, por modelo")
print(f"{'=' * 70}")
for nome_do_modelo, (melhores_parametros, acuracia) in resultados.items():
    print(f"{nome_do_modelo}: {melhores_parametros} -> acurácia no teste = {acuracia * 100:.2f}%")
print(f"{'=' * 70}")
