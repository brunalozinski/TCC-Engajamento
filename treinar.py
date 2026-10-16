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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import cv2

from features import RAIZ, criar_detector_predictor, extrair_features

DATASET = RAIZ / "ER_Dataset"

detector, predictor = criar_detector_predictor()


def carregar_dataset(raiz):
    """
    Lê todas as imagens do dataset e extrai as características (landmarks)
    de cada uma. Além do vetor de características (X) e do nível de
    engajamento (y), agora também devolvemos o NOME DO ARQUIVO de cada
    imagem, porque vamos precisar dele para descobrir de qual "vídeo" cada
    imagem provavelmente veio (ver função `identificar_grupo_do_arquivo`
    logo abaixo).
    """
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


# ---------------------------------------------------------------------
# CORREÇÃO DE VAZAMENTO DE DADOS (DATA LEAKAGE)
#
# O ER_Dataset é feito de FRAMES DE VÍDEO. Isso significa que imagens com
# número de sequência vizinho (ex: "..._0020.jpg" e "..._0021.jpg") são,
# na prática, quase a mesma foto tirada com uma fração de segundo de
# diferença: mesma pessoa, mesma pose, mesmo fundo.
#
# Se a gente dividir treino/teste sorteando IMAGEM POR IMAGEM (como o
# train_test_split normal faz), corre o risco de colocar a imagem 0020 no
# treino e a 0021 (quase idêntica) no teste. Nesse caso, o modelo não
# estaria realmente aprendendo a reconhecer engajamento — estaria só
# "reconhecendo" uma imagem que ele já viu quase igual.
#
# A solução é agrupar os frames que pertencem ao mesmo "pedaço de vídeo"
# num único GRUPO, e depois garantir que cada grupo inteiro vá só para o
# treino OU só para o teste, nunca os dois. Isso é feito com o
# GroupShuffleSplit do scikit-learn.
#
# Como o dataset não guarda um ID de vídeo/pessoa explícito, criamos um
# "grupo" artificial a partir do nome do arquivo: separamos os frames por
# categoria original (ex: "Engaged_Confused") e, dentro de cada categoria,
# cortamos um novo grupo sempre que a numeração dá um salto grande (mais
# de 2 números de diferença). Uma sequência de números "andando de perto"
# (0020, 0021, 0022...) vira um grupo só; um salto (de 0022 para 0040, por
# exemplo) indica que provavelmente começou outro pedaço de vídeo.
# ---------------------------------------------------------------------

def extrair_categoria_e_numero(nome_do_arquivo):
    """
    Os arquivos do ER_Dataset têm nomes como 'Engaged_Confused_0020.jpg'.
    Esta função separa isso em duas partes:
      categoria -> 'Engaged_Confused' (tudo antes do número)
      numero    -> 20 (o número de sequência do frame, como inteiro)
    Se o nome não tiver um número no final, devolve numero = None.
    """
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    ultimo_pedaco = partes_do_nome[-1]
    categoria = "_".join(partes_do_nome[:-1])
    numero = int(ultimo_pedaco) if ultimo_pedaco.isdigit() else None
    return categoria, numero


def identificar_grupo_de_cada_imagem(nomes_arquivo):
    """
    Recebe a lista de nomes de arquivo (na mesma ordem de X e y) e devolve
    uma lista de "rótulos de grupo", um por imagem. Duas imagens só têm o
    MESMO rótulo de grupo se estiverem na mesma categoria original E se a
    numeração entre elas for uma sequência próxima (sem saltos grandes).

    Isso é o que permite ao GroupShuffleSplit saber quais imagens "andam
    juntas" (provavelmente vêm do mesmo vídeo) e não podem ser separadas
    entre treino e teste.
    """
    # Primeiro, descobrimos a categoria e o número de cada imagem.
    categorias = []
    numeros = []
    for nome in nomes_arquivo:
        categoria, numero = extrair_categoria_e_numero(nome)
        categorias.append(categoria)
        numeros.append(numero)

    # Para cada categoria, vamos processar as imagens em ordem crescente
    # de número, e criar um novo grupo sempre que houver um salto > 2.
    # Guardamos o resultado num dicionário: índice original -> nome do grupo.
    grupo_de_cada_indice = [None] * len(nomes_arquivo)

    categorias_unicas = sorted(set(categorias))
    for categoria_atual in categorias_unicas:
        # Pega os índices (posições na lista original) de todas as
        # imagens desta categoria.
        indices_desta_categoria = [
            indice for indice, cat in enumerate(categorias) if cat == categoria_atual
        ]
        # Ordena esses índices pelo número de sequência da imagem.
        indices_desta_categoria.sort(key=lambda indice: numeros[indice])

        numero_do_grupo_atual = 0
        numero_anterior = None
        for indice in indices_desta_categoria:
            numero_atual = numeros[indice]
            if numero_anterior is not None and numero_atual is not None:
                salto = numero_atual - numero_anterior
                if salto > 2:
                    # Salto grande: provavelmente começou outro vídeo/pedaço.
                    numero_do_grupo_atual += 1
            nome_do_grupo = f"{categoria_atual}_grupo{numero_do_grupo_atual}"
            grupo_de_cada_indice[indice] = nome_do_grupo
            numero_anterior = numero_atual

    return np.array(grupo_de_cada_indice)


print("Carregando dataset...")
X, y, nomes_arquivo = carregar_dataset(DATASET)

print(f"\nTotal carregado: {len(X)} amostras")

print("Identificando grupos de frames (provável mesmo vídeo/pessoa)...")
grupos = identificar_grupo_de_cada_imagem(nomes_arquivo)
print(f"  Número de grupos distintos encontrados: {len(set(grupos))}")

print("Dividindo treino/teste por GRUPO (frames do mesmo grupo ficam sempre")
print("juntos, para não vazar informação entre treino e teste)...")
# Usamos o StratifiedGroupKFold em vez do GroupShuffleSplit simples.
# A diferença: o GroupShuffleSplit só garante que grupos não se misturem
# entre treino e teste, mas pode (por azar do sorteio) deixar uma classe
# inteira fora do teste, como aconteceu numa tentativa anterior (a classe
# "Muito Alto" ficou com 0 imagens no teste). O StratifiedGroupKFold faz a
# mesma coisa (respeita os grupos), mas também tenta manter a proporção de
# cada classe (0, 1, 2, 3) parecida entre treino e teste.
#
# n_splits=5 significa que o dataset é dividido em 5 "fatias" (cada uma
# ~20% dos grupos). Pegamos a primeira fatia como teste e o resto como
# treino, o que dá a mesma proporção 80/20 de antes.
divisor_por_grupo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
indices_treino, indices_teste = next(divisor_por_grupo.split(X, y, groups=grupos))

X_train, X_test = X[indices_treino], X[indices_teste]
y_train, y_test = y[indices_treino], y[indices_teste]

print(f"  Imagens de treino: {len(X_train)}")
print(f"  Imagens de teste:  {len(X_test)}")

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
