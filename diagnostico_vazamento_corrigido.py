"""
Script de DIAGNÓSTICO (versão 2) para confirmar se a correção de vazamento
de dados aplicada no treinar.py realmente resolveu o problema.

O QUE ESTE SCRIPT FAZ:
Repete o mesmo teste do 1-NN (classificador "burro" que só copia a
resposta do vizinho mais parecido do treino) que já fizemos antes, mas
agora usando a divisão treino/teste POR GRUPO (StratifiedGroupKFold), a
mesma lógica que foi colocada no treinar.py.

Se a correção funcionou, a acurácia do 1-NN deve CAIR bastante em relação
ao resultado anterior (96,21% com a divisão antiga, contaminada), ficando
mais perto do "chute aleatório" entre as 4 classes.

Este script NÃO altera nenhum arquivo do projeto, só imprime o resultado.
"""

from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from features import RAIZ, criar_detector_predictor, extrair_features

PASTA_DATASET = RAIZ / "ER_Dataset"


def carregar_dataset_com_nomes(pasta_raiz):
    """Igual ao script de diagnóstico anterior: carrega imagens, extrai
    características e guarda o nome de cada arquivo."""
    lista_de_features = []
    lista_de_rotulos = []
    lista_de_nomes_arquivo = []

    detector, predictor = criar_detector_predictor()

    subpastas_dos_niveis = sorted(p for p in Path(pasta_raiz).iterdir() if p.is_dir())

    for pasta_do_nivel in subpastas_dos_niveis:
        nivel_de_engajamento = int(pasta_do_nivel.name)
        arquivos_de_imagem = [
            arquivo for arquivo in pasta_do_nivel.iterdir()
            if arquivo.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        print(f"  Nível {nivel_de_engajamento}: {len(arquivos_de_imagem)} imagens")

        for caminho_da_imagem in tqdm(arquivos_de_imagem, desc=f"Nível {nivel_de_engajamento}"):
            imagem = cv2.imread(str(caminho_da_imagem))
            if imagem is None:
                continue
            vetor_de_features, _ = extrair_features(imagem, detector, predictor)
            if vetor_de_features is None:
                continue
            lista_de_features.append(vetor_de_features)
            lista_de_rotulos.append(nivel_de_engajamento)
            lista_de_nomes_arquivo.append(caminho_da_imagem.name)

    return (
        np.array(lista_de_features),
        np.array(lista_de_rotulos),
        np.array(lista_de_nomes_arquivo),
    )


def extrair_categoria_e_numero(nome_do_arquivo):
    """Mesma lógica usada no treinar.py: separa 'Engaged_Confused_0020.jpg'
    em categoria='Engaged_Confused' e numero=20."""
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    ultimo_pedaco = partes_do_nome[-1]
    categoria = "_".join(partes_do_nome[:-1])
    numero = int(ultimo_pedaco) if ultimo_pedaco.isdigit() else None
    return categoria, numero


def identificar_grupo_de_cada_imagem(nomes_arquivo):
    """Mesma lógica de agrupamento do treinar.py: frames da mesma categoria
    com números próximos (salto <= 2) formam um único grupo, que depois
    será mantido inteiro do lado do treino OU do teste."""
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


print("=" * 70)
print("DIAGNÓSTICO DE VAZAMENTO (VERSÃO 2) — DIVISÃO CORRIGIDA POR GRUPO")
print("=" * 70)

print("\nPasso 1: carregando o dataset e extraindo as características...")
todas_as_features, todos_os_rotulos, todos_os_nomes_de_arquivo = carregar_dataset_com_nomes(PASTA_DATASET)
print(f"\nTotal de imagens carregadas com sucesso: {len(todas_as_features)}")

print("\nPasso 2: identificando grupos (frames prováveis do mesmo vídeo)...")
grupos = identificar_grupo_de_cada_imagem(todos_os_nomes_de_arquivo)
print(f"  Número de grupos distintos encontrados: {len(set(grupos))}")

print("\nPasso 3: dividindo treino/teste com StratifiedGroupKFold")
print("(mesma técnica agora usada no treinar.py corrigido)...")
divisor_por_grupo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
indices_treino, indices_teste = next(
    divisor_por_grupo.split(todas_as_features, todos_os_rotulos, groups=grupos)
)

X_treino = todas_as_features[indices_treino]
y_treino = todos_os_rotulos[indices_treino]
X_teste = todas_as_features[indices_teste]
y_teste = todos_os_rotulos[indices_teste]

print(f"  Imagens de treino: {len(X_treino)}")
print(f"  Imagens de teste:  {len(X_teste)}")

print("\nPasso 4: treinando o classificador 1-NN (1 vizinho mais próximo)")
classificador_1nn = KNeighborsClassifier(n_neighbors=1)
classificador_1nn.fit(X_treino, y_treino)
previsoes_do_1nn = classificador_1nn.predict(X_teste)

acuracia_do_1nn = accuracy_score(y_teste, previsoes_do_1nn)
print(f"\nACURÁCIA DO 1-NN NO TESTE (divisão corrigida por grupo): {acuracia_do_1nn * 100:.2f}%")

nomes_das_classes = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]
print("\nRelatório detalhado do 1-NN por classe:")
print(classification_report(y_teste, previsoes_do_1nn, target_names=nomes_das_classes))

print("=" * 70)
print("COMPARAÇÃO COM O RESULTADO ANTERIOR (divisão aleatória, contaminada)")
print("=" * 70)
print("  1-NN com divisão aleatória (train_test_split):   96.21%  <- vazamento")
print(f"  1-NN com divisão por grupo (StratifiedGroupKFold): {acuracia_do_1nn * 100:.2f}%")
print("=" * 70)
