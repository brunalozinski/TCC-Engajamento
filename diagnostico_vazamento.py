"""
Script de DIAGNÓSTICO para investigar vazamento de dados (data leakage)
entre treino e teste no dataset ER_Dataset.

CONTEXTO DO PROBLEMA:
O ER_Dataset foi montado copiando imagens do dataset original do Kaggle
("Student-engagement-dataset"). Nesse dataset original, os arquivos de
cada categoria têm nomes com números sequenciais, por exemplo:

    Confused/0020.jpg
    Confused/0021.jpg
    Confused/0022.jpg

Esses números sequenciais são um forte indício de que as imagens vêm de
frames extraídos de vídeo. Ou seja, a imagem 0020 e a imagem 0021
provavelmente mostram a MESMA pessoa, na MESMA pose, com uma fração de
segundo de diferença entre uma e outra — são quase cópias uma da outra.

O treinamento atual (treinar.py) divide o dataset em treino e teste com
train_test_split, escolhendo imagens ALEATORIAMENTE, sem se preocupar em
manter frames vizinhos (da mesma pessoa/vídeo) sempre do mesmo lado.
Isso pode fazer com que a imagem 0020 caia no treino e a imagem 0021,
quase idêntica, caia no teste. Nesse caso, o modelo não estaria
aprendendo a reconhecer "engajamento" de verdade — estaria apenas
reconhecendo um rosto/pose que ele já viu (quase) igual no treino.

COMO VAMOS TESTAR ISSO:
Vamos usar um classificador extremamente simples chamado "1-NN"
(1-Nearest-Neighbor = "1 vizinho mais próximo"). Esse classificador NÃO
aprende nenhum padrão: para cada imagem de teste, ele apenas procura,
entre TODAS as imagens de treino, qual é a mais parecida (mais próxima
em termos de distância matemática) e copia a resposta dela.

Se esse classificador "burro" (que só decora e copia) já acertar uma
porcentagem muito alta (perto de 95%), isso é um forte sinal de que o
teste contém imagens quase idênticas às do treino — ou seja, vazamento
de dados. Um classificador de verdade (SVM, Random Forest, MLP) que
tiver uma acurácia parecida com a do 1-NN também pode estar "decorando"
em vez de aprendendo o padrão real de engajamento.

Este script NÃO altera nenhum arquivo do projeto. Ele só imprime um
diagnóstico no terminal.
"""

from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from features import RAIZ, criar_detector_predictor, extrair_features

# Pasta onde estão as imagens já organizadas por nível de engajamento
# (0 = Muito Baixo, 1 = Baixo, 2 = Alto, 3 = Muito Alto).
PASTA_DATASET = RAIZ / "ER_Dataset"


def carregar_dataset_com_nomes(pasta_raiz):
    """
    Lê todas as imagens do dataset, extrai as características (landmarks
    normalizados) de cada rosto encontrado, e devolve três listas:

      lista_de_features   -> os vetores de números que descrevem cada rosto
      lista_de_rotulos     -> o nível de engajamento (0, 1, 2 ou 3) de cada imagem
      lista_de_nomes_arquivo -> o nome do arquivo de cada imagem (para investigar
                                 depois se imagens "vizinhas" pelo nome foram
                                 separadas entre treino e teste)

    Esta função é parecida com a função carregar_dataset() do treinar.py,
    mas guarda também o nome do arquivo, que o treinar.py descarta.
    """
    lista_de_features = []
    lista_de_rotulos = []
    lista_de_nomes_arquivo = []

    detector, predictor = criar_detector_predictor()

    # Cada subpasta (0, 1, 2, 3) representa um nível de engajamento.
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
                # Não conseguiu abrir o arquivo de imagem, então pula ele.
                continue

            vetor_de_features, _ = extrair_features(imagem, detector, predictor)
            if vetor_de_features is None:
                # Não conseguiu achar um rosto nesta imagem, então pula ela.
                continue

            lista_de_features.append(vetor_de_features)
            lista_de_rotulos.append(nivel_de_engajamento)
            lista_de_nomes_arquivo.append(caminho_da_imagem.name)

    return (
        np.array(lista_de_features),
        np.array(lista_de_rotulos),
        np.array(lista_de_nomes_arquivo),
    )


def extrair_numero_da_sequencia(nome_do_arquivo):
    """
    Os arquivos do ER_Dataset têm nomes como:

        'Engaged_Confused_0020.jpg'
        'Not Engaged_Drowsy_0018.jpg'

    Os últimos 4 dígitos antes da extensão são o número de sequência do
    frame dentro do vídeo original. Esta função extrai esse número.
    Se não conseguir extrair (nome em formato inesperado), devolve None.
    """
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    ultimo_pedaco = partes_do_nome[-1]
    if ultimo_pedaco.isdigit():
        return int(ultimo_pedaco)
    return None


def extrair_categoria_original(nome_do_arquivo):
    """
    Devolve o nome da categoria original (tudo antes do número de
    sequência), por exemplo 'Engaged_Confused' ou 'Not Engaged_Drowsy'.
    Isso identifica de qual vídeo/pasta original a imagem veio.
    """
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    return "_".join(partes_do_nome[:-1])


def contar_pares_vizinhos_separados(nomes_treino, nomes_teste):
    """
    Investiga quantas imagens do TESTE têm uma "vizinha de sequência"
    (mesmo prefixo de categoria e número de frame a uma distância de até
    2 quadros) no TREINO. Isso é uma evidência extra de vazamento: se o
    frame 0020 está no treino e o frame 0021 (quase idêntico) está no
    teste, contamos isso como um "par vizinho separado".
    """
    # Organiza as imagens de treino por categoria original, guardando só
    # os números de sequência de cada uma.
    numeros_de_treino_por_categoria = {}
    for nome in nomes_treino:
        categoria = extrair_categoria_original(nome)
        numero = extrair_numero_da_sequencia(nome)
        if numero is None:
            continue
        numeros_de_treino_por_categoria.setdefault(categoria, set()).add(numero)

    total_de_imagens_teste_com_vizinho_no_treino = 0

    for nome in nomes_teste:
        categoria = extrair_categoria_original(nome)
        numero = extrair_numero_da_sequencia(nome)
        if numero is None:
            continue

        numeros_do_treino_nesta_categoria = numeros_de_treino_por_categoria.get(categoria, set())

        # Considera "vizinho" qualquer frame a até 2 posições de distância.
        tem_vizinho_proximo = False
        for distancia in (1, 2):
            if (numero - distancia) in numeros_do_treino_nesta_categoria:
                tem_vizinho_proximo = True
            if (numero + distancia) in numeros_do_treino_nesta_categoria:
                tem_vizinho_proximo = True

        if tem_vizinho_proximo:
            total_de_imagens_teste_com_vizinho_no_treino += 1

    return total_de_imagens_teste_com_vizinho_no_treino


# ---------------------------------------------------------------------
# INÍCIO DO DIAGNÓSTICO
# ---------------------------------------------------------------------

print("=" * 70)
print("DIAGNÓSTICO DE VAZAMENTO DE DADOS (DATA LEAKAGE)")
print("=" * 70)

print("\nPasso 1: carregando o dataset e extraindo as características...")
todas_as_features, todos_os_rotulos, todos_os_nomes_de_arquivo = carregar_dataset_com_nomes(PASTA_DATASET)
print(f"\nTotal de imagens carregadas com sucesso: {len(todas_as_features)}")

print("\nPasso 2: dividindo em treino e teste EXATAMENTE como o treinar.py faz")
print("(mesmo random_state=42, mesma proporção 80/20, estratificado por nível)...")

# Para conseguir rastrear os nomes de arquivo junto com a divisão, usamos
# um vetor de índices (0, 1, 2, 3, ...) e aplicamos o mesmo split nele.
indices_de_todas_as_imagens = np.arange(len(todas_as_features))

indices_treino, indices_teste = train_test_split(
    indices_de_todas_as_imagens,
    test_size=0.2,
    random_state=42,
    stratify=todos_os_rotulos,
)

X_treino = todas_as_features[indices_treino]
y_treino = todos_os_rotulos[indices_treino]
nomes_treino = todos_os_nomes_de_arquivo[indices_treino]

X_teste = todas_as_features[indices_teste]
y_teste = todos_os_rotulos[indices_teste]
nomes_teste = todos_os_nomes_de_arquivo[indices_teste]

print(f"  Imagens de treino: {len(X_treino)}")
print(f"  Imagens de teste:  {len(X_teste)}")

print("\nPasso 3: evidência extra pelo nome do arquivo")
print("Verificando quantas imagens do TESTE têm um frame vizinho (mesmo vídeo,")
print("número de sequência próximo) que ficou no TREINO...")

quantidade_com_vizinho_no_treino = contar_pares_vizinhos_separados(nomes_treino, nomes_teste)
percentual_com_vizinho = 100.0 * quantidade_com_vizinho_no_treino / len(nomes_teste)

print(f"  {quantidade_com_vizinho_no_treino} de {len(nomes_teste)} imagens de teste "
      f"({percentual_com_vizinho:.1f}%) têm um frame muito próximo no treino.")

print("\nPasso 4: treinando o classificador 1-NN (1 vizinho mais próximo)")
print("Esse classificador NÃO aprende nenhum padrão de engajamento: ele só")
print("procura, para cada imagem de teste, a imagem de treino mais parecida,")
print("e copia o rótulo dela.")

classificador_1nn = KNeighborsClassifier(n_neighbors=1)
classificador_1nn.fit(X_treino, y_treino)
previsoes_do_1nn = classificador_1nn.predict(X_teste)

acuracia_do_1nn = accuracy_score(y_teste, previsoes_do_1nn)

print(f"\nACURÁCIA DO 1-NN NO TESTE: {acuracia_do_1nn * 100:.2f}%")

nomes_das_classes = ["Muito Baixo", "Baixo", "Alto", "Muito Alto"]
print("\nRelatório detalhado do 1-NN por classe:")
print(classification_report(y_teste, previsoes_do_1nn, target_names=nomes_das_classes))

print("=" * 70)
print("INTERPRETAÇÃO DO RESULTADO")
print("=" * 70)
if acuracia_do_1nn >= 0.90:
    print(
        "A acurácia do 1-NN ficou muito alta (>= 90%). Isso é um forte indício\n"
        "de VAZAMENTO DE DADOS: um classificador que apenas 'copia' a resposta\n"
        "do exemplo de treino mais parecido já acerta quase tudo. Provavelmente\n"
        "existem imagens quase idênticas (frames vizinhos do mesmo vídeo) indo\n"
        "parar uma no treino e outra no teste."
    )
else:
    print(
        "A acurácia do 1-NN não ficou tão alta. Isso é um indício de que o\n"
        "teste não está tão 'contaminado' por imagens quase idênticas do\n"
        "treino, mas vale continuar de olho nisso."
    )
print("=" * 70)
