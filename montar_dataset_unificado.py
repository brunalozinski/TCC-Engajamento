"""
Monta ER_Dataset_Unificado/, juntando:
  1) ER_Dataset (Kaggle) -- copiado com o ID de PESSOA REAL no nome do
     arquivo (de mapeamento_pessoas_kaggle.py), em vez do nome original
     (que so tinha categoria + numero de frame, sem identidade real).
  2) ER_Dataset_Externo -- copiado como esta (ja tem pessoa/sessao no nome).

NADA e movido ou apagado do ER_Dataset nem do ER_Dataset_Externo originais
-- este script so copia. ER_Dataset_Unificado/ e uma pasta nova.

Convencao de nome final (igual nas duas origens, pra poder juntar tudo numa
mesma logica de split por pessoa depois):

    kaggle_pessoa<ID>_<categoria>_<numero_original>.jpg   (origem Kaggle)
    pessoaN_sessaoXX_classeY_NNNN.jpg                      (origem externa, sem mudanca)

O prefixo "kaggle_" deixa claro a origem e evita qualquer colisao de nome
com as pessoas do conjunto externo (pessoa1, pessoa2, pessoa3), que usam
IDs numericos diferentes dos IDs de letra (A-F) do Kaggle.

Uso:
    python montar_dataset_unificado.py
"""

import shutil
from pathlib import Path

from features import RAIZ
from mapeamento_pessoas_kaggle import PESSOA_POR_GRUPO

ORIGEM_KAGGLE = RAIZ / "ER_Dataset"
ORIGEM_EXTERNO = RAIZ / "ER_Dataset_Externo"
DESTINO = RAIZ / "ER_Dataset_Unificado"


# Reimplementado aqui (em vez de "from treinar import ...") porque treinar.py
# e um SCRIPT, nao um modulo: importa-lo executaria o treinamento inteiro
# (carrega 2120 imagens com dlib e treina os 3 modelos) so para pegar esta
# funcao. A logica abaixo e identica a de treinar.py.

def _extrair_categoria_e_numero(nome_do_arquivo):
    nome_sem_extensao = nome_do_arquivo.rsplit(".", 1)[0]
    partes_do_nome = nome_sem_extensao.split("_")
    ultimo_pedaco = partes_do_nome[-1]
    categoria = "_".join(partes_do_nome[:-1])
    numero = int(ultimo_pedaco) if ultimo_pedaco.isdigit() else None
    return categoria, numero


def identificar_grupo_de_cada_imagem(nomes_arquivo):
    categorias = []
    numeros = []
    for nome in nomes_arquivo:
        categoria, numero = _extrair_categoria_e_numero(nome)
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
            grupo_de_cada_indice[indice] = f"{categoria_atual}_grupo{numero_do_grupo_atual}"
            numero_anterior = numero_atual

    return grupo_de_cada_indice


def copiar_kaggle():
    total_copiado = 0
    total_sem_mapeamento = 0
    for pasta_nivel in sorted(p for p in ORIGEM_KAGGLE.iterdir() if p.is_dir()):
        nivel = pasta_nivel.name
        pasta_destino = DESTINO / nivel
        pasta_destino.mkdir(parents=True, exist_ok=True)

        imagens = [f for f in pasta_nivel.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        nomes = [f.name for f in imagens]
        grupos = identificar_grupo_de_cada_imagem(nomes)

        for img_file, grupo in zip(imagens, grupos):
            pessoa = PESSOA_POR_GRUPO.get(grupo)
            if pessoa is None:
                print(f"  AVISO: grupo '{grupo}' sem mapeamento de pessoa, pulando {img_file.name}")
                total_sem_mapeamento += 1
                continue
            novo_nome = f"kaggle_pessoa{pessoa}_{img_file.name}"
            shutil.copy2(img_file, pasta_destino / novo_nome)
            total_copiado += 1

        print(f"  Nivel {nivel} (Kaggle): {len(imagens)} imagens processadas")

    print(f"\nTotal copiado do Kaggle: {total_copiado}")
    if total_sem_mapeamento:
        print(f"Total SEM mapeamento de pessoa (nao copiado): {total_sem_mapeamento}")


def copiar_externo():
    total_copiado = 0
    for pasta_nivel in sorted(p for p in ORIGEM_EXTERNO.iterdir() if p.is_dir()):
        nivel = pasta_nivel.name
        pasta_destino = DESTINO / nivel
        pasta_destino.mkdir(parents=True, exist_ok=True)

        imagens = [f for f in pasta_nivel.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for img_file in imagens:
            shutil.copy2(img_file, pasta_destino / img_file.name)
            total_copiado += 1

        print(f"  Nivel {nivel} (Externo): {len(imagens)} imagens copiadas")

    print(f"\nTotal copiado do Externo: {total_copiado}")


def main():
    print("Copiando ER_Dataset (Kaggle) com ID de pessoa real no nome...")
    copiar_kaggle()

    print("\nCopiando ER_Dataset_Externo...")
    copiar_externo()

    print(f"\nER_Dataset_Unificado pronto em: {DESTINO}")
    print("NOTA: ER_Dataset e ER_Dataset_Externo originais nao foram alterados.")


if __name__ == "__main__":
    main()
