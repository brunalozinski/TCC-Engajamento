"""
ETAPA 2: estudo de ABLACAO das features, no mesmo protocolo da etapa 1.

Tres conjuntos de features sao comparados (tarefa binaria engajado x nao):
  bruto     : 136 coordenadas dos 68 pontos normalizados (o que se usava)
  geometrico: 20 medidas derivadas (EAR, MAR, sobrancelha, yaw/pitch/roll...)
              -- ver features.features_geometricas
  bruto+geo : os dois juntos (156)

Protocolos (dados vindos de features_multi.npz, sem reextrair imagens):
  A) Dataset de fora: treina em duas fontes, testa na terceira (3 rodadas)
  B) LOPO so nos grupos com >= MIN_CLASSE imagens de CADA classe (os demais
     tem poucos exemplos de uma classe e geram notas ruidosas, como visto na
     etapa 1)

Tambem calcula, para cada feature geometrica, o AUC univariado (engajado x
nao) em cada fonte: AUC ~0,5 = sem sinal; longe de 0,5 e com o MESMO sentido
nas tres fontes = sinal consistente.

Gera: resultados_etapa2.csv, relatorio_etapa2_features.txt
Nao altera arquivos existentes.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import RAIZ, geometricas_do_vetor, NOMES_GEOMETRICAS
from avaliar_multi_datasets import avaliar, CACHE

warnings.filterwarnings("ignore")

SAIDA_CSV = RAIZ / "resultados_etapa2.csv"
SAIDA_TXT = RAIZ / "relatorio_etapa2_features.txt"
MIN_CLASSE = 30


def main():
    d = np.load(CACHE, allow_pickle=True)
    X_bruto, y, fonte, grupo = d["X"], d["y"], d["fonte"], d["grupo"]
    X_geo = np.array([geometricas_do_vetor(x) for x in X_bruto])
    conjuntos = {"bruto": X_bruto, "geometrico": X_geo,
                 "bruto+geo": np.hstack([X_bruto, X_geo])}

    grupos_ok = [g for g in sorted(set(grupo))
                 if ((grupo == g) & (y == 1)).sum() >= MIN_CLASSE
                 and ((grupo == g) & (y == 0)).sum() >= MIN_CLASSE]
    fontes = ["unificado", "dataset", "EED"]

    tabelas = []
    for nome, X in conjuntos.items():
        lopo, _ = avaliar(X, y, [grupo == g for g in grupos_ok], grupos_ok,
                          f"LOPO [{nome}]")
        lodo, _ = avaliar(X, y, [fonte == f for f in fontes], fontes,
                          f"Dataset de fora [{nome}]")
        lopo["tarefa"] = "LOPO"; lodo["tarefa"] = "Dataset_de_fora"
        t = pd.concat([lopo, lodo]); t["features"] = nome
        tabelas.append(t)
    tudo = pd.concat(tabelas)
    tudo.to_csv(SAIDA_CSV, index=False)

    auc_linhas = []
    for i, nome in enumerate(NOMES_GEOMETRICAS):
        linha = {"feature": nome}
        for f in fontes:
            m = fonte == f
            linha[f] = roc_auc_score(y[m], X_geo[m, i])
        auc_linhas.append(linha)
    auc = pd.DataFrame(auc_linhas).set_index("feature")
    sentidos = np.sign(auc[fontes].values - 0.5)
    auc["consistente"] = (np.abs(sentidos.sum(axis=1)) == 3)
    auc["forca_min"] = (auc[fontes] - 0.5).abs().min(axis=1)

    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t); f.write(t + "\n")

        w("ETAPA 2 - ABLACAO DE FEATURES (binario, acaso da acuracia balanceada = 0,50)")
        w("=" * 76)
        w(f"Imagens: {len(y)} | LOPO nos grupos com >= {MIN_CLASSE} imagens de cada "
          f"classe: {', '.join(grupos_ok)}")

        for tarefa, titulo in (("Dataset_de_fora", "A) UM DATASET INTEIRO DE FORA"),
                               ("LOPO", "B) LOPO (grupos com amostra suficiente)")):
            sub = tudo[(tudo.tarefa == tarefa) & (tudo.modelo != "Chute_classe_comum")]
            w(f"\n{'#' * 76}\n{titulo}\n{'#' * 76}")
            w("\n-- Acuracia balanceada media entre os grupos de teste --")
            w(sub.groupby(["features", "modelo"]).acuracia_bal.mean()
              .unstack("modelo").round(3).to_string())
            w("\n-- Desvio entre os grupos de teste --")
            w(sub.groupby(["features", "modelo"]).acuracia_bal.std()
              .unstack("modelo").round(3).to_string())
            w("\n-- Detalhe por grupo de teste --")
            w(sub.pivot_table(index=["teste", "features"], columns="modelo",
                              values="acuracia_bal").round(3).to_string())

        w(f"\n{'#' * 76}\nC) AUC UNIVARIADO DE CADA FEATURE GEOMETRICA\n{'#' * 76}")
        w("(0,5 = sem sinal; 'consistente' = mesmo sentido nas 3 fontes)\n")
        w(auc.sort_values("forca_min", ascending=False).round(3).to_string())
    print(f"\nSalvo: {SAIDA_TXT.name}, {SAIDA_CSV.name}")


if __name__ == "__main__":
    main()
