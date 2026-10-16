"""
Avaliacao dos modelos sobre TRES fontes de dados juntas, em formato BINARIO
(engajado x nao engajado):

  1) ER_Dataset_Unificado  : niveis 0,1 -> nao engajado ; 2,3 -> engajado
                             (8 pessoas; kB so treina por ter 6 imagens)
  2) dataset               : Desengajado -> 0 ; Engajado -> 1
                             "Parcialmente engajado" e EXCLUIDO (decisao do
                             grupo). Pessoa = numero em "imgN_..." (suposicao:
                             N identifica a pessoa/video).
  3) dataset-2 (EED)       : interested, stimulated, fulfilled, satisfied -> 1
                             distracted -> 0
                             neutral, confused, overwhelmed, skeptical sao
                             EXCLUIDAS (rotulo ambiguo). O EED nao identifica
                             pessoas, entao conta como UM unico grupo.

Dois testes de generalizacao, ambos sem deixar nada do teste vazar para o treino:
  A) LOPO: deixa um grupo (pessoa) de fora, treina com todo o resto.
  B) Leave-one-dataset-out: treina com duas fontes e testa na terceira.

Metrica principal: ACURACIA BALANCEADA (acaso = 0,50), porque as classes sao
desbalanceadas. Grupos com uma classe so tem a metrica reportada, mas ficam
fora da media "grupos com as duas classes".

Features em cache (features_multi.npz). Nao altera arquivos existentes.
Gera: features_multi.npz, resultados_multi.csv, relatorio_multi_datasets.txt

Uso:
    python avaliar_multi_datasets.py [--refazer]
"""

import argparse
import re
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from features import RAIZ, criar_detector_predictor, extrair_features
from avaliar_loo import criar_modelos, LADO_MAXIMO

warnings.filterwarnings("ignore")

CACHE_UNIFICADO = RAIZ / "features_unificado.npz"
CACHE = RAIZ / "features_multi.npz"
SAIDA_CSV = RAIZ / "resultados_multi.csv"
SAIDA_TXT = RAIZ / "relatorio_multi_datasets.txt"

DATASET_1 = RAIZ / "dataset"
DATASET_2 = RAIZ / "dataset-2" / "EED"
MIN_AVALIAR = 50

EED_ENGAJADO = {"interested", "stimulated", "fulfilled", "satisfied"}
EED_NAO_ENGAJADO = {"distracted"}


def lista_dataset1():
    itens = []
    for pasta, rotulo in (("Engajado", 1), ("Desengajado", 0)):
        for p in sorted((DATASET_1 / pasta).glob("*.jpg")):
            m = re.match(r"img(\d+)_", p.name)
            pessoa = "D:img" + (m.group(1) if m else "?")
            itens.append((p, rotulo, "dataset", pessoa))
    return itens


def lista_eed():
    itens = []
    for sub in ("train", "test"):
        for classe in sorted((DATASET_2 / sub).iterdir()):
            if classe.name in EED_ENGAJADO:
                rotulo = 1
            elif classe.name in EED_NAO_ENGAJADO:
                rotulo = 0
            else:
                continue
            for p in sorted(classe.glob("*.jpg")):
                itens.append((p, rotulo, "EED", "EED"))
    return itens


def extrair(itens, descricao):
    detector, predictor = criar_detector_predictor()
    X, y, fonte, grupo, sem_rosto = [], [], [], [], 0
    for caminho, rotulo, origem, pessoa in tqdm(itens, desc=descricao):
        img = cv2.imread(str(caminho))
        if img is None:
            continue
        maior = max(img.shape[:2])
        if maior > LADO_MAXIMO:
            esc = LADO_MAXIMO / maior
            img = cv2.resize(img, None, fx=esc, fy=esc,
                             interpolation=cv2.INTER_AREA)
        f, _ = extrair_features(img, detector, predictor)
        if f is None:
            sem_rosto += 1
            continue
        X.append(f); y.append(rotulo); fonte.append(origem); grupo.append(pessoa)
    print(f"{descricao}: {len(X)} com rosto, {sem_rosto} sem rosto.")
    return X, y, fonte, grupo, sem_rosto


def montar_cache():
    d = np.load(CACHE_UNIFICADO, allow_pickle=True)
    X = list(d["X"]); y = list((d["y"] >= 2).astype(int))
    fonte = ["unificado"] * len(X)
    grupo = ["U:" + p for p in d["pessoas"]]
    sem = {"unificado": 140}
    for itens, nome in ((lista_dataset1(), "dataset"), (lista_eed(), "EED")):
        a, b, c, g, s = extrair(itens, f"Extraindo {nome}")
        X += a; y += b; fonte += c; grupo += g; sem[nome] = s
    np.savez(CACHE, X=np.array(X), y=np.array(y), fonte=np.array(fonte),
             grupo=np.array(grupo), sem_rosto=np.array(list(sem.items())))


def metricas(yt, yp):
    return {
        "acuracia": accuracy_score(yt, yp),
        "acuracia_bal": balanced_accuracy_score(yt, yp) if len(set(yt)) > 1
        else np.nan,
        "f1_macro": f1_score(yt, yp, average="macro"),
    }


def treinar_prever(X, y, treino, teste):
    modelos = criar_modelos()
    preds = {}
    for nome, fabrica in modelos.items():
        preds[nome] = fabrica().fit(X[treino], y[treino]).predict(X[teste])
    preds["Chute_classe_comum"] = np.full(
        int(teste.sum()), np.bincount(y[treino]).argmax())
    return preds


def avaliar(X, y, grupos_teste, rotulos_teste, tarefa):
    """grupos_teste: lista de mascaras booleanas de teste (treino = o resto)."""
    linhas, pooled = [], {}
    for nome_teste, teste in tqdm(list(zip(rotulos_teste, grupos_teste)),
                                  desc=tarefa):
        treino = ~teste
        preds = treinar_prever(X, y, treino, teste)
        for modelo, p in preds.items():
            pooled.setdefault(modelo, ([], []))
            pooled[modelo][0].append(y[teste]); pooled[modelo][1].append(p)
            linhas.append({
                "tarefa": tarefa, "teste": nome_teste, "modelo": modelo,
                "n_teste": int(teste.sum()),
                "n_engajado": int((y[teste] == 1).sum()),
                "n_nao_engajado": int((y[teste] == 0).sum()),
                **metricas(y[teste], p)})
    return pd.DataFrame(linhas), pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()
    if args.refazer or not CACHE.exists():
        montar_cache()
    d = np.load(CACHE, allow_pickle=True)
    X, y, fonte, grupo = d["X"], d["y"], d["fonte"], d["grupo"]

    # --- A) LOPO ---
    nomes = [g for g in sorted(set(grupo)) if (grupo == g).sum() >= MIN_AVALIAR]
    lopo, pooled_lopo = avaliar(X, y, [grupo == g for g in nomes], nomes, "LOPO")

    # --- B) deixa um dataset inteiro de fora ---
    fontes = ["unificado", "dataset", "EED"]
    lodo, pooled_lodo = avaliar(X, y, [fonte == f for f in fontes], fontes,
                                "Dataset_de_fora")

    tudo = pd.concat([lopo, lodo])
    tudo.to_csv(SAIDA_CSV, index=False)

    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t); f.write(t + "\n")

        w("AVALIACAO BINARIA COM TRES FONTES DE DADOS")
        w("=" * 72)
        w(f"Imagens com rosto: {len(y)}  (engajado={int((y == 1).sum())}, "
          f"nao engajado={int((y == 0).sum())})")
        w("\nImagens por fonte x classe:")
        w(pd.crosstab(fonte, y).rename(columns={0: "nao_engajado",
                                                1: "engajado"}).to_string())
        w(f"\nGrupos avaliados no LOPO (>= {MIN_AVALIAR} imagens): {len(nomes)}")
        w("Acaso da acuracia balanceada = 0,50.")

        for titulo, df, pooled in (
            ("A) LOPO: um grupo (pessoa) de fora por vez", lopo, pooled_lopo),
            ("B) UM DATASET INTEIRO DE FORA (treina nos outros dois)", lodo,
             pooled_lodo),
        ):
            w(f"\n{'#' * 72}\n{titulo}\n{'#' * 72}")
            agg = []
            for m, (ys, ps) in pooled.items():
                r = metricas(np.concatenate(ys), np.concatenate(ps))
                agg.append({"modelo": m, **r})
            w("\n-- Agregado (todas as predicoes juntas) --")
            w(pd.DataFrame(agg).round(3).to_string(index=False))
            ambas = df[(df.n_engajado > 0) & (df.n_nao_engajado > 0)]
            w(f"\n-- Media por grupo, so grupos com as duas classes "
              f"({ambas.teste.nunique()} grupos) --")
            w(ambas.groupby("modelo")[["acuracia_bal", "f1_macro"]]
              .agg(["mean", "std"]).round(3).to_string())
            w("\n-- Acuracia balanceada por grupo (NaN = grupo com uma classe so) --")
            w(df.pivot(index="teste", columns="modelo", values="acuracia_bal")
              .round(3).to_string())
            if titulo.startswith("B"):
                w("\n-- Acuracia simples por dataset --")
                w(df.pivot(index="teste", columns="modelo", values="acuracia")
                  .round(3).to_string())
    print(f"\nSalvo: {SAIDA_TXT.name}, {SAIDA_CSV.name}, {CACHE.name}")


if __name__ == "__main__":
    main()
