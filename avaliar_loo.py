"""
ETAPA 1 do plano de melhoria: AVALIACAO "LEAVE-ONE-PERSON-OUT" (LOPO).

Para cada pessoa do ER_Dataset_Unificado, treina com TODAS AS OUTRAS pessoas e
testa na pessoa deixada de fora. Assim a nota mede o que importa: acertar em
alguem que o modelo nunca viu.

Roda duas versoes do problema:
  - binario : engajado (niveis 2 e 3) x nao engajado (niveis 0 e 1)
  - 4 niveis: o problema original

Modelos: SVM, Random Forest, MLP (mesmas configuracoes do treinar.py),
Regressao Logistica (baseline simples) e um "chute" (classe mais comum do
treino) como piso de comparacao.

Cuidados que este script toma (ver relatorio gerado):
  - A pessoa B do Kaggle tem so 6 imagens: entra no TREINO, mas nao e avaliada
    sozinha (pouco material para uma nota confiavel).
  - Varias pessoas do Kaggle tem UMA classe so (ex.: A e C so "nao engajado",
    F so "engajado"). Para elas a acuracia e so o recall dessa classe. Por isso
    o numero principal e a ACURACIA BALANCEADA calculada juntando as
    predicoes de todas as pessoas (pooled), e tambem se mostra a media por
    pessoa.

As features sao extraidas uma vez e guardadas em cache (features_unificado.npz).

Nao altera nenhum arquivo existente. Gera:
  features_unificado.npz, resultados_loo.csv, relatorio_loo_baseline.txt

Uso:
    python avaliar_loo.py              (usa o cache se existir)
    python avaliar_loo.py --refazer    (reextrai as features)
"""

import argparse
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import RAIZ, criar_detector_predictor, extrair_features

warnings.filterwarnings("ignore")

DATASET = RAIZ / "ER_Dataset_Unificado"
CACHE = RAIZ / "features_unificado.npz"
SAIDA_CSV = RAIZ / "resultados_loo.csv"
SAIDA_TXT = RAIZ / "relatorio_loo_baseline.txt"

MIN_IMAGENS_AVALIAR = 50   # pessoa com menos que isso so entra no treino
SEMENTE = 42
LADO_MAXIMO = 1280         # reduz imagens maiores que isso antes do dlib


def pessoa_do_arquivo(nome):
    """'kaggle_pessoaC_...' -> 'kC' ; 'pessoa2_sessao01_...' -> 'p2'."""
    m = re.match(r"kaggle_pessoa([A-Z])_", nome)
    if m:
        return "k" + m.group(1)
    m = re.match(r"pessoa(\d+)_", nome)
    if m:
        return "p" + m.group(1)
    raise ValueError(f"Nome de arquivo fora do padrao: {nome}")


def extrair_tudo():
    detector, predictor = criar_detector_predictor()
    arquivos = sorted(DATASET.rglob("*.jpg"))
    X, y, pessoas, nomes, sem_rosto = [], [], [], [], 0
    for caminho in tqdm(arquivos, desc="Extraindo features"):
        img = cv2.imread(str(caminho))
        if img is None:
            continue
        # Imagens do conjunto externo sao 4K; reduz ao tamanho do Kaggle
        # (lado maior = 1280 px). Os pontos sao normalizados, entao a escala
        # nao altera as features, e a deteccao fica ~10x mais rapida.
        maior = max(img.shape[:2])
        if maior > LADO_MAXIMO:
            esc = LADO_MAXIMO / maior
            img = cv2.resize(img, None, fx=esc, fy=esc,
                             interpolation=cv2.INTER_AREA)
        f, _ = extrair_features(img, detector, predictor)
        if f is None:
            sem_rosto += 1
            continue
        X.append(f)
        y.append(int(caminho.parent.name))
        pessoas.append(pessoa_do_arquivo(caminho.name))
        nomes.append(caminho.name)
    print(f"{len(X)} imagens com rosto; {sem_rosto} sem rosto (ignoradas).")
    np.savez(CACHE, X=np.array(X), y=np.array(y),
             pessoas=np.array(pessoas), nomes=np.array(nomes))


def carregar():
    d = np.load(CACHE, allow_pickle=True)
    return d["X"], d["y"], d["pessoas"]


def criar_modelos():
    return {
        "SVM": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
        "Random_Forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEMENTE,
            n_jobs=-1),
        "MLP": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                  random_state=SEMENTE))]),
        "Reg_Logistica": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    }


def rodar_lopo(X, y, pessoas, tarefa):
    """Devolve (DataFrame por pessoa x modelo, dict de predicoes agregadas)."""
    modelos = criar_modelos()
    avaliaveis = [p for p in sorted(set(pessoas))
                  if (pessoas == p).sum() >= MIN_IMAGENS_AVALIAR]
    linhas = []
    pooled = {n: ([], []) for n in list(modelos) + ["Chute_classe_comum"]}

    for p in tqdm(avaliaveis, desc=f"LOPO {tarefa}"):
        teste = pessoas == p
        treino = ~teste
        y_tr, y_te = y[treino], y[teste]
        pred_por_modelo = {}
        for nome, fabrica in modelos.items():
            m = fabrica().fit(X[treino], y_tr)
            pred_por_modelo[nome] = m.predict(X[teste])
        mais_comum = np.bincount(y_tr).argmax()
        pred_por_modelo["Chute_classe_comum"] = np.full(len(y_te), mais_comum)

        for nome, pred in pred_por_modelo.items():
            pooled[nome][0].append(y_te)
            pooled[nome][1].append(pred)
            linhas.append({
                "tarefa": tarefa, "pessoa_testada": p, "modelo": nome,
                "n_teste": int(teste.sum()),
                "classes_na_pessoa": "".join(str(c) for c in sorted(set(y_te))),
                "acuracia": accuracy_score(y_te, pred),
                "f1_macro": f1_score(y_te, pred, average="macro"),
            })
    return pd.DataFrame(linhas), pooled


def resumo_pooled(pooled, tarefa):
    linhas = []
    for nome, (ys, ps) in pooled.items():
        yt, yp = np.concatenate(ys), np.concatenate(ps)
        linhas.append({
            "tarefa": tarefa, "modelo": nome,
            "acuracia": accuracy_score(yt, yp),
            "acuracia_balanceada": balanced_accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro"),
        })
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()

    if args.refazer or not CACHE.exists():
        extrair_tudo()
    X, y4, pessoas = carregar()
    y_bin = (y4 >= 2).astype(int)

    tabelas, pooleds = [], []
    for tarefa, y in (("binario", y_bin), ("4_niveis", y4)):
        por_pessoa, pooled = rodar_lopo(X, y, pessoas, tarefa)
        tabelas.append(por_pessoa)
        pooleds.append(resumo_pooled(pooled, tarefa))
    por_pessoa = pd.concat(tabelas)
    agregado = pd.concat(pooleds)
    por_pessoa.to_csv(SAIDA_CSV, index=False)

    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t)
            f.write(t + "\n")

        w("BASELINE LEAVE-ONE-PERSON-OUT (ER_Dataset_Unificado)")
        w("=" * 70)
        w(f"Imagens com rosto: {len(y4)}")
        ctab = pd.crosstab(pessoas, y4)
        w("\nImagens por pessoa x nivel (0..3):")
        w(ctab.to_string())
        w("\nATENCAO: pessoas kA, kC (so niveis 0/1) e kF (so nivel 2) tem UMA "
          "classe binaria so;\nnelas a acuracia e apenas o recall dessa classe. "
          "kB (6 imagens) so treina.")
        for tarefa in ("binario", "4_niveis"):
            w(f"\n{'#' * 70}\nTAREFA: {tarefa}\n{'#' * 70}")
            ag = agregado[agregado.tarefa == tarefa].drop(columns="tarefa")
            w("\n-- Resultado agregado (predicoes de todas as pessoas juntas) --")
            w(ag.round(3).to_string(index=False))
            pp = por_pessoa[por_pessoa.tarefa == tarefa]
            w("\n-- Acuracia por pessoa deixada de fora --")
            w(pp.pivot(index="pessoa_testada", columns="modelo",
                       values="acuracia").round(3).to_string())
            media = pp.groupby("modelo")[["acuracia", "f1_macro"]].agg(
                ["mean", "std"]).round(3)
            w("\n-- Media e desvio entre pessoas --")
            w(media.to_string())
    print(f"\nSalvo: {SAIDA_TXT.name}, {SAIDA_CSV.name}, {CACHE.name}")


if __name__ == "__main__":
    main()
