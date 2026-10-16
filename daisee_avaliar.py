"""
DAiSEE: treina e avalia os modelos usando SOMENTE o DAiSEE.

Entrada: daisee_resultados/daisee_features.npz (daisee_extrair_features.py).

Cada clipe vira um vetor (agregado dos ate 8 frames com rosto; clipes com menos
de MIN_FRAMES frames validos sao descartados):
  bruto      : media dos 136 numeros dos pontos normalizados (136)
  geometrico : media e desvio das 20 medidas geometricas de features.py (40)
Normalizacao opcional por pessoa ("centro"): subtrai a media da pessoa (todos os
clipes dela). Nao usa rotulos; e transdutiva (usa todos os clipes da pessoa).

Tarefas (rotulo = Engagement do DAiSEE, 0 a 3):
  binario   : nao engajado (0,1) x engajado (2,3)         [tarefa principal]
  4_niveis  : 0, 1, 2, 3
  2_vs_3    : alto (2) x muito alto (3), so clipes 2 e 3   [a variacao do DAiSEE
              esta quase toda aqui: 94% dos clipes sao 2 ou 3]

Protocolos (nenhuma pessoa aparece em treino e teste ao mesmo tempo):
  A) Divisao OFICIAL: treina em Train; testa em Validation e em Test.
  B) Validacao cruzada de 5 dobras AGRUPADA POR PESSOA (StratifiedGroupKFold) em
     todas as pessoas: mais estavel que um unico teste.

Metrica principal: acuracia balanceada (acaso = 1/numero de classes). Para o
binario tambem AUC (independe de limiar). "Chute" = prever sempre a classe mais
comum do treino.

Gera: daisee_resultados/resultados_daisee.csv e relatorio_daisee.txt
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from features import RAIZ, geometricas_do_vetor

warnings.filterwarnings("ignore")

SAIDA = RAIZ / "daisee_resultados"
ENTRADA = SAIDA / "daisee_features.npz"
SAIDA_CSV = SAIDA / "resultados_daisee.csv"
SAIDA_TXT = SAIDA / "relatorio_daisee.txt"
MIN_FRAMES = 3
SEMENTE = 0


def modelos():
    return {
        "Reg_Logistica": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "SVM": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(class_weight="balanced"))]),
        "Random_Forest": lambda: RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=SEMENTE,
            n_jobs=2),
        "MLP": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=200,
                                  early_stopping=True, random_state=SEMENTE))]),
    }


def montar_vetores(d):
    X, pessoa = d["X"], d["pessoa"].astype(str)
    valido = np.isfinite(X[:, :, 0])
    n_ok = valido.sum(axis=1)
    G = np.full((X.shape[0], X.shape[1], 20), np.nan)
    for i, j in zip(*np.nonzero(valido)):
        G[i, j] = geometricas_do_vetor(X[i, j])
    manter = n_ok >= MIN_FRAMES
    bruto = np.nanmean(X, axis=1)
    geo = np.hstack([np.nanmean(G, axis=1), np.nanstd(G, axis=1)])
    return manter, n_ok, bruto, geo, pessoa


def normalizar_pessoa(V, pessoa):
    Vn = V.copy()
    for p in set(pessoa):
        m = pessoa == p
        Vn[m] = V[m] - V[m].mean(axis=0)
    return Vn


def pontuacao(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def avaliar(yt, yp, score, tarefa):
    r = {"acuracia": float((yt == yp).mean()),
         "acuracia_bal": balanced_accuracy_score(yt, yp),
         "f1_macro": f1_score(yt, yp, average="macro")}
    classes = sorted(set(yt))
    rec = recall_score(yt, yp, labels=classes, average=None, zero_division=0)
    for c, v in zip(classes, rec):
        r[f"recall_{c}"] = float(v)
    r["auc"] = (roc_auc_score(yt, score) if score is not None and len(classes) == 2
                else np.nan)
    return r


def rodar(V, y, treino, teste, nome_tarefa, linhas, contexto):
    """Treina todos os modelos em `treino` e avalia em `teste`."""
    mais_comum = np.bincount(y[treino]).argmax()
    yt = y[teste]
    r = avaliar(yt, np.full(len(yt), mais_comum), None, nome_tarefa)
    linhas.append({**contexto, "modelo": "Chute_classe_comum", **r})
    if len(set(y[treino])) < 2:
        return
    for nome, fab in modelos().items():
        m = fab().fit(V[treino], y[treino])
        pred = m.predict(V[teste])
        score = pontuacao(m, V[teste]) if len(set(y[treino])) == 2 else None
        linhas.append({**contexto, "modelo": nome,
                       **avaliar(yt, pred, score, nome_tarefa)})


def main():
    d = np.load(ENTRADA, allow_pickle=True)
    manter, n_ok, bruto, geo, pessoa = montar_vetores(d)
    divisao = d["divisao"].astype(str)
    eng = d["engajamento"].astype(int)
    print(f"{len(eng)} clipes; descartados por ter < {MIN_FRAMES} frames com rosto: "
          f"{int((~manter).sum())}")

    k = manter
    eng, divisao, pessoa = eng[k], divisao[k], pessoa[k]
    base = {"bruto": bruto[k], "geometrico": geo[k]}
    vetores = {}
    for f, V in base.items():
        vetores[(f, "nenhuma")] = V
        vetores[(f, "centro_pessoa")] = normalizar_pessoa(V, pessoa)

    tarefas = {
        "binario": (np.ones(len(eng), bool), (eng >= 2).astype(int)),
        "4_niveis": (np.ones(len(eng), bool), eng),
        "2_vs_3": (eng >= 2, (eng == 3).astype(int)),
    }

    linhas = []
    total = len(tarefas) * len(vetores)
    barra = tqdm(total=total, desc="configuracoes")
    for tarefa, (usar, y_full) in tarefas.items():
        for (feat, norm), V_full in vetores.items():
            V, y = V_full[usar], y_full[usar]
            div, pes = divisao[usar], pessoa[usar]
            ctx = {"tarefa": tarefa, "features": feat, "normalizacao": norm}
            # A) divisao oficial
            tr = div == "Train"
            for alvo in ("Validation", "Test"):
                rodar(V, y, tr, div == alvo, tarefa, linhas,
                      {**ctx, "protocolo": f"A_oficial_{alvo}", "dobra": 0})
            # B) validacao cruzada agrupada por pessoa
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                        random_state=SEMENTE)
            for i, (a, b) in enumerate(sgkf.split(V, y, groups=pes)):
                m_tr = np.zeros(len(y), bool); m_tr[a] = True
                m_te = np.zeros(len(y), bool); m_te[b] = True
                rodar(V, y, m_tr, m_te, tarefa, linhas,
                      {**ctx, "protocolo": "B_cv_pessoas", "dobra": i + 1})
            barra.update(1)
    barra.close()

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA_CSV, index=False)

    acaso = {"binario": 0.5, "4_niveis": 0.25, "2_vs_3": 0.5}
    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t); f.write(t + "\n")

        w("DAiSEE: RESULTADOS (somente DAiSEE)")
        w("=" * 78)
        w(f"Clipes usados: {len(eng)} de {len(d['engajamento'])} "
          f"(>= {MIN_FRAMES} frames com rosto) | pessoas: {len(set(pessoa))}")
        tab = pd.crosstab(divisao, eng)
        tab["total"] = tab.sum(axis=1)
        w("\nClipes por divisao e nivel de Engagement:")
        w(tab.to_string())
        w(f"\nbinario: nao engajado = {int((eng <= 1).sum())} "
          f"({(eng <= 1).mean() * 100:.1f}%), engajado = {int((eng >= 2).sum())}")
        for tarefa in tarefas:
            sub = df[df.tarefa == tarefa]
            w(f"\n{'#' * 78}\nTAREFA: {tarefa}   (acaso da acuracia balanceada = "
              f"{acaso[tarefa]})\n{'#' * 78}")
            sub = sub.assign(config=sub.features + " | " + sub.normalizacao)
            for prot, titulo in (("A_oficial_Test", "A) Divisao oficial: TESTE"),
                                 ("A_oficial_Validation",
                                  "A) Divisao oficial: VALIDACAO")):
                p = sub[sub.protocolo == prot]
                w(f"\n-- {titulo} (acuracia balanceada) --")
                w(p.pivot_table(index="config", columns="modelo",
                                values="acuracia_bal").round(3).to_string())
            p = sub[sub.protocolo == "B_cv_pessoas"]
            w("\n-- B) CV por pessoa, 5 dobras: media da acuracia balanceada --")
            w(p.pivot_table(index="config", columns="modelo",
                            values="acuracia_bal").round(3).to_string())
            w("-- B) desvio entre as dobras --")
            w(p.pivot_table(index="config", columns="modelo",
                            values="acuracia_bal", aggfunc="std").round(3).to_string())
            if tarefa != "4_niveis":
                w("\n-- AUC (media das dobras do B; 0,5 = sem sinal) --")
                w(p.pivot_table(index="config", columns="modelo",
                                values="auc").round(3).to_string())
                w("\n-- AUC no TESTE oficial --")
                pt = sub[sub.protocolo == "A_oficial_Test"]
                w(pt.pivot_table(index="config", columns="modelo",
                                 values="auc").round(3).to_string())
            else:
                w("\n-- Recall por nivel (CV, SVM, geometrico | centro_pessoa) --")
                r = p[(p.modelo == "SVM") & (p.config == "geometrico | centro_pessoa")]
                cols = [c for c in r.columns if c.startswith("recall_")]
                w(r[cols].mean().round(3).to_string())
    print(f"\nSalvo: {SAIDA_TXT}")


if __name__ == "__main__":
    main()
