"""
Teste: NORMALIZACAO POR PESSOA das features.

Hipotese (ver diagnostico da etapa 1b): os modelos reagem ao rosto/ambiente de
cada pessoa, nao a expressao (ex.: o SVM chama quase tudo de "nao engajado" na
pessoa 4, seja qual for o estado). Se cada pessoa tiver as features deslocadas
pela sua propria "linha de base" (formato do rosto, camera, iluminacao),
subtrair essa linha de base deve deixar so a variacao de expressao.

Como: para cada pessoa (grupo), subtrai a MEDIA das features dela (modo
"centro") ou subtrai a media e divide pelo desvio (modo "zscore"). Nao usa
rotulos. Na demo, equivale a calibrar com os primeiros segundos da sessao.

Atencao (limitacao): a media e calculada com TODOS os frames da pessoa, entao
o teste e "transdutivo" -- numa demo ao vivo a media viria de uma janela inicial.
E pessoas com UMA classe so (kA, kC, kF) perdem o proprio sinal ao serem
centradas (a media delas ja e a classe). Por isso duas versoes de treino:
  todos     : normaliza e usa todos os grupos no treino
  2_classes : so grupos com >= 30 imagens de CADA classe entram no treino

Protocolo igual ao das etapas anteriores (LOPO nos grupos com >= 30 imagens de
cada classe + uma fonte inteira de fora), metrica = acuracia balanceada.
Modelos: SVM, Regressao Logistica, Random Forest (semente 0; MLP fica de fora
para o teste caber em tempo razoavel).

Gera: resultados_normalizacao_pessoa.csv, relatorio_normalizacao_pessoa.txt
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from features import RAIZ, geometricas_do_vetor
from avaliar_multi_datasets import CACHE

warnings.filterwarnings("ignore")

SAIDA_CSV = RAIZ / "resultados_normalizacao_pessoa.csv"
SAIDA_TXT = RAIZ / "relatorio_normalizacao_pessoa.txt"
MIN_CLASSE = 30
NOVAS = ["U:p4", "U:p5", "U:p6"]


def modelos():
    return {
        "SVM": lambda: Pipeline([("sc", StandardScaler()),
                                 ("clf", SVC(class_weight="balanced"))]),
        "Reg_Logistica": lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random_Forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1),
    }


def normalizar(X, grupo, modo):
    if modo == "nenhuma":
        return X
    Xn = X.copy()
    for g in set(grupo):
        m = grupo == g
        mu = X[m].mean(axis=0)
        if modo == "centro":
            Xn[m] = X[m] - mu
        else:  # zscore
            Xn[m] = (X[m] - mu) / (X[m].std(axis=0) + 1e-6)
    return Xn


def main():
    d = np.load(CACHE, allow_pickle=True)
    X_bruto, y, fonte, grupo = d["X"], d["y"], d["fonte"], d["grupo"]
    X_geo = np.array([geometricas_do_vetor(x) for x in X_bruto])

    ok2 = {g: ((grupo == g) & (y == 1)).sum() >= MIN_CLASSE
           and ((grupo == g) & (y == 0)).sum() >= MIN_CLASSE
           for g in set(grupo)}
    grupos_teste = sorted(g for g, v in ok2.items() if v)
    fontes = ["unificado", "dataset", "EED"]
    testes = [("LOPO", g, grupo == g) for g in grupos_teste] + \
             [("Fonte_de_fora", f, fonte == f) for f in fontes]
    eh_2classes = np.array([ok2[g] for g in grupo])

    configs = [(f, "nenhuma", "todos") for f in ("bruto", "geo")]
    for f in ("bruto", "geo"):
        for modo in ("centro", "zscore"):
            for treino in ("todos", "2_classes"):
                configs.append((f, modo, treino))

    linhas = []
    for feat, modo, treino in tqdm(configs, desc="configuracoes"):
        base = X_bruto if feat == "bruto" else X_geo
        X = normalizar(base, grupo, modo)
        permitido = np.ones(len(y), bool) if treino == "todos" else eh_2classes
        for protocolo, nome, teste in testes:
            tr = permitido & ~teste
            if len(set(y[tr])) < 2:
                continue
            for mod, fab in modelos().items():
                pred = fab().fit(X[tr], y[tr]).predict(X[teste])
                linhas.append({
                    "features": feat, "normalizacao": modo, "treino": treino,
                    "protocolo": protocolo, "teste": nome, "modelo": mod,
                    "acuracia_bal": balanced_accuracy_score(y[teste], pred),
                    "prop_prevista_engajado": float(pred.mean())})
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA_CSV, index=False)

    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t); f.write(t + "\n")

        w("NORMALIZACAO POR PESSOA (binario, acaso da acuracia balanceada = 0,50)")
        w("=" * 78)
        w(f"Imagens: {len(y)} | grupos de teste LOPO: {', '.join(grupos_teste)}")
        df["config"] = (df.features + " | " + df.normalizacao + " | treino=" +
                        df.treino)
        for prot in ("LOPO", "Fonte_de_fora"):
            sub = df[df.protocolo == prot]
            w(f"\n{'#' * 78}\n{prot}: nota media entre os grupos de teste\n{'#' * 78}")
            w(sub.groupby(["config", "modelo"]).acuracia_bal.mean()
              .unstack("modelo").round(3).to_string())
        sub = df[df.protocolo == "LOPO"]
        w(f"\n{'#' * 78}\nLOPO: so as pessoas novas (p4, p5, p6)\n{'#' * 78}")
        w(sub[sub.teste.isin(NOVAS)].groupby(["config", "modelo"]).acuracia_bal
          .mean().unstack("modelo").round(3).to_string())
        w(f"\n{'#' * 78}\nLOPO: detalhe por pessoa (SVM)\n{'#' * 78}")
        w(sub[sub.modelo == "SVM"].pivot_table(
            index="config", columns="teste", values="acuracia_bal")
          .round(2).to_string())
        w(f"\n{'#' * 78}\nProporcao prevista como 'engajado' (SVM, LOPO; "
          f"proximo de 0 ou 1 em tudo = modelo colapsou)\n{'#' * 78}")
        w(sub[(sub.modelo == "SVM") & sub.teste.isin(NOVAS + ["U:p1", "U:p2", "U:p3"])]
          .pivot_table(index="config", columns="teste",
                       values="prop_prevista_engajado").round(2).to_string())
    print(f"\nSalvo: {SAIDA_TXT.name}, {SAIDA_CSV.name}")


if __name__ == "__main__":
    main()
