"""
ETAPA 2b: o mesmo teste da etapa 2, agora mais confiavel.

Tres ajustes, todos simples de explicar:

  1) VARIAS SEMENTES. Random Forest e MLP dependem de um sorteio (semente).
     Treinamos com SEMENTES diferentes e reportamos media e desvio. SVM e
     Regressao Logistica nao sorteiam nada, entao rodam uma vez so.

  2) COM x SEM O EED. O EED mistura emocao (sorriso) com engajamento. Para ver
     se ele distorce a conclusao, repetimos tudo treinando e testando SEM ele.

  3) ABLACAO POR GRUPO DE FEATURES. Tiramos um grupo de cada vez das 20
     medidas geometricas (olhos, boca, sobrancelha, cabeca, rosto) e vemos o
     quanto a nota cai. Se cai muito, o grupo importa. Feito com o SVM, que foi
     o modelo com sinal claro na etapa 2.

Protocolo (igual ao da etapa 2): uma fonte de dados inteira de fora, e uma
pessoa de fora por vez (so grupos com >= 30 imagens de cada classe). Metrica:
acuracia balanceada (acaso = 0,50).

Gera: resultados_etapa2b.csv, relatorio_etapa2b_robustez.txt
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from features import RAIZ, geometricas_do_vetor, NOMES_GEOMETRICAS
from avaliar_multi_datasets import CACHE

warnings.filterwarnings("ignore")

SAIDA_CSV = RAIZ / "resultados_etapa2b.csv"
SAIDA_TXT = RAIZ / "relatorio_etapa2b_robustez.txt"
SEMENTES = [0, 1, 2, 3, 4]
MIN_CLASSE = 30

GRUPOS_GEO = {
    "olhos": ["ear_esq", "ear_dir", "ear_medio", "ear_diferenca"],
    "boca": ["mar_interno", "boca_largura", "boca_altura", "boca_curvatura"],
    "sobrancelha": ["sobr_olho_esq", "sobr_olho_dir", "sobr_olho_medio",
                    "sobr_inclin_esq", "sobr_inclin_dir",
                    "sobr_distancia_interna"],
    "cabeca": ["yaw", "pitch", "roll"],
    "rosto": ["rosto_razao_largura_altura", "queixo_nariz", "nariz_linha_olhos"],
}


def modelo(nome, semente):
    if nome == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", class_weight="balanced"))])
    if nome == "Reg_Logistica":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))])
    if nome == "Random_Forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=semente, n_jobs=-1)
    if nome == "MLP":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(256, 128),
                                               max_iter=300,
                                               random_state=semente))])
    raise ValueError(nome)


ALEATORIOS = {"Random_Forest", "MLP"}


def testes(fonte, grupo, y, usar_eed):
    """Lista de (protocolo, nome, mascara_teste, mascara_treino_permitido)."""
    permitido = np.ones(len(y), bool) if usar_eed else (fonte != "EED")
    fontes = ["unificado", "dataset"] + (["EED"] if usar_eed else [])
    saida = [("Fonte_de_fora", f, fonte == f, permitido) for f in fontes]
    for g in sorted(set(grupo[permitido])):
        m = grupo == g
        if ((m & (y == 1)).sum() >= MIN_CLASSE
                and (m & (y == 0)).sum() >= MIN_CLASSE):
            saida.append(("LOPO", g, m, permitido))
    return saida


def rodar(X, y, lista_testes, modelos, rotulo):
    linhas = []
    for protocolo, nome, teste, permitido in tqdm(lista_testes, desc=rotulo):
        treino = permitido & ~teste
        for mod in modelos:
            sementes = SEMENTES if mod in ALEATORIOS else [0]
            notas = []
            for s in sementes:
                p = modelo(mod, s).fit(X[treino], y[treino]).predict(X[teste])
                notas.append(balanced_accuracy_score(y[teste], p))
            linhas.append({"protocolo": protocolo, "teste": nome, "modelo": mod,
                           "nota_media": np.mean(notas),
                           "nota_desvio_sementes": np.std(notas)})
    return pd.DataFrame(linhas)


def main():
    d = np.load(CACHE, allow_pickle=True)
    X_bruto, y, fonte, grupo = d["X"], d["y"], d["fonte"], d["grupo"]
    X_geo = np.array([geometricas_do_vetor(x) for x in X_bruto])
    idx = {n: i for i, n in enumerate(NOMES_GEOMETRICAS)}

    todos = []
    for usar_eed in (True, False):
        cenario = "com_EED" if usar_eed else "sem_EED"
        lista = testes(fonte, grupo, y, usar_eed)
        # 1) e 2): bruto x geometrico, 4 modelos, varias sementes
        for nome_f, X in (("bruto", X_bruto), ("geometrico", X_geo)):
            t = rodar(X, y, lista, ["SVM", "Reg_Logistica", "Random_Forest", "MLP"],
                      f"{cenario} {nome_f}")
            t["cenario"] = cenario; t["features"] = nome_f
            todos.append(t)
        # 3) ablacao por grupo (SVM)
        for g, cols in GRUPOS_GEO.items():
            manter = [i for n, i in idx.items() if n not in cols]
            t = rodar(X_geo[:, manter], y, lista, ["SVM"], f"{cenario} sem {g}")
            t["cenario"] = cenario; t["features"] = f"geo_sem_{g}"
            todos.append(t)
    df = pd.concat(todos)
    df.to_csv(SAIDA_CSV, index=False)

    with open(SAIDA_TXT, "w", encoding="utf-8") as f:
        def w(t=""):
            print(t); f.write(t + "\n")

        w("ETAPA 2b - ROBUSTEZ: varias sementes, com/sem EED, ablacao por grupo")
        w("=" * 76)
        w(f"Sementes (Random Forest e MLP): {SEMENTES}. Acaso = 0,50.")
        for cenario in ("com_EED", "sem_EED"):
            c = df[df.cenario == cenario]
            w(f"\n{'#' * 76}\nCENARIO: {cenario.replace('_', ' ')}\n{'#' * 76}")
            for prot in ("Fonte_de_fora", "LOPO"):
                p = c[(c.protocolo == prot) & c.features.isin(["bruto", "geometrico"])]
                w(f"\n-- {prot}: nota media entre os grupos de teste "
                  f"({p.teste.nunique()} grupos) --")
                tab = p.groupby(["features", "modelo"]).nota_media.agg(["mean", "std"])
                w(tab.round(3).unstack("modelo").to_string())
                # diferenca geometrico - bruto por modelo, e em quantos grupos ganhou
                piv = p.pivot_table(index=["teste", "modelo"], columns="features",
                                    values="nota_media")
                piv["dif"] = piv["geometrico"] - piv["bruto"]
                res = piv.groupby("modelo").dif.agg(
                    media_dif="mean", grupos_onde_geo_ganha=lambda s: int((s > 0).sum()),
                    total_grupos="count")
                w("\n   Geometrico menos bruto (positivo = geometrico melhor):")
                w(res.round(3).to_string())
                # estabilidade com sementes
                inst = p[p.modelo.isin(ALEATORIOS)].groupby(
                    ["features", "modelo"]).nota_desvio_sementes.mean()
                w("\n   Desvio medio causado so pela semente (RF e MLP):")
                w(inst.round(3).to_string())
            w(f"\n-- Ablacao por grupo de features (SVM), {cenario} --")
            ab = c[(c.modelo == "SVM") & c.features.str.startswith("geo")]
            ab = ab.copy()
            ab["features"] = ab.features.replace({"geometrico": "geo_completo"})
            tab = ab.groupby(["protocolo", "features"]).nota_media.mean().unstack(
                "protocolo").round(3)
            ref = tab.loc["geo_completo"]
            tab["queda_Fonte"] = (ref["Fonte_de_fora"] - tab["Fonte_de_fora"]).round(3)
            tab["queda_LOPO"] = (ref["LOPO"] - tab["LOPO"]).round(3)
            w(tab.to_string())
            w("(queda positiva = tirar o grupo piorou; o grupo importa)")
        w("\n-- Detalhe por grupo de teste (bruto x geometrico) --")
        det = df[df.features.isin(["bruto", "geometrico"])].pivot_table(
            index=["cenario", "protocolo", "teste", "features"], columns="modelo",
            values="nota_media")
        w(det.round(3).to_string())
    print(f"\nSalvo: {SAIDA_TXT.name}, {SAIDA_CSV.name}")


if __name__ == "__main__":
    main()
