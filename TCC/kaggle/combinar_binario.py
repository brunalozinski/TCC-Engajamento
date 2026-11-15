"""
Combina DAiSEE + EngageNet (embeddings DINOv2 ja calculados) em 2 classes e
monta o dataset Kaggle `daisee-engagenet-binario` (nao mexe nos existentes).

Classes: 0 = nao engajado (originais 0 e 1), 1 = engajado (originais 2 e 3).
Igualdade: em cada divisao oficial (Train/Validation/Test) as duas classes ficam
com o mesmo numero de videos, e a mistura de fontes (DAiSEE/EngageNet) e a mesma
nas duas classes (a fonte nao serve de atalho). A classe menor define o tamanho;
na maior sorteia-se (semente fixa) dentro de cada fonte.
Sequencia: 3 frames por video (o DAiSEE so tem 3; do EngageNet pega 3 dos 16,
espacados igualmente).

Saida: combinado_dataset/{Train,Validation,Test}.npz + dataset-metadata.json
  X (n,3,1536) float16, y (0/1), y_orig (0..3), fonte, pessoa, clipe,
  balanceado (bool; no Test, False = video extra fora do teste balanceado)
  Train e Validation so trazem os videos balanceados.

Uso:  python combinar_binario.py
"""

import json
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "combinado_dataset"
USUARIO, DATASET = "brunalozinski", "daisee-engagenet-binario"
FONTES = {"DAiSEE": AQUI / "saida_daisee" / "emb",
          "EngageNet": AQUI / "saida_engagenet" / "emb"}
IDX16 = [2, 8, 13]      # 3 frames espacados entre os 16 do EngageNet
SEMENTE = 0


def carregar(fonte, div):
    d = np.load(FONTES[fonte] / f"{div}.npz", allow_pickle=True)
    X = d["X"]
    if X.shape[1] == 16:
        X = X[:, IDX16]
    assert X.shape[1] == 3
    y0 = d["y"].astype(int)
    return dict(X=X, y_orig=y0, y=(y0 >= 2).astype(int),
                fonte=np.full(len(y0), fonte), pessoa=d["pessoa"].astype(str),
                clipe=d["clipe"].astype(str))


def juntar(partes):
    return {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}


def main():
    rng = np.random.default_rng(SEMENTE)
    SAIDA.mkdir(exist_ok=True)
    for div in ("Train", "Validation", "Test"):
        d = juntar([carregar(f, div) for f in FONTES])
        # nao engajados definem o tamanho de cada fonte; engajados sao sorteados igual
        escolhidos = []
        for f in FONTES:
            n_nao = ((d["fonte"] == f) & (d["y"] == 0)).sum()
            escolhidos.append(np.where((d["fonte"] == f) & (d["y"] == 0))[0])
            pos = np.where((d["fonte"] == f) & (d["y"] == 1))[0]
            escolhidos.append(rng.choice(pos, min(n_nao, len(pos)), replace=False))
        sel = np.sort(np.concatenate(escolhidos))
        bal = np.zeros(len(d["y"]), bool); bal[sel] = True
        idx = np.arange(len(bal)) if div == "Test" else sel
        out = {k: v[idx] for k, v in d.items()}
        out["balanceado"] = bal[idx]
        np.savez(SAIDA / f"{div}.npz", **out)
        b = out["balanceado"]
        print(div, "salvos:", len(idx), "| balanceados:", b.sum())
        for f in FONTES:
            m = b & (out["fonte"] == f)
            print(f"  {f:10s} nao={int((m & (out['y'] == 0)).sum()):5d}  eng={int((m & (out['y'] == 1)).sum()):5d}")
    (SAIDA / "dataset-metadata.json").write_text(json.dumps({
        "title": "DAiSEE EngageNet binario", "id": f"{USUARIO}/{DATASET}",
        "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
