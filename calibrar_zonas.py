"""
Calibra as zonas de engajamento da demo (Baixo / Indefinido / Alto) com a VALIDACAO.

O modelo so devolve P(engajado) (2 classes). As 3 zonas da demo sao uma camada de
apresentacao sobre essa probabilidade, e os dois cortes sao escolhidos assim, usando
SOMENTE os videos da Validacao (o Teste nao e usado para escolher nada):

  corte_baixo : o MAIOR valor tal que, entre os videos com P <= corte_baixo, pelo menos
                ALVO_PRECISAO sao realmente "nao engajado".
  corte_alto  : o MENOR valor tal que, entre os videos com P >= corte_alto, pelo menos
                ALVO_PRECISAO sao realmente "engajado".

Entre os dois cortes fica a zona "Indefinido": o modelo esta em duvida. Depois de
escolhidos, os cortes sao avaliados no Teste (a partir das predicoes salvas do treino)
e o resultado vai para calibracao_zonas/relatorio_<experimento>.txt.

Uso (Python 3.11 do projeto):
    python calibrar_zonas.py C
    python calibrar_zonas.py E        # calcula embeddings da Validacao (~12 min na CPU)
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent
KAGGLE = RAIZ / "TCC" / "kaggle"
SAIDA = RAIZ / "calibracao_zonas"

ALVO_PRECISAO = 0.80      # pureza minima de cada zona extrema
ALVO_FALLBACK = 0.75      # usado se 0,80 nao for alcancavel
MIN_VIDEOS = 20           # uma zona extrema precisa de pelo menos 20 videos na Validacao


def escolher_cortes(p, y, alvo):
    """p = P(engajado), y = 1 engajado. Devolve (corte_baixo, corte_alto) ou None."""
    baixo = alto = None
    for a in np.unique(p):                       # candidatos em ordem crescente
        sel = p <= a
        if sel.sum() >= MIN_VIDEOS and (y[sel] == 0).mean() >= alvo:
            baixo = a                            # fica com o maior que satisfaz
    for b in np.unique(p)[::-1]:
        sel = p >= b
        if sel.sum() >= MIN_VIDEOS and (y[sel] == 1).mean() >= alvo:
            alto = b                             # fica com o menor que satisfaz
    return baixo, alto


def resumo_zonas(p, y, baixo, alto, titulo):
    zonas = {
        "Baixo (P <= %.2f)" % baixo: (p <= baixo, 0),
        "Indefinido": ((p > baixo) & (p < alto), None),
        "Alto (P >= %.2f)" % alto: (p >= alto, 1),
    }
    linhas = [titulo, "  zona                      videos   % dos videos   pureza (acerto da zona)"]
    for nome, (sel, classe) in zonas.items():
        n = int(sel.sum())
        pureza = "-" if classe is None or n == 0 else f"{(y[sel] == classe).mean():.1%}"
        if classe is None and n:
            pureza = f"{(y[sel] == 1).mean():.1%} engajados"
        linhas.append(f"  {nome:<25} {n:>6}   {n / len(p):>10.1%}   {pureza}")
    return "\n".join(linhas)


def probabilidades_validacao(variante):
    if variante == "C":
        from motor_dinov2_lstm import ModeloLSTM, PASTA_MODELO
        modelo = ModeloLSTM()
        modelo.load_state_dict(torch.load(PASTA_MODELO / "lstm.pt", map_location="cpu", weights_only=True))
        modelo.eval()
        norm = np.load(PASTA_MODELO / "normalizacao.npz")
        v = np.load(KAGGLE / "combinado_dataset" / "Validation.npz", allow_pickle=True)
        x = ((v["X"].astype(np.float32) - norm["mu"]) / norm["sd"]).astype(np.float32)
        with torch.no_grad():
            p = modelo(torch.from_numpy(x)).softmax(-1)[:, 1].numpy()
        return p, v["y"].astype(int), len(p)
    # E: precisa dos embeddings do DINOv2 ajustado, entao recalcula a partir das imagens
    from motor_dinov2_lstm import criar_motor
    motor = criar_motor("E")
    man = pd.read_csv(KAGGLE / "combinado_imagens" / "manifesto_imagens.csv",
                      dtype={"pessoa": str, "clipe": str})
    m = man[man.divisao == "Validation"].reset_index(drop=True)
    SAIDA.mkdir(exist_ok=True)
    cache = SAIDA / "embeddings_validacao_E.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        emb = np.zeros((len(m), 3, 1536), dtype=np.float32)
        for i, r in enumerate(m.itertuples()):
            for k in range(3):
                img = cv2.imread(str(KAGGLE / "combinado_imagens" / r.divisao / r.pasta / f"{k:02d}.jpg"))
                emb[i, k] = motor.embedding(img)
            if i % 50 == 0:
                print(f"embeddings da Validacao: {i}/{len(m)}", flush=True)
        np.save(cache, emb)
    p = np.array([motor.p_engajado_sequencia(e) for e in emb])
    return p, m.y.values.astype(int), len(p)


def main(variante):
    SAIDA.mkdir(exist_ok=True)
    p, y, n = probabilidades_validacao(variante)
    alvo = ALVO_PRECISAO
    baixo, alto = escolher_cortes(p, y, alvo)
    if baixo is None or alto is None:
        alvo = ALVO_FALLBACK
        baixo, alto = escolher_cortes(p, y, alvo)
    assert baixo is not None and alto is not None and baixo < alto, "nao foi possivel calibrar"
    baixo = math.floor(baixo * 100) / 100        # arredonda para o lado conservador
    alto = math.ceil(alto * 100) / 100

    pred = {"C": KAGGLE / "saida_combinado" / "predicoes_daisee-engagenet-binario.csv",
            "E": KAGGLE / "saida_finetune" / "predicoes_dinov2-finetune-k2.csv"}[variante]
    t = pd.read_csv(pred)
    bal = t.balanceado.values.astype(bool)

    texto = [f"Calibracao das zonas, experimento {variante}",
             f"Base de calibracao: Validacao balanceada, {n} videos ({int(y.sum())} engajados).",
             f"Criterio: pureza >= {alvo:.0%} nas zonas extremas, com pelo menos {MIN_VIDEOS} videos.",
             f"Cortes escolhidos: corte_baixo = {baixo:.2f}, corte_alto = {alto:.2f}", "",
             resumo_zonas(p, y, baixo, alto, "VALIDACAO (onde os cortes foram escolhidos)"), "",
             resumo_zonas(t.p_engajado.values[bal], t.y.values[bal], baixo, alto,
                          "TESTE balanceado (1.564 videos, nao usado para escolher)"), "",
             resumo_zonas(t.p_engajado.values, t.y.values, baixo, alto,
                          "TESTE completo, distribuicao natural (4.030 videos, 81% engajados)"), "",
             "Observacoes:",
             "- A pureza foi calibrada em dados balanceados; com mais engajados no uso real, "
             "a pureza da zona Baixo cai e a da zona Alto sobe.",
             "- Na demo, a probabilidade e a media de varios sorteios de 3 embeddings da janela, "
             "entao pode ser um pouco menos extrema que a de um clipe isolado."]
    (SAIDA / f"relatorio_{variante}.txt").write_text("\n".join(texto), encoding="utf-8")
    pasta = {"C": RAIZ / "modelo_dinov2_lstm", "E": RAIZ / "modelo_dinov2_finetune"}[variante]
    (pasta / "zonas.json").write_text(json.dumps({
        "corte_baixo": baixo, "corte_alto": alto, "pureza_alvo": alvo,
        "base": f"Validacao balanceada ({n} videos)"}, indent=1), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ))
    main(sys.argv[1].upper())
