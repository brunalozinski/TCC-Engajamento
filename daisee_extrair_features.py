"""
DAiSEE: extrai os 68 pontos do rosto (normalizados, 136 numeros) de 8 frames de
cada clipe de 10 s. Usa SOMENTE a pasta C:\\TCC\\DAiSEE.

Por clipe: 8 frames espacados de 5% a 95% da duracao; em cada um roda o mesmo
detector do projeto (dlib, com MediaPipe de reserva) e a mesma normalizacao de
features.py. Frames sem rosto ficam como NaN (guarda-se quantos deram certo).

Divisao e rotulos: os CSVs oficiais em DAiSEE/Labels (Train/Validation/Test, ja
separados por pessoa: nenhuma pessoa aparece em duas divisoes). A pessoa e os 6
primeiros digitos do ClipID.

Retomavel: salva a cada BLOCO clipes em daisee_resultados/daisee_features.npz e,
se for interrompido, continua de onde parou.

Uso:
    python daisee_extrair_features.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import RAIZ, criar_detector_predictor, extrair_features

DAISEE = RAIZ / "DAiSEE"
SAIDA = RAIZ / "daisee_resultados"
ARQUIVO = SAIDA / "daisee_features.npz"
N_FRAMES = 8
FRACOES = np.linspace(0.05, 0.95, N_FRAMES)
BLOCO = 400


def ler_rotulos():
    partes = []
    for divisao in ("Train", "Validation", "Test"):
        d = pd.read_csv(DAISEE / "Labels" / f"{divisao}Labels.csv")
        d.columns = [c.strip() for c in d.columns]
        d["divisao"] = divisao
        partes.append(d)
    a = pd.concat(partes, ignore_index=True)
    a["clipe"] = a.ClipID.str.replace(r"\.\w+$", "", regex=True)
    a["pessoa"] = a.ClipID.str[:6]
    return a


def localizar_videos():
    return {p.stem: p for p in (DAISEE / "DataSet").rglob("*")
            if p.suffix.lower() in (".avi", ".mp4")}


def extrair_clipe(caminho, detector, predictor):
    cap = cv2.VideoCapture(str(caminho))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saida = np.full((N_FRAMES, 136), np.nan, dtype=np.float32)
    if n <= 0:
        cap.release()
        return saida
    for i, frac in enumerate(FRACOES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(n * frac), n - 1))
        ok, frame = cap.read()
        if not ok:
            continue
        feats, _ = extrair_features(frame, detector, predictor)
        if feats is not None:
            saida[i] = feats
    cap.release()
    return saida


def salvar(ids, X, rot):
    SAIDA.mkdir(exist_ok=True)
    r = rot.set_index("clipe").loc[ids]
    np.savez_compressed(
        ARQUIVO, clip=np.array(ids), X=np.array(X),
        pessoa=r.pessoa.values, divisao=r.divisao.values,
        engajamento=r.Engagement.values, tedio=r.Boredom.values,
        confusao=r.Confusion.values, frustracao=r.Frustration.values)


def main():
    rot = ler_rotulos()
    videos = localizar_videos()
    rot = rot[rot["clipe"].isin(videos)].drop_duplicates("clipe")
    print(f"{len(rot)} clipes rotulados com video; {rot.pessoa.nunique()} pessoas.")

    ids, X = [], []
    if ARQUIVO.exists():
        d = np.load(ARQUIVO, allow_pickle=True)
        ids, X = list(d["clip"]), list(d["X"])
        print(f"Retomando: {len(ids)} clipes ja processados.")
    feitos = set(ids)
    pendentes = [c for c in rot["clipe"] if c not in feitos]

    detector, predictor = criar_detector_predictor(com_reserva=True)
    t0 = time.time()
    for k, clip in enumerate(tqdm(pendentes, desc="DAiSEE"), 1):
        ids.append(clip)
        X.append(extrair_clipe(videos[clip], detector, predictor))
        if k % BLOCO == 0:
            salvar(ids, X, rot)
    salvar(ids, X, rot)

    arr = np.array(X)
    com_rosto = np.isfinite(arr[:, :, 0]).sum(axis=1)
    print(f"\nConcluido em {(time.time() - t0) / 60:.1f} min. "
          f"Clipes: {len(ids)}; frames com rosto: "
          f"{int(com_rosto.sum())}/{arr.shape[0] * N_FRAMES}; "
          f"clipes sem nenhum rosto: {int((com_rosto == 0).sum())}; "
          f"frames resgatados pelo MediaPipe: {detector.usos_reserva}")
    print(f"Salvo: {ARQUIVO}")


if __name__ == "__main__":
    main()
