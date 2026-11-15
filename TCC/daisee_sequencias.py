"""
DAiSEE -> sequencias de recortes de rosto (para embeddings + LSTM).
Usa SOMENTE C:\\TCC\\DAiSEE.

Por clipe (10 s): N_FRAMES frames igualmente espacados; em cada um recorta o
maior rosto com face_crop.recortar_rosto (o MESMO recorte do restante do
projeto) e salva 224x224. Se um frame nao tem rosto, repete o ultimo recorte
valido do clipe (a sequencia mantem sempre N_FRAMES passos). Clipes sem nenhum
rosto sao registrados no manifesto e ficam fora do treino.

Saida (fine tuning/daisee_seq/):
  <Train|Validation|Test>/<clipe>/<k>.jpg   (k = 00..N_FRAMES-1)
  manifesto_seq.csv  (divisao, pessoa, clipe, engajamento, n_ok, status)

Divisao oficial do DAiSEE (nenhuma pessoa em duas divisoes). Retomavel.

Uso:
    python daisee_sequencias.py --max 20   # teste rapido
    python daisee_sequencias.py            # tudo
    python daisee_sequencias.py --zip      # gera daisee_seq.zip
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from face_crop import recortar_rosto

AQUI = Path(__file__).resolve().parent
DAISEE = AQUI.parent / "DAiSEE"
SAIDA = AQUI / "daisee_seq"
N_FRAMES = 16
TAMANHO = 224


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
    return a.drop_duplicates("clipe")


def localizar_videos():
    return {p.stem: p for p in (DAISEE / "DataSet").rglob("*")
            if p.suffix.lower() in (".avi", ".mp4")}


def frames_do_clipe(caminho):
    """N_FRAMES frames igualmente espacados (centro de cada fatia do clipe)."""
    cap = cv2.VideoCapture(str(caminho))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if n > 0:
        for k in range(N_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    min(int(n * (k + 0.5) / N_FRAMES), n - 1))
            ok, frame = cap.read()
            frames.append(frame if ok else None)
    cap.release()
    return frames


def processar_clipe(caminho, pasta):
    recortes = [None] * N_FRAMES
    for k, frame in enumerate(frames_do_clipe(caminho)):
        if frame is not None:
            recortes[k], _ = recortar_rosto(frame, tamanho=TAMANHO)
    validos = [r for r in recortes if r is not None]
    if not validos:
        return 0
    pasta.mkdir(parents=True, exist_ok=True)
    ultimo = validos[0]   # frames iniciais sem rosto usam o 1o recorte valido
    for k, r in enumerate(recortes):
        if r is None:
            r = ultimo
        ultimo = r
        cv2.imwrite(str(pasta / f"{k:02d}.jpg"), r,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(validos)


def extrair(maximo):
    rot = ler_rotulos()
    videos = localizar_videos()
    rot = rot[rot.clipe.isin(videos)]
    if maximo:
        rot = rot.sample(maximo, random_state=0)
    print(f"{len(rot)} clipes com video; {rot.pessoa.nunique()} pessoas.")

    SAIDA.mkdir(exist_ok=True)
    caminho_man = SAIDA / "manifesto_seq.csv"
    novo = not caminho_man.exists()
    feitos = set(pd.read_csv(caminho_man).clipe.astype(str)) if not novo else set()
    t0 = time.time()
    with open(caminho_man, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["divisao", "pessoa", "clipe", "engajamento", "n_ok",
                        "status"])
        for r in tqdm(list(rot.itertuples()), desc="clipes"):
            if r.clipe in feitos:
                continue
            n_ok = processar_clipe(videos[r.clipe],
                                   SAIDA / r.divisao / r.clipe)
            w.writerow([r.divisao, r.pessoa, r.clipe, r.Engagement, n_ok,
                        "ok" if n_ok else "sem_rosto"])
            f.flush()
    m = pd.read_csv(caminho_man)
    print(f"\nConcluido em {(time.time() - t0) / 60:.1f} min.")
    print(m.groupby(["divisao", "status"]).size().unstack(fill_value=0).to_string())
    print(m[m.status == "ok"].groupby(["divisao", "engajamento"]).size()
          .unstack(fill_value=0).to_string())
    print(f"Pasta: {SAIDA}")


def empacotar():
    print(f"Gerando {SAIDA}.zip ...")
    shutil.make_archive(str(SAIDA), "zip", root_dir=SAIDA)
    print("Pronto.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--max", type=int, default=0, help="so N clipes (teste)")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    empacotar() if a.zip else extrair(a.max)
