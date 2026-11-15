"""
EngageNet -> sequencias de recortes de rosto (para embeddings + LSTM).
Usa SOMENTE C:\\TCC\\EngageNet-Dataset (Train/Validation/Test + rotulos).

Por video: N_FRAMES frames igualmente espacados; em cada um recorta o maior
rosto com face_crop.recortar_rosto (o MESMO recorte do DAiSEE) e salva 224x224.
Frame sem rosto repete o ultimo recorte valido. Video sem nenhum rosto fica no
manifesto como "sem_rosto". Videos da classe SNP (Subject Not Present) sao
ignorados. Rotulos: Not-Engaged=0, Barely-engaged=1, Engaged=2, Highly-Engaged=3.

Saida (TCC/engagenet_seq/):
  <Train|Validation|Test>/<video>/<k>.jpg
  manifesto_seq.csv  (divisao, pessoa, clipe, engajamento, n_ok, status)

Retomavel: pula videos ja no manifesto; videos que deram erro (ex.: arquivo
ainda sendo baixado) nao entram no manifesto e sao tentados de novo.

Uso:
    python engagenet_sequencias.py Validation Test
    python engagenet_sequencias.py Train
    python engagenet_sequencias.py --zip      # gera engagenet_seq.zip
"""

import argparse
import csv
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from face_crop import recortar_rosto  # noqa: E402

RAIZ = AQUI.parent / "EngageNet-Dataset"
SAIDA = AQUI / "engagenet_seq"
N_FRAMES = 16
TAMANHO = 224
CLASSES = {"not-engaged": 0, "barely-engaged": 1, "engaged": 2,
           "highly-engaged": 3}
ROTULOS = {"Train": "train_engagement_labels.xlsx",
           "Validation": "validation_engagement_labels.xlsx",
           "Test": "test_engagement_labels.csv"}


def ler_rotulos(divisao):
    arq = RAIZ / ROTULOS[divisao]
    d = pd.read_excel(arq) if arq.suffix == ".xlsx" else pd.read_csv(arq)
    d["classe"] = d.label.str.strip().str.lower().map(CLASSES)
    d = d.dropna(subset=["classe"])   # SNP e rotulos desconhecidos ficam fora
    d["classe"] = d.classe.astype(int)
    d["clipe"] = d.chunk.str.replace(r"\.mp4$", "", regex=True)
    d["pessoa"] = d.chunk.str.extract(r"(subject_\d+)")[0]
    return d


def frames_do_video(caminho):
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


def processar(args):
    divisao, pessoa, clipe, classe = args
    try:
        frames = frames_do_video(RAIZ / divisao / f"{clipe}.mp4")
        if not frames or all(f is None for f in frames):
            return None   # ilegivel (ex.: ainda baixando): tenta de novo depois
        recortes = [recortar_rosto(f, tamanho=TAMANHO)[0] if f is not None
                    else None for f in frames]
        validos = [r for r in recortes if r is not None]
        if validos:
            pasta = SAIDA / divisao / clipe
            pasta.mkdir(parents=True, exist_ok=True)
            ultimo = validos[0]
            for k, r in enumerate(recortes):
                r = ultimo if r is None else r
                ultimo = r
                cv2.imwrite(str(pasta / f"{k:02d}.jpg"), r,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        return (divisao, pessoa, clipe, classe, len(validos),
                "ok" if validos else "sem_rosto")
    except Exception:
        return None


def extrair(divisoes, processos):
    SAIDA.mkdir(exist_ok=True)
    man = SAIDA / "manifesto_seq.csv"
    novo = not man.exists()
    feitos = set(pd.read_csv(man, dtype=str).clipe) if not novo else set()
    tarefas = []
    for div in divisoes:
        existentes = {p.stem for p in (RAIZ / div).glob("*.mp4")}
        d = ler_rotulos(div)
        d = d[d.clipe.isin(existentes) & ~d.clipe.isin(feitos)]
        tarefas += [(div, r.pessoa, r.clipe, r.classe) for r in d.itertuples()]
    print(f"{len(tarefas)} videos a processar ({', '.join(divisoes)}).")
    t0 = time.time()
    falhas = 0
    with open(man, "a", newline="", encoding="utf-8") as f, Pool(processos) as p:
        w = csv.writer(f)
        if novo:
            w.writerow(["divisao", "pessoa", "clipe", "engajamento", "n_ok",
                        "status"])
        for i, linha in enumerate(tqdm(p.imap_unordered(processar, tarefas,
                                                        chunksize=4),
                                       total=len(tarefas)), 1):
            if linha is None:
                falhas += 1
                continue
            w.writerow(linha)
            if i % 50 == 0:
                f.flush()
    m = pd.read_csv(man)
    print(f"\nConcluido em {(time.time() - t0) / 60:.1f} min; "
          f"{falhas} videos com erro (serao refeitos na proxima rodada).")
    print(m.groupby(["divisao", "status"]).size().unstack(fill_value=0).to_string())
    print(m[m.status == "ok"].groupby(["divisao", "engajamento"]).size()
          .unstack(fill_value=0).to_string())


def empacotar():
    print(f"Gerando {SAIDA}.zip ...")
    shutil.make_archive(str(SAIDA), "zip", root_dir=SAIDA)
    print("Pronto.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("divisoes", nargs="*", default=[])
    ap.add_argument("--processos", type=int, default=7)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    empacotar() if a.zip else extrair(a.divisoes, a.processos)
