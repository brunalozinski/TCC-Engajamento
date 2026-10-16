"""
Inclui as pessoas novas (pessoa4, pessoa5, pessoa6) no ER_Dataset_Unificado e
nos caches de features, SEM reextrair nada do que ja existe.

O que faz, em ordem:
  1) Guarda copias dos resultados e caches ANTERIORES em
     resultados_antes_pessoas_4_6/ (a documentacao cita esses numeros).
  2) Copia os frames novos de ER_Dataset_Externo/<nivel>/ para
     ER_Dataset_Unificado/<nivel>/ (so os que ainda nao estao la).
  3) Extrai os 68 pontos desses frames com o detector COM RESERVA (dlib +
     MediaPipe; a pessoa6 foi filmada de baixo para cima e o dlib sozinho quase
     nao acha o rosto dela).
  4) Acrescenta as linhas novas em features_unificado.npz e reconstroi
     features_multi.npz (mantem as linhas de dataset e EED como estavam).

As imagens antigas NAO sao reprocessadas: continuam com o detector dlib de
sempre, entao os numeros anteriores seguem comparaveis. Rodar de novo e seguro
(pula o que ja foi incluido).

Uso:
    python incluir_pessoas_novas.py
"""

import re
import shutil
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from features import RAIZ, criar_detector_predictor, extrair_features
from avaliar_loo import pessoa_do_arquivo

EXTERNO = RAIZ / "ER_Dataset_Externo"
UNIFICADO = RAIZ / "ER_Dataset_Unificado"
CACHE_U = RAIZ / "features_unificado.npz"
CACHE_M = RAIZ / "features_multi.npz"
ANTES = RAIZ / "resultados_antes_pessoas_4_6"
PADRAO_NOVAS = re.compile(r"^pessoa[456]_sessao\d+_classe[0-3]_\d+\.jpg$")


def guardar_resultados_anteriores():
    ANTES.mkdir(exist_ok=True)
    padroes = ["relatorio_*.txt", "resultados_*.csv", "features_*.npz"]
    n = 0
    for pad in padroes:
        for f in RAIZ.glob(pad):
            destino = ANTES / f.name
            if not destino.exists():          # nunca sobrescreve a copia antiga
                shutil.copy2(f, destino); n += 1
    print(f"Resultados/caches anteriores guardados em {ANTES.name}/ (+{n} arquivos).")


def main():
    guardar_resultados_anteriores()

    du = np.load(CACHE_U, allow_pickle=True)
    X_u, y_u, pessoas_u, nomes_u = (list(du["X"]), list(du["y"]),
                                    list(du["pessoas"]), list(du["nomes"]))
    ja_incluidos = set(nomes_u)

    novas = []
    for nivel in "0123":
        for f in sorted((EXTERNO / nivel).glob("*.jpg")):
            if PADRAO_NOVAS.match(f.name) and f.name not in ja_incluidos:
                novas.append((f, int(nivel)))
    print(f"{len(novas)} frames novos para incluir.")
    if not novas:
        return

    # 2) copia para o unificado
    for f, nivel in novas:
        destino = UNIFICADO / str(nivel)
        destino.mkdir(parents=True, exist_ok=True)
        if not (destino / f.name).exists():
            shutil.copy2(f, destino / f.name)

    # 3) extrai com detector de reserva
    detector, predictor = criar_detector_predictor(com_reserva=True)
    sem_rosto = Counter()
    novos = 0
    for f, nivel in tqdm(novas, desc="Extraindo pessoas novas"):
        img = cv2.imread(str(f))
        feats = extrair_features(img, detector, predictor)[0] if img is not None else None
        if feats is None:
            sem_rosto[pessoa_do_arquivo(f.name)] += 1
            continue
        X_u.append(feats); y_u.append(nivel)
        pessoas_u.append(pessoa_do_arquivo(f.name)); nomes_u.append(f.name)
        novos += 1
    print(f"{novos} frames com 68 pontos; sem rosto: {dict(sem_rosto) or 'nenhum'}; "
          f"resgatados pelo MediaPipe: {detector.usos_reserva}")

    # 4) grava caches
    np.savez(CACHE_U, X=np.array(X_u), y=np.array(y_u),
             pessoas=np.array(pessoas_u), nomes=np.array(nomes_u))

    dm = np.load(CACHE_M, allow_pickle=True)
    manter = dm["fonte"] != "unificado"
    X_m = np.vstack([np.array(X_u), dm["X"][manter]])
    y_m = np.concatenate([(np.array(y_u) >= 2).astype(int), dm["y"][manter]])
    fonte_m = np.concatenate([np.array(["unificado"] * len(X_u)), dm["fonte"][manter]])
    grupo_m = np.concatenate([np.array(["U:" + p for p in pessoas_u]),
                              dm["grupo"][manter]])
    np.savez(CACHE_M, X=X_m, y=y_m, fonte=fonte_m, grupo=grupo_m,
             sem_rosto=dm["sem_rosto"])

    print("\nPor pessoa gravada (nivel: imagens com 68 pontos):")
    for p in ("p1", "p2", "p3", "p4", "p5", "p6"):
        idx = [i for i, q in enumerate(pessoas_u) if q == p]
        c = Counter(int(y_u[i]) for i in idx)
        print(f"  {p}: " + "  ".join(f"{n}:{c.get(n, 0)}" for n in range(4)) +
              f"  (total {len(idx)})")
    print(f"\nfeatures_multi.npz agora: {len(y_m)} imagens "
          f"(engajado={int((y_m == 1).sum())}, nao engajado={int((y_m == 0).sum())})")


if __name__ == "__main__":
    main()
