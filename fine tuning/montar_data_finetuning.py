"""
Monta a pasta data/ do fine-tuning a partir dos datasets que o projeto JA tem,
sem reabrir nenhum video.

Para cada imagem: detecta o maior rosto, recorta (quadrado, margem de 20%,
256x256 -- ver face_crop.py) e salva em:

    fine tuning/data/engajado/      <pessoa>__<fonte>_<nome_original>.jpg
    fine tuning/data/nao_engajado/  <pessoa>__<fonte>_<nome_original>.jpg

O ID da pessoa e tudo que vem antes de "__" no nome do arquivo. Use isso na
divisao treino/teste do notebook (dividir POR PESSOA, nunca por imagem).

Fontes (mesmas decisoes de avaliar_multi_datasets.py):
  unificado : ER_Dataset_Unificado; niveis 0,1 -> nao_engajado; 2,3 -> engajado
              (pessoas kA, kB, kC, kE, kF = Kaggle; p1, p2, p3 = gravadas)
  dataset   : Desengajado -> nao_engajado; Engajado -> engajado;
              "Parcialmente engajado" EXCLUIDO. Pessoa = dN (de imgN_...).
  eed       : interested/stimulated/fulfilled/satisfied -> engajado;
              distracted -> nao_engajado; demais emocoes EXCLUIDAS.
              O EED nao identifica pessoas: todas contam como "eed".

Nao altera nada fora de "fine tuning/data/". Pode ser interrompido e rodado de
novo: imagens ja geradas sao puladas.

Gera tambem data/manifesto.csv (uma linha por imagem de origem, com o motivo
de descarte quando nao ha rosto) e data/resumo.txt.

Uso:
    python montar_data_finetuning.py
    python montar_data_finetuning.py --limite 40     (teste rapido)
    python montar_data_finetuning.py --refazer       (regera tudo)
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from face_crop import recortar_rosto

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
DATA = AQUI / "data"

UNIFICADO = RAIZ / "ER_Dataset_Unificado"
DATASET_1 = RAIZ / "dataset"
DATASET_2 = RAIZ / "dataset-2" / "EED"

EED_ENGAJADO = {"interested", "stimulated", "fulfilled", "satisfied"}
EED_NAO_ENGAJADO = {"distracted"}


def pessoa_unificado(nome):
    m = re.match(r"kaggle_pessoa([A-Z])_", nome)
    if m:
        return "k" + m.group(1)
    m = re.match(r"pessoa(\d+)_", nome)
    if m:
        return "p" + m.group(1)
    raise ValueError(f"Nome fora do padrao: {nome}")


def listar_imagens():
    """Devolve lista de (caminho, pessoa, fonte, rotulo)."""
    itens = []
    for nivel in "0123":
        rotulo = "engajado" if nivel in "23" else "nao_engajado"
        for p in sorted((UNIFICADO / nivel).glob("*.jpg")):
            itens.append((p, pessoa_unificado(p.name), "unificado", rotulo))

    for pasta, rotulo in (("Engajado", "engajado"),
                          ("Desengajado", "nao_engajado")):
        for p in sorted((DATASET_1 / pasta).glob("*.jpg")):
            m = re.match(r"img(\d+)_", p.name)
            pessoa = "d" + (m.group(1) if m else "x")
            itens.append((p, pessoa, "dataset", rotulo))

    for sub in ("train", "test"):
        for classe in sorted((DATASET_2 / sub).iterdir()):
            if classe.name in EED_ENGAJADO:
                rotulo = "engajado"
            elif classe.name in EED_NAO_ENGAJADO:
                rotulo = "nao_engajado"
            else:
                continue
            for p in sorted(classe.glob("*.jpg")):
                itens.append((p, "eed", "eed", rotulo))
    return itens


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--limite", type=int, default=0,
                    help="Processa so as N primeiras imagens (teste)")
    ap.add_argument("--refazer", action="store_true",
                    help="Regera mesmo as imagens que ja existem")
    args = ap.parse_args()

    itens = listar_imagens()
    if args.limite:
        # amostra espalhada entre as fontes, nao so as primeiras
        passo = max(len(itens) // args.limite, 1)
        itens = itens[::passo][:args.limite]
    print(f"{len(itens)} imagens de origem.")

    for rotulo in ("engajado", "nao_engajado"):
        (DATA / rotulo).mkdir(parents=True, exist_ok=True)

    linhas, contagem, descartes = [], Counter(), Counter()
    for caminho, pessoa, fonte, rotulo in tqdm(itens, desc="Recortando"):
        saida = DATA / rotulo / f"{pessoa}__{fonte}_{caminho.stem}.jpg"
        if saida.exists() and not args.refazer:
            motivo = "ok"
        else:
            img = cv2.imread(str(caminho))
            if img is None:
                motivo = "ilegivel"
            else:
                recorte, info = recortar_rosto(img)
                motivo = info["motivo"]
                if recorte is not None:
                    cv2.imwrite(str(saida), recorte,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        if motivo == "ok":
            contagem[(pessoa, rotulo)] += 1
        else:
            descartes[(fonte, motivo)] += 1
        linhas.append({"origem": str(caminho.relative_to(RAIZ)),
                       "pessoa": pessoa, "fonte": fonte, "rotulo": rotulo,
                       "resultado": motivo,
                       "arquivo": saida.name if motivo == "ok" else ""})

    with open(DATA / "manifesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    pessoas = sorted({p for p, _ in contagem})
    texto = ["RESUMO DE data/ (rostos recortados)", "=" * 56,
             f"{'pessoa':<8}{'engajado':>10}{'nao_engajado':>14}{'total':>8}"]
    for p in pessoas:
        e, n = contagem[(p, "engajado")], contagem[(p, "nao_engajado")]
        texto.append(f"{p:<8}{e:>10}{n:>14}{e + n:>8}")
    tot_e = sum(v for (p, r), v in contagem.items() if r == "engajado")
    tot_n = sum(v for (p, r), v in contagem.items() if r == "nao_engajado")
    texto.append(f"{'TOTAL':<8}{tot_e:>10}{tot_n:>14}{tot_e + tot_n:>8}")
    texto.append("")
    texto.append("Descartadas (fonte, motivo): " +
                 (", ".join(f"{k}={v}" for k, v in sorted(descartes.items()))
                  or "nenhuma"))
    texto.append("")
    texto.append("Dividir SEMPRE por pessoa: grupo = nome.split('__')[0]")
    (DATA / "resumo.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
