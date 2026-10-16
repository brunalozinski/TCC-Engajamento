"""
Extrai frames de um video JA RECORTADOS NO ROSTO, para o fine-tuning, e (opcional)
tambem salva o frame inteiro no formato do pipeline de landmarks.

Junta o melhor dos dois extratores antigos:
  - de extrair_frames_video.py : pessoa/sessao no nome do arquivo (para dividir
    o treino por pessoa) e deteccao com dlib;
  - de extract_faces.py        : amostragem POR TEMPO (frames por segundo), igual
    para videos de 30 e 60 fps, e rotulo binario.
  - novo: recorte do rosto (face_crop.py), o mesmo que o treino e a demo usam.

Saidas:
  fine tuning/data/<rotulo>/<pessoa>__<sessao>_<rotulo>[_n<nivel>]_<NNNN>.jpg
      -> recorte quadrado 256x256 do maior rosto (entrada do fine-tuning).
  ER_Dataset_Externo/<nivel>/<pessoa>_sessao<sessao>_classe<nivel>_<NNNN>.jpg
      -> SO se voce passar --nivel e --salvar-frame-inteiro: frame reduzido
         (lado maior 1280) no nome antigo, para os scripts de landmarks
         (avaliar_*.py) continuarem funcionando com os mesmos videos.

Uso:
    python extrair_frames_rosto.py VIDEO PESSOA SESSAO ROTULO [opcoes]

    ROTULO: engajado | nao_engajado

Exemplos:
    python extrair_frames_rosto.py joana_foco.mp4 pessoa4 01 engajado --nivel 3
    python extrair_frames_rosto.py joana_foco.mp4 pessoa4 01 engajado --nivel 3 --salvar-frame-inteiro
    python extrair_frames_rosto.py joana_sono.mp4 pessoa4 01 nao_engajado --nivel 0 --fps 2

Cuidados:
  - Use um identificador de pessoa SEM "__" e sem espacos (ex.: pessoa4).
  - Uma pessoa = um identificador em TODOS os videos dela (mesmo em dias
    diferentes). Assim ela fica inteira no treino ou inteira no teste.
  - Frames seguidos sao quase iguais; por isso o padrao e 3 fps e nao 30.
"""

import argparse
from pathlib import Path

import cv2

from face_crop import recortar_rosto, reduzir

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
DATA = AQUI / "data"
EXTERNO = RAIZ / "ER_Dataset_Externo"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("pessoa", help="ID da pessoa (ex.: pessoa4), sem '__'")
    ap.add_argument("sessao", help="ID da gravacao (ex.: 01)")
    ap.add_argument("rotulo", choices=["engajado", "nao_engajado"])
    ap.add_argument("--nivel", type=int, choices=[0, 1, 2, 3],
                    help="Nivel original 0-3 (guardado no nome; necessario "
                         "para --salvar-frame-inteiro)")
    ap.add_argument("--fps", type=float, default=3.0,
                    help="Frames por segundo a amostrar (padrao: 3)")
    ap.add_argument("--max-segundos", type=float, default=0,
                    help="Para apos N segundos de video (teste)")
    ap.add_argument("--saida", default=str(DATA),
                    help="Pasta de saida dos recortes (padrao: fine tuning/data)")
    ap.add_argument("--salvar-frame-inteiro", action="store_true",
                    help="Tambem salva o frame inteiro em ER_Dataset_Externo/")
    args = ap.parse_args()

    if "__" in args.pessoa or " " in args.pessoa:
        ap.error("PESSOA nao pode conter '__' nem espacos.")
    if args.salvar_frame_inteiro and args.nivel is None:
        ap.error("--salvar-frame-inteiro exige --nivel.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir o video: {args.video}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    intervalo = max(int(round(fps_video / args.fps)), 1)

    pasta_recortes = Path(args.saida) / args.rotulo
    pasta_recortes.mkdir(parents=True, exist_ok=True)
    pasta_inteiro = None
    if args.salvar_frame_inteiro:
        pasta_inteiro = EXTERNO / str(args.nivel)
        pasta_inteiro.mkdir(parents=True, exist_ok=True)

    sufixo_nivel = f"_n{args.nivel}" if args.nivel is not None else ""
    lidos = amostrados = salvos = 0
    descartes = {"sem_rosto": 0, "rosto_pequeno": 0}
    varios_rostos = 0
    via_reserva = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_segundos and lidos / fps_video >= args.max_segundos:
            break
        if lidos % intervalo == 0:
            amostrados += 1
            recorte, info = recortar_rosto(frame)
            if recorte is None:
                descartes[info["motivo"]] += 1
            else:
                salvos += 1
                varios_rostos += info["n_rostos"] > 1
                via_reserva += info["detector"] == "mediapipe"
                nome = (f"{args.pessoa}__{args.sessao}_{args.rotulo}"
                        f"{sufixo_nivel}_{salvos:04d}.jpg")
                cv2.imwrite(str(pasta_recortes / nome), recorte,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                if pasta_inteiro is not None:
                    nome_antigo = (f"{args.pessoa}_sessao{args.sessao}_"
                                   f"classe{args.nivel}_{salvos:04d}.jpg")
                    cv2.imwrite(str(pasta_inteiro / nome_antigo),
                                reduzir(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
        lidos += 1
    cap.release()

    print(f"Video: {fps_video:.1f} fps, {lidos} frames lidos "
          f"({lidos / fps_video:.0f} s); amostra a cada {intervalo} frames.")
    print(f"Amostrados: {amostrados} | salvos: {salvos} | "
          f"sem rosto: {descartes['sem_rosto']} | "
          f"rosto pequeno: {descartes['rosto_pequeno']}")
    if via_reserva:
        print(f"{via_reserva} frames detectados pelo MediaPipe (reserva; o dlib "
              "falhou, comum com camera abaixo do rosto).")
    if varios_rostos:
        print(f"ATENCAO: {varios_rostos} frames tinham mais de um rosto "
              "(foi usado o maior). Confira se e a pessoa certa.")
    print(f"Recortes em: {pasta_recortes}")
    if pasta_inteiro is not None:
        print(f"Frames inteiros em: {pasta_inteiro}")


if __name__ == "__main__":
    main()
