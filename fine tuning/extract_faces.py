"""
Extrai frames de um video a uma taxa fixa (padrao: 5 fps), detecta se ha um
rosto no frame e, caso haja, salva o frame inteiro (sem recortar) em uma
pasta de destino. Isso preserva o contexto/detalhes do frame completo para
o treinamento, usando a deteccao apenas para descartar frames sem rosto.

Uso:
    python extract_faces.py --video caminho/video.mp4 --label engajado --out data

Isso vai salvar os frames em:
    data/engajado/video_000001.jpg
    data/engajado/video_000002.jpg
    ...
"""

import argparse
import os

import cv2


def extract_faces(video_path, label, out_dir, fps_target=5, min_size=60):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
    frame_interval = max(round(video_fps / fps_target), 1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
            )

            if len(faces) > 0:
                saved_idx += 1
                out_path = os.path.join(
                    label_dir, f"{video_name}_{saved_idx:05d}.jpg"
                )
                cv2.imwrite(out_path, frame)
            else:
                no_face_count += 1

        frame_idx += 1

    cap.release()
    print(f"[{video_name}] frames lidos: {frame_idx}, frames salvos: {saved_idx}, "
          f"frames sem rosto detectado: {no_face_count}")
    print(f"Salvo em: {label_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extrai frames com rosto de um video para dataset.")
    parser.add_argument("--video", required=True, help="Caminho do arquivo de video")
    parser.add_argument("--label", required=True, help="Nome da classe (ex: engajado, nao_engajado)")
    parser.add_argument("--out", default="data", help="Pasta raiz de saida (default: data)")
    parser.add_argument("--fps", type=float, default=5, help="Frames por segundo a extrair (default: 5)")
    parser.add_argument("--min-size", type=int, default=60, help="Tamanho minimo do rosto em pixels (default: 60)")
    args = parser.parse_args()

    extract_faces(
        video_path=args.video,
        label=args.label,
        out_dir=args.out,
        fps_target=args.fps,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
