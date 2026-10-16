"""
Extrai frames de um video para montar o conjunto de teste EXTERNO (pessoas
novas, nunca vistas no ER_Dataset original), usado para avaliar generalizacao
de verdade -- ver doc/relatorio_experimentos_vazamento_dados.md.

Uso:
    python extrair_frames_video.py <video> <pessoa> <sessao> <classe> [--intervalo N]

Exemplo:
    python extrair_frames_video.py video_joana_focada.mp4 joana 01 3 --intervalo 10

Isso le "video_joana_focada.mp4", guarda 1 a cada 10 frames onde um rosto foi
detectado, e salva em:
    ER_Dataset_Externo/3/joana_sessao01_classe3_0001.jpg
    ER_Dataset_Externo/3/joana_sessao01_classe3_0002.jpg
    ...

O nome do arquivo guarda pessoa e sessao EXPLICITAMENTE (diferente do
ER_Dataset original, onde isso teve que ser adivinhado pelo numero do frame).
Use esses campos para garantir, na hora de treinar/avaliar, que todos os
frames da mesma pessoa fiquem sempre do mesmo lado (treino OU teste).
"""

import argparse

import cv2

from features import RAIZ, criar_detector_predictor, extrair_features

PASTA_EXTERNA = RAIZ / "ER_Dataset_Externo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Caminho do arquivo de video (ex: video.mp4)")
    parser.add_argument("pessoa", help="Identificador da pessoa (ex: joana, pessoa01)")
    parser.add_argument("sessao", help="Identificador da sessao/gravacao (ex: 01)")
    parser.add_argument("classe", type=int, choices=[0, 1, 2, 3],
                         help="Nivel de engajamento: 0=Muito Baixo, 1=Baixo, 2=Alto, 3=Muito Alto")
    parser.add_argument("--intervalo", type=int, default=10,
                         help="Salva 1 a cada N frames do video (default: 10)")
    parser.add_argument("--exigir-rosto", action="store_true", default=True,
                         help="So salva frames onde um rosto foi detectado (default: ligado)")
    args = parser.parse_args()

    pasta_destino = PASTA_EXTERNA / str(args.classe)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    print("Carregando detector de rosto...")
    detector, predictor = criar_detector_predictor()

    print(f"Abrindo video: {args.video}")
    captura = cv2.VideoCapture(args.video)
    if not captura.isOpened():
        print("Nao foi possivel abrir o video. Verifique o caminho.")
        return

    numero_do_frame = 0
    frames_salvos = 0
    frames_sem_rosto = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        if numero_do_frame % args.intervalo == 0:
            if args.exigir_rosto:
                features, _ = extrair_features(frame, detector, predictor)
                if features is None:
                    frames_sem_rosto += 1
                    numero_do_frame += 1
                    continue

            frames_salvos += 1
            nome_arquivo = (
                f"{args.pessoa}_sessao{args.sessao}_classe{args.classe}_"
                f"{frames_salvos:04d}.jpg"
            )
            cv2.imwrite(str(pasta_destino / nome_arquivo), frame)

        numero_do_frame += 1

    captura.release()

    print(f"\nVideo tinha {numero_do_frame} frames no total.")
    print(f"Frames salvos: {frames_salvos}")
    if args.exigir_rosto:
        print(f"Frames descartados (sem rosto detectado): {frames_sem_rosto}")
    print(f"Salvos em: {pasta_destino}")


if __name__ == "__main__":
    main()
