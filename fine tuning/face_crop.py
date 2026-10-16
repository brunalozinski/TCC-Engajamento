"""
Recorte de rosto compartilhado pelo pipeline de fine-tuning.

Usado por:
  - montar_data_finetuning.py  (gera data/ a partir dos datasets existentes)
  - extrair_frames_rosto.py    (extrai frames de videos novos)
  - o app, se for trocado o Haar Cascade por esta funcao

Por que existe: treino e uso precisam recortar o rosto EXATAMENTE do mesmo
jeito (mesmo detector, mesma margem, mesmo tamanho). Se o treino usar um
recorte e a demo outro, o modelo se comporta de forma diferente no uso real.

Decisoes:
  - Detector principal: dlib HOG (o mesmo do restante do projeto).
  - Detector de RESERVA: MediaPipe (BlazeFace). O dlib so reconhece rosto
    frontal em angulo "normal"; em videos filmados de baixo para cima (camera
    abaixo do rosto, comum em notebook/celular apoiado) ele falha quase sempre
    (pessoa6: 17 de 287 frames). O MediaPipe so e usado quando o dlib nao acha
    nada, entao os recortes ja gerados com o dlib nao mudam. Calibracao: nas
    imagens em que os dois acham rosto, o tamanho da caixa e praticamente o
    mesmo (razao mediana 1,02; centro desloca ~5% do rosto), entao as caixas
    sao tratadas como equivalentes.
  - Recorte QUADRADO (o processador do Swin/ViT redimensiona para 224x224; um
    recorte retangular ficaria esticado).
  - Margem de 20% de cada lado, igual ao app original (FACE_MARGIN = 0.2).
  - A imagem e reduzida (lado maior = 1280) antes de detectar: frames 4K
    ficam ~10x mais rapidos e o resultado e o mesmo.
"""

from pathlib import Path

import cv2
import dlib

LADO_MAX_DETECCAO = 1280   # reduz imagens maiores que isso antes de detectar
MARGEM = 0.2               # 20% do tamanho do rosto, de cada lado
TAMANHO_SAIDA = 256        # lado do recorte salvo (o modelo reduz para 224)
ROSTO_MIN = 60             # rosto menor que isso (pixels) e descartado

MODELO_MEDIAPIPE = (Path(__file__).resolve().parent / "modelos"
                    / "blaze_face_short_range.tflite")

_detector = None
_detector_mp = None
_mp_indisponivel = False


def obter_detector():
    """Cria o detector de rosto do dlib uma unica vez."""
    global _detector
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
    return _detector


def _obter_detector_mediapipe():
    """Cria o detector MediaPipe sob demanda; None se nao estiver disponivel."""
    global _detector_mp, _mp_indisponivel
    if _detector_mp is not None or _mp_indisponivel:
        return _detector_mp
    try:
        import mediapipe as mp  # import tardio: so quando o dlib falha
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        opcoes = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(MODELO_MEDIAPIPE)),
            min_detection_confidence=0.5)
        _detector_mp = (mp, vision.FaceDetector.create_from_options(opcoes))
    except Exception:
        _mp_indisponivel = True   # sem mediapipe ou sem o arquivo do modelo
    return _detector_mp


def reduzir(imagem_bgr, lado_max=LADO_MAX_DETECCAO):
    """Reduz a imagem se o lado maior passar de lado_max (mantem proporcao)."""
    maior = max(imagem_bgr.shape[:2])
    if maior <= lado_max:
        return imagem_bgr
    escala = lado_max / maior
    return cv2.resize(imagem_bgr, None, fx=escala, fy=escala,
                      interpolation=cv2.INTER_AREA)


def _detectar_caixas(img):
    """Devolve (lista de (cx, cy, lado), nome do detector usado)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    rostos = obter_detector()(gray)
    if len(rostos) > 0:
        return ([((r.left() + r.right()) / 2.0, (r.top() + r.bottom()) / 2.0,
                  max(r.width(), r.height())) for r in rostos], "dlib")

    mp_det = _obter_detector_mediapipe()
    if mp_det is not None:
        mp, detector = mp_det
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=rgb))
        caixas = [((b.origin_x + b.width / 2.0), (b.origin_y + b.height / 2.0),
                   max(b.width, b.height))
                  for b in (d.bounding_box for d in res.detections)]
        if caixas:
            return caixas, "mediapipe"
    return [], "nenhum"


def recortar_rosto(imagem_bgr, margem=MARGEM, tamanho=TAMANHO_SAIDA,
                   rosto_min=ROSTO_MIN):
    """
    Detecta o MAIOR rosto da imagem e devolve (recorte, info).

    recorte : imagem quadrada tamanho x tamanho (BGR), ou None se nao houver
              rosto valido.
    info    : dict com 'motivo' ('ok', 'sem_rosto', 'rosto_pequeno'),
              'n_rostos', 'detector' ('dlib', 'mediapipe' ou 'nenhum') e
              'caixa' (x0, y0, lado) na imagem reduzida.
    """
    img = reduzir(imagem_bgr)
    caixas, quem = _detectar_caixas(img)
    if not caixas:
        return None, {"motivo": "sem_rosto", "n_rostos": 0,
                      "detector": quem, "caixa": None}

    cx, cy, lado_rosto = max(caixas, key=lambda c: c[2])
    if lado_rosto < rosto_min:
        return None, {"motivo": "rosto_pequeno", "n_rostos": len(caixas),
                      "detector": quem, "caixa": None}

    h, w = img.shape[:2]
    lado = int(round(lado_rosto * (1 + 2 * margem)))
    lado = min(lado, h, w)                      # nunca maior que a imagem
    x0 = int(round(cx - lado / 2.0))
    y0 = int(round(cy - lado / 2.0))
    x0 = min(max(x0, 0), w - lado)              # desliza para caber na imagem
    y0 = min(max(y0, 0), h - lado)

    recorte = img[y0:y0 + lado, x0:x0 + lado]
    recorte = cv2.resize(recorte, (tamanho, tamanho),
                         interpolation=cv2.INTER_AREA)
    return recorte, {"motivo": "ok", "n_rostos": len(caixas),
                     "detector": quem, "caixa": (x0, y0, lado)}
