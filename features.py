"""
Extração de características faciais para o classificador de engajamento.

Este módulo é usado TANTO pelo treinamento (treinar.py) QUANTO pela demo
ao vivo (paginas/classificar.py), para garantir que os dois usem exatamente
a mesma forma de transformar um rosto em números.

A ideia: pegar os 68 pontos do rosto (dlib) e deixá-los "neutros" em relação
a posição e tamanho, para que não importe onde o rosto está na imagem nem a
que distância da câmera ele está.
"""

from pathlib import Path

import cv2
import dlib
import numpy as np

# Raiz do projeto (esta pasta) e caminho do modelo de landmarks do dlib.
RAIZ = Path(__file__).resolve().parent
CAMINHO_PREDICTOR = RAIZ / "shape_predictor_68_face_landmarks.dat"

# Índices dos pontos do dlib usados como referência.
PONTA_NARIZ = 30
CANTO_OLHO_ESQ = 36   # canto externo do olho esquerdo
CANTO_OLHO_DIR = 45   # canto externo do olho direito


CAMINHO_MODELO_RESERVA = RAIZ / "fine tuning" / "modelos" / "blaze_face_short_range.tflite"


class DetectorComReserva:
    """
    Detector de rosto do dlib com RESERVA do MediaPipe.

    O detector HOG do dlib so reconhece rosto frontal em angulo "normal":
    em videos filmados de baixo para cima (camera abaixo do rosto) ele quase
    nunca acha nada. Esta classe tenta o dlib primeiro (mesmo resultado de
    sempre quando ele acha) e, SO se ele nao achar nada, tenta o MediaPipe e
    converte a caixa para o formato do dlib. O preditor de 68 pontos funciona
    normalmente com essa caixa.

    Se o MediaPipe ou o arquivo do modelo nao estiverem disponiveis, vira
    apenas o detector do dlib. Usa-se como o detector original: detector(gray).
    """

    def __init__(self):
        self._dlib = dlib.get_frontal_face_detector()
        self._mp = None
        self._indisponivel = False
        self.usos_reserva = 0   # quantas vezes o MediaPipe resgatou um rosto

    def _obter_mediapipe(self):
        if self._mp is not None or self._indisponivel:
            return self._mp
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            opcoes = vision.FaceDetectorOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(CAMINHO_MODELO_RESERVA)),
                min_detection_confidence=0.5)
            self._mp = (mp, vision.FaceDetector.create_from_options(opcoes))
        except Exception:
            self._indisponivel = True
        return self._mp

    def __call__(self, gray, *args):
        faces = self._dlib(gray, *args)
        if len(faces) > 0:
            return faces
        reserva = self._obter_mediapipe()
        if reserva is None:
            return faces
        mp, detector = reserva
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        retangulos = dlib.rectangles()
        for d in res.detections:
            b = d.bounding_box
            retangulos.append(dlib.rectangle(
                int(b.origin_x), int(b.origin_y),
                int(b.origin_x + b.width), int(b.origin_y + b.height)))
        if len(retangulos) > 0:
            self.usos_reserva += 1
        return retangulos


def criar_detector_predictor(com_reserva=False):
    """
    Cria o detector de rosto e o preditor de 68 pontos do dlib.

    com_reserva=False (padrao): detector do dlib puro, como sempre foi (usado
    pela demo e pelo treino). com_reserva=True: tenta o MediaPipe quando o
    dlib nao acha rosto (ver DetectorComReserva).
    """
    detector = DetectorComReserva() if com_reserva else dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(CAMINHO_PREDICTOR))
    return detector, predictor


def _pontos_do_rosto(imagem, detector, predictor):
    """Retorna (68 pontos em pixels, retângulo do rosto) ou (None, None)."""
    if imagem.ndim == 3:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagem
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    shape = predictor(gray, face)
    pontos = np.array([[p.x, p.y] for p in shape.parts()], dtype=float)
    return pontos, face


def normalizar_pontos(pontos):
    """
    Deixa os 68 pontos invariantes a posição e tamanho:

    1) POSIÇÃO: move a ponta do nariz (ponto 30) para a origem (0, 0).
    2) TAMANHO: divide tudo pela distância entre os cantos externos dos
       olhos (pontos 36 e 45), que é uma medida estável do rosto.

    Assim, o mesmo rosto gera o mesmo vetor esteja ele no canto da imagem
    ou no centro, perto ou longe da câmera.
    """
    pontos = pontos - pontos[PONTA_NARIZ]
    dist_olhos = np.linalg.norm(pontos[CANTO_OLHO_ESQ] - pontos[CANTO_OLHO_DIR])
    if dist_olhos < 1e-6:
        return None
    return pontos / dist_olhos


def extrair_features(imagem, detector, predictor):
    """
    Recebe uma imagem (BGR do OpenCV) e devolve:
      - vetor de 136 números (68 pontos x,y normalizados), ou None se não achar rosto
      - o retângulo do rosto detectado (útil para desenhar na tela), ou None
    """
    pontos, face = _pontos_do_rosto(imagem, detector, predictor)
    if pontos is None:
        return None, None
    pontos_norm = normalizar_pontos(pontos)
    if pontos_norm is None:
        return None, None
    return pontos_norm.flatten(), face


def ear_dos_pontos(pontos):
    """
    Calcula o EAR (Eye Aspect Ratio) médio dos dois olhos a partir dos 68
    pontos já detectados (em pixels). Retorna None se não houver pontos.

    EAR é a razão entre a "altura" e a "largura" do olho. Fica alto com o
    olho aberto e cai perto de zero quando o olho fecha. É a forma clássica
    de detectar olho fechado / piscada.
    """
    if pontos is None:
        return None

    def _ear(p):
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
        return vertical / horizontal if horizontal > 1e-6 else 0.0

    olho_esq = pontos[[36, 37, 38, 39, 40, 41]]
    olho_dir = pontos[[42, 43, 44, 45, 46, 47]]
    return (_ear(olho_esq) + _ear(olho_dir)) / 2.0


def eye_aspect_ratio(imagem, detector, predictor):
    """Compatibilidade: detecta o rosto e devolve o EAR médio (ou None)."""
    pontos, _ = _pontos_do_rosto(imagem, detector, predictor)
    return ear_dos_pontos(pontos)


def analisar_rosto(imagem, detector, predictor):
    """
    Faz UMA única detecção de rosto e devolve tudo o que a demo precisa:

      features : vetor de 136 números normalizados, ou None
      face     : retângulo do rosto (dlib), ou None
      ear      : EAR médio dos olhos, ou None

    Usar isto (em vez de chamar extrair_features + eye_aspect_ratio) evita
    detectar o rosto duas vezes por frame e garante que as features e o EAR
    venham exatamente dos mesmos pontos.
    """
    pontos, face = _pontos_do_rosto(imagem, detector, predictor)
    if pontos is None:
        return None, None, None
    ear = ear_dos_pontos(pontos)
    pontos_norm = normalizar_pontos(pontos)
    features = None if pontos_norm is None else pontos_norm.flatten()
    return features, face, ear


# ---------------------------------------------------------------------------
# Características geométricas derivadas (ETAPA 2 do plano de melhoria)
# ---------------------------------------------------------------------------

NOMES_GEOMETRICAS = [
    "ear_esq", "ear_dir", "ear_medio", "ear_diferenca",
    "mar_interno", "boca_largura", "boca_altura", "boca_curvatura",
    "sobr_olho_esq", "sobr_olho_dir", "sobr_olho_medio",
    "sobr_inclin_esq", "sobr_inclin_dir", "sobr_distancia_interna",
    "yaw", "pitch", "roll",
    "rosto_razao_largura_altura", "queixo_nariz", "nariz_linha_olhos",
]


def features_geometricas(pontos_norm):
    """
    Recebe os 68 pontos JA NORMALIZADOS (68x2, ver normalizar_pontos) e devolve
    um vetor com poucas medidas interpretaveis (nomes em NOMES_GEOMETRICAS):

      - EAR de cada olho: abertura do olho (cai com olho fechado/sonolencia)
      - MAR e medidas da boca: abertura (bocejo), largura e curvatura
        (sorriso x boca caida)
      - Sobrancelha: distancia ate o olho (levantada x franzida), inclinacao
      - yaw / pitch / roll: PROXIES 2D da direcao da cabeca (esquerda-direita,
        cima-baixo, inclinacao lateral). Nao usam solvePnP porque este
        precisaria dos parametros da camera de cada dataset.

    Sao razoes entre distancias dentro do mesmo rosto, entao nao dependem da
    posicao nem do tamanho do rosto (e mudam menos de pessoa para pessoa que
    as coordenadas brutas). Retorna None se receber None.
    """
    if pontos_norm is None:
        return None
    p = np.asarray(pontos_norm, dtype=float)
    eps = 1e-6

    def d(a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    def ear(i):
        return (d(i + 1, i + 5) + d(i + 2, i + 4)) / (2.0 * d(i, i + 3) + eps)

    ear_e, ear_d = ear(36), ear(42)

    mar = (d(61, 67) + d(62, 66) + d(63, 65)) / (2.0 * d(60, 64) + eps)
    largura_boca = d(48, 54)
    altura_boca = d(51, 57)
    # y cresce para baixo: sorriso = cantos acima do centro => valor negativo
    curvatura = float((p[48][1] + p[54][1]) / 2.0 - (p[51][1] + p[57][1]) / 2.0)

    centro_olho_e = p[36:42].mean(axis=0)
    centro_olho_d = p[42:48].mean(axis=0)
    sobr_e = float(centro_olho_e[1] - p[17:22, 1].mean())
    sobr_d = float(centro_olho_d[1] - p[22:27, 1].mean())
    inclin_e = float(p[17][1] - p[21][1])    # ponta externa vs interna
    inclin_d = float(p[26][1] - p[22][1])
    dist_sobr = d(21, 22)

    yaw = float(np.log((abs(p[30][0] - p[0][0]) + eps) /
                       (abs(p[16][0] - p[30][0]) + eps)))
    meio_olhos_y = (p[36][1] + p[45][1]) / 2.0
    pitch = float((p[30][1] - meio_olhos_y) / (abs(p[8][1] - p[30][1]) + eps))
    roll = float(np.arctan2(p[45][1] - p[36][1], p[45][0] - p[36][0]))

    razao_rosto = d(0, 16) / (abs(p[8][1] - p[19:25, 1].mean()) + eps)
    queixo_nariz = float(p[8][1] - p[30][1])
    nariz_olhos = float(p[30][1] - meio_olhos_y)

    return np.array([
        ear_e, ear_d, (ear_e + ear_d) / 2.0, abs(ear_e - ear_d),
        mar, largura_boca, altura_boca, curvatura,
        sobr_e, sobr_d, (sobr_e + sobr_d) / 2.0,
        inclin_e, inclin_d, dist_sobr,
        yaw, pitch, roll,
        razao_rosto, queixo_nariz, nariz_olhos,
    ])


def geometricas_do_vetor(vetor_136):
    """Atalho: 136 numeros normalizados (como no cache) -> features geometricas."""
    return features_geometricas(np.asarray(vetor_136, dtype=float).reshape(68, 2))
