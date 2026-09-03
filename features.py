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


def criar_detector_predictor():
    """Cria o detector de rosto e o preditor de 68 pontos do dlib."""
    detector = dlib.get_frontal_face_detector()
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


def eye_aspect_ratio(imagem, detector, predictor):
    """
    Retorna o EAR (Eye Aspect Ratio) médio dos dois olhos, ou None.

    EAR é a razão entre a "altura" e a "largura" do olho. Fica alto com o
    olho aberto e cai perto de zero quando o olho fecha. É a forma clássica
    de detectar olho fechado / piscada.
    """
    pontos, _ = _pontos_do_rosto(imagem, detector, predictor)
    if pontos is None:
        return None

    def _ear(p):
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
        return vertical / horizontal if horizontal > 1e-6 else 0.0

    olho_esq = pontos[[36, 37, 38, 39, 40, 41]]
    olho_dir = pontos[[42, 43, 44, 45, 46, 47]]
    return (_ear(olho_esq) + _ear(olho_dir)) / 2.0
