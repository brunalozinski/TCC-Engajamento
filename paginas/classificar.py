import time
from collections import deque, Counter
from pathlib import Path

import streamlit as st
import cv2
import dlib
import numpy as np
import joblib
import pandas as pd

# Raiz do projeto (…/paginas/classificar.py -> …/)
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PREDICTOR = RAIZ / "shape_predictor_68_face_landmarks.dat"
CAMINHO_MODELO = RAIZ / "model_Random_Forest.pkl"

# --- Parâmetros da janela temporal ---
FPS_ALVO = 2                 # frames processados por segundo
JANELA_SEG = 20              # duração da janela deslizante (segundos)
JANELA_MAX = FPS_ALVO * JANELA_SEG
MIN_AMOSTRAS = 5             # mínimo de amostras válidas antes de exibir resultado
LARGURA_MIN_ROSTO = 100      # rostos menores que isso são ignorados (baixa qualidade)
ALPHA_EMA = 0.1             # suavização exponencial do score contínuo
CONFIRMACOES_TROCA = 3      # amostras seguidas numa nova classe para trocar o rótulo


@st.cache_resource
def carregar_recursos():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(CAMINHO_PREDICTOR))
    modelo = joblib.load(CAMINHO_MODELO)
    return detector, predictor, modelo


def extrair_features(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    shape = predictor(gray, face)
    coords = np.array([[p.x, p.y] for p in shape.parts()], dtype=float)
    coords /= face.width()
    return coords.flatten(), face


def exibir():
    st.header("Classificação em Tempo Real")
    st.write(
        "O engajamento é um estado comportamental sustentado, não um instante. "
        f"Por isso a classificação é feita a partir de uma janela deslizante de "
        f"{JANELA_SEG} segundos: cada frame contribui com uma probabilidade e o "
        "resultado exibido é a média suavizada dessa janela."
    )

    rotulos = {
        0: ("Muito Baixo", "🔴"),
        1: ("Baixo", "🟠"),
        2: ("Alto", "🟢"),
        3: ("Muito Alto", "🔵"),
    }

    detector, predictor, modelo = carregar_recursos()
    classes_modelo = list(modelo.classes_)

    iniciar = st.checkbox("Ativar webcam")

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        resultado_placeholder = st.empty()
        confianca_placeholder = st.empty()
        status_placeholder = st.empty()
    grafico_placeholder = st.empty()

    if not iniciar:
        return

    # Estado da janela temporal
    janela_proba = deque(maxlen=JANELA_MAX)   # probabilidades [p0..p3] por amostra
    historico_score = deque(maxlen=JANELA_MAX)  # score contínuo suavizado ao longo do tempo
    ema_score = None
    rotulo_atual = None
    contador_troca = 0

    cap = cv2.VideoCapture(0)
    try:
        while iniciar:
            inicio = time.time()
            ret, frame = cap.read()
            if not ret:
                st.error("Não foi possível acessar a webcam.")
                break

            features, face = extrair_features(frame, detector, predictor)
            rosto_valido = features is not None and face.width() >= LARGURA_MIN_ROSTO

            if rosto_valido:
                proba_raw = modelo.predict_proba([features])[0]
                # Reordena para o índice de classe 0..3 independentemente da ordem interna
                proba = np.zeros(4)
                for i, c in enumerate(classes_modelo):
                    proba[int(c)] = proba_raw[i]
                janela_proba.append(proba)

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Agrega a janela
            if len(janela_proba) >= MIN_AMOSTRAS:
                media = np.mean(janela_proba, axis=0)
                classe_janela = int(np.argmax(media))
                score = float(np.dot(media, [0, 1, 2, 3]))
                ema_score = score if ema_score is None else \
                    ALPHA_EMA * score + (1 - ALPHA_EMA) * ema_score
                historico_score.append(ema_score)

                # Histerese: só troca o rótulo após N amostras seguidas na nova classe
                if rotulo_atual is None:
                    rotulo_atual = classe_janela
                elif classe_janela != rotulo_atual:
                    contador_troca += 1
                    if contador_troca >= CONFIRMACOES_TROCA:
                        rotulo_atual = classe_janela
                        contador_troca = 0
                else:
                    contador_troca = 0

                nome, emoji = rotulos[rotulo_atual]
                confianca = media[rotulo_atual]
                resultado_placeholder.markdown(f"### {emoji} {nome}")
                confianca_placeholder.progress(
                    float(confianca),
                    text=f"Confiança na janela: {confianca:.0%} · Score: {ema_score:.2f}/3",
                )
                status_placeholder.caption(
                    f"{len(janela_proba)}/{JANELA_MAX} amostras na janela"
                )

                if face is not None and rotulo_atual is not None:
                    cv2.putText(
                        frame, nome, (face.left(), max(face.top() - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )
            else:
                resultado_placeholder.markdown("### ⏳ Coletando amostras…")
                status_placeholder.caption(
                    f"{len(janela_proba)}/{MIN_AMOSTRAS} amostras mínimas"
                )

            if not rosto_valido:
                status_placeholder.caption("⚠️ Nenhum rosto frontal válido neste frame")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if len(historico_score) >= 2:
                grafico_placeholder.line_chart(
                    pd.DataFrame({"Engajamento (0–3)": list(historico_score)})
                )

            # Throttle para ~FPS_ALVO
            espera = (1.0 / FPS_ALVO) - (time.time() - inicio)
            if espera > 0:
                time.sleep(espera)
    finally:
        cap.release()
