import time
from collections import deque
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd

from features import RAIZ, criar_detector_predictor, extrair_features, eye_aspect_ratio

CAMINHO_MODELO = RAIZ / "model_Random_Forest.pkl"

# --- Parâmetros da janela temporal ---
FPS_ALVO = 2                 # frames processados por segundo
JANELA_SEG = 20              # duração da janela deslizante (segundos)
JANELA_MAX = FPS_ALVO * JANELA_SEG
MIN_AMOSTRAS = 5             # mínimo de amostras antes de exibir um resultado
LARGURA_MIN_ROSTO = 100      # rostos menores que isso são ignorados (baixa qualidade)
ALPHA_EMA = 0.15             # suavização exponencial (0 = lento, 1 = sem suavizar)

# --- Verificação de olhos fechados ---
EAR_LIMIAR = 0.15            # abaixo disso o olho é considerado fechado
EAR_AMOSTRAS = 4            # amostras seguidas de olho fechado para valer (~2 s)

# Limiar de decisão: engajado se P(engajado) >= isto
LIMIAR_ENGAJADO = 0.5


@st.cache_resource
def carregar_recursos():
    detector, predictor = criar_detector_predictor()
    modelo = joblib.load(CAMINHO_MODELO)
    return detector, predictor, modelo


def prob_engajado(proba_por_classe):
    """Junta os 4 níveis em 2: engajado = níveis 2 e 3."""
    return float(proba_por_classe[2] + proba_por_classe[3])


def montar_relatorio(historico):
    """Recebe a lista de amostras da sessão e devolve (DataFrame, resumo dict)."""
    df = pd.DataFrame(historico)
    if df.empty:
        return df, {}
    duracao = df["segundos"].iloc[-1] - df["segundos"].iloc[0]
    resumo = {
        "Início": df["horario"].iloc[0],
        "Fim": df["horario"].iloc[-1],
        "Duração (s)": round(duracao, 1),
        "Amostras": len(df),
        "% do tempo engajado": f"{(df['engajado'].mean() * 100):.0f}%",
        "P(engajado) média": f"{df['prob_engajado'].mean():.2f}",
        "P(engajado) mínima": f"{df['prob_engajado'].min():.2f}",
        "P(engajado) máxima": f"{df['prob_engajado'].max():.2f}",
    }
    return df, resumo


def exibir():
    st.header("Classificação em Tempo Real")
    st.write(
        "O engajamento é um estado que se sustenta no tempo, não um instante. "
        f"Por isso a decisão vem de uma janela de {JANELA_SEG} segundos: cada frame "
        "contribui com uma probabilidade e o que aparece na tela é a média suavizada "
        "dessa janela. Ao final, é possível salvar o relatório da sessão."
    )

    detector, predictor, modelo = carregar_recursos()
    classes_modelo = [int(c) for c in modelo.classes_]

    # Histórico da sessão persiste entre os reruns do Streamlit
    st.session_state.setdefault("historico", [])

    iniciar = st.checkbox("Ativar webcam")

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        resultado_placeholder = st.empty()
        barra_placeholder = st.empty()
        status_placeholder = st.empty()
    grafico_placeholder = st.empty()

    if iniciar:
        # Nova captura: zera o histórico ao (re)ativar a webcam
        if not st.session_state.get("_capturando", False):
            st.session_state["historico"] = []
            st.session_state["_capturando"] = True

        janela_proba = deque(maxlen=JANELA_MAX)
        historico_score = deque(maxlen=JANELA_MAX)
        ema_prob = None
        ear_baixo_seguidas = 0
        t_inicio = time.time()

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

                olhos_fechados = False
                if rosto_valido:
                    ear = eye_aspect_ratio(frame, detector, predictor)
                    if ear is not None and ear < EAR_LIMIAR:
                        ear_baixo_seguidas += 1
                    else:
                        ear_baixo_seguidas = 0
                    olhos_fechados = ear_baixo_seguidas >= EAR_AMOSTRAS

                    proba_raw = modelo.predict_proba([features])[0]
                    proba = np.zeros(4)
                    for i, c in enumerate(classes_modelo):
                        proba[c] = proba_raw[i]
                    janela_proba.append(proba)

                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    ear = None

                if len(janela_proba) >= MIN_AMOSTRAS:
                    media = np.mean(janela_proba, axis=0)
                    p_eng = prob_engajado(media)
                    ema_prob = p_eng if ema_prob is None else \
                        ALPHA_EMA * p_eng + (1 - ALPHA_EMA) * ema_prob

                    if olhos_fechados:
                        engajado = False
                        rotulo = "🔴 Não engajado (olhos fechados)"
                    elif ema_prob >= LIMIAR_ENGAJADO:
                        engajado = True
                        rotulo = "🟢 Engajado"
                    else:
                        engajado = False
                        rotulo = "🟠 Não engajado"

                    resultado_placeholder.markdown(f"### {rotulo}")
                    barra_placeholder.progress(
                        float(ema_prob),
                        text=f"P(engajado) = {ema_prob:.0%}",
                    )
                    txt_status = f"{len(janela_proba)}/{JANELA_MAX} amostras"
                    if ear is not None:
                        txt_status += f" · EAR={ear:.2f}"
                    status_placeholder.caption(txt_status)
                    historico_score.append(ema_prob)

                    # Registra a amostra no relatório da sessão
                    agora = time.time()
                    st.session_state["historico"].append({
                        "horario": datetime.now().strftime("%H:%M:%S"),
                        "segundos": round(agora - t_inicio, 1),
                        "prob_engajado": round(float(ema_prob), 3),
                        "engajado": bool(engajado),
                        "olhos_fechados": bool(olhos_fechados),
                        "ear": round(float(ear), 3) if ear is not None else None,
                    })
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
                        pd.DataFrame({"P(engajado)": list(historico_score)})
                    )

                espera = (1.0 / FPS_ALVO) - (time.time() - inicio)
                if espera > 0:
                    time.sleep(espera)
        finally:
            cap.release()
    else:
        st.session_state["_capturando"] = False

    # --- Relatório da sessão (aparece depois que a webcam é desligada) ---
    historico = st.session_state.get("historico", [])
    if historico and not iniciar:
        st.divider()
        st.subheader("Relatório da sessão")
        df, resumo = montar_relatorio(historico)

        st.table(pd.DataFrame(resumo.items(), columns=["Métrica", "Valor"]))
        st.line_chart(df.set_index("segundos")[["prob_engajado"]])

        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Baixar relatório (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"relatorio_engajamento_{carimbo}.csv",
            mime="text/csv",
        )
        resumo_txt = "\n".join(f"{k}: {v}" for k, v in resumo.items())
        st.download_button(
            "⬇️ Baixar resumo (TXT)",
            data=resumo_txt.encode("utf-8"),
            file_name=f"resumo_engajamento_{carimbo}.txt",
            mime="text/plain",
        )
        if st.button("Limpar relatório"):
            st.session_state["historico"] = []
            st.rerun()
