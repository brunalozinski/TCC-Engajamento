import time
from collections import deque
from datetime import datetime

import streamlit as st
import altair as alt
import cv2
import numpy as np
import joblib
import pandas as pd

from features import RAIZ, criar_detector_predictor, analisar_rosto

CAMINHO_MODELO = RAIZ / "model_Random_Forest.pkl"

# --- Parâmetros da janela temporal ---
FPS_ALVO = 2                 # frames processados por segundo
JANELA_SEG = 20              # duração de cada janela (segundos)
MIN_AMOSTRAS = 5             # mínimo de amostras para uma janela ser válida
LARGURA_MIN_ROSTO = 100      # rostos menores que isso são ignorados (baixa qualidade)
ALPHA_EMA = 0.15             # suavização exponencial do score parcial ao vivo

# --- Verificação de olhos fechados ---
# A ideia: NÃO usar um limiar fixo (olho pequeno ou olhar baixo dispararia
# falso positivo). Em vez disso, mede-se o EAR natural da pessoa nos primeiros
# segundos (olhos abertos) e só conta como "fechado" quando o EAR cai bem
# abaixo desse valor por MAIS de 2 segundos seguidos.
CALIBRACAO_SEG = 3.0        # tempo inicial para medir o EAR natural (olhos abertos)
EAR_FRAC_FECHADO = 0.60     # "fechado" = EAR abaixo de 60% do EAR natural
EAR_FALLBACK = 0.18        # limiar usado se a calibração não coletar amostras
FECHADO_SEG = 2.0          # tempo CONTÍNUO de olho fechado para confirmar

# --- Três categorias de engajamento (dois cortes de score) ---
# P(engajado) = P(nível 2) + P(nível 3)
LIMIAR_MEDIO = 0.40         # abaixo disso: Baixo
LIMIAR_ALTO = 0.65         # a partir disso: Alto; entre os dois: Moderado

CATEGORIAS = {
    "baixo": "🔴 Baixo engajamento",
    "moderado": "🟠 Engajamento moderado",
    "alto": "🟢 Alto engajamento",
}


@st.cache_resource
def carregar_recursos():
    detector, predictor = criar_detector_predictor()
    modelo = joblib.load(CAMINHO_MODELO)
    return detector, predictor, modelo


def prob_engajado(proba_por_classe):
    """Junta os 4 níveis em 1 score: engajado = níveis 2 e 3."""
    return float(proba_por_classe[2] + proba_por_classe[3])


def categorizar(p_eng, olhos_fechados):
    """Mapeia o score da janela para uma das 3 categorias."""
    if olhos_fechados:
        return "baixo"
    if p_eng < LIMIAR_MEDIO:
        return "baixo"
    if p_eng < LIMIAR_ALTO:
        return "moderado"
    return "alto"


def grafico_combo(df, x, y, x_titulo="Janela", y_titulo="P(engajado)"):
    """Um único gráfico com colunas + linha sobre os mesmos pontos."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:O", title=x_titulo, axis=alt.Axis(labelAngle=0))
    )
    barras = base.mark_bar(color="#9ecae1").encode(
        y=alt.Y(f"{y}:Q", title=y_titulo)
    )
    linha = base.mark_line(color="#08519c", point=True).encode(y=f"{y}:Q")
    return (barras + linha).properties(height=320)


def montar_relatorio(historico):
    """Recebe a lista de janelas concluídas e devolve (DataFrame, resumo dict)."""
    df = pd.DataFrame(historico)
    if df.empty:
        return df, {}
    contagem = df["categoria"].value_counts()
    total = len(df)
    resumo = {
        "Início": df["hora_inicio"].iloc[0],
        "Fim": df["hora_fim"].iloc[-1],
        "Janelas concluídas": total,
        "Duração total (s)": round(df["fim_seg"].iloc[-1] - df["inicio_seg"].iloc[0], 1),
        "P(engajado) média": f"{df['prob_engajado'].mean():.2f}",
        "Janelas com olhos fechados": int(df["olhos_fechados"].sum()),
    }
    for chave, rotulo in CATEGORIAS.items():
        n = int(contagem.get(chave, 0))
        resumo[rotulo] = f"{n} janela(s) · {n / total * 100:.0f}%"
    return df, resumo


def exibir():
    st.header("Classificação em Tempo Real")
    st.write(
        "O engajamento é um estado que se sustenta no tempo, não um instante. "
        f"Por isso a decisão é fechada a cada **janela de {JANELA_SEG} segundos**: "
        "durante a janela um cronômetro mostra quanto falta; quando ela termina, "
        "a média das probabilidades vira uma de três categorias "
        "(**Baixo**, **Moderado** ou **Alto** engajamento) e uma nova janela começa. "
        "Um indicador ao vivo mostra se os olhos estão abertos ou fechados."
    )

    detector, predictor, modelo = carregar_recursos()
    classes_modelo = [int(c) for c in modelo.classes_]

    st.session_state.setdefault("historico", [])

    iniciar = st.checkbox("Ativar webcam")

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        cronometro_placeholder = st.empty()
        alerta_olhos_placeholder = st.empty()
        parcial_placeholder = st.empty()
        resultado_placeholder = st.empty()
        status_placeholder = st.empty()
    linha_tempo_placeholder = st.empty()
    grafico_placeholder = st.empty()

    if iniciar:
        # Nova captura: zera o histórico ao (re)ativar a webcam
        if not st.session_state.get("_capturando", False):
            st.session_state["historico"] = []
            st.session_state["_capturando"] = True

        t_inicio = time.time()

        # Estado da janela atual
        janela_num = 1
        janela_proba = []
        janela_t0 = time.time()
        janela_hora0 = datetime.now()
        ema_prob = None
        frames_com_rosto = 0
        frames_olhos_fechados = 0

        # Calibração / estado do EAR
        ear_calib = []
        ear_base = None
        t_ear_baixo = None          # instante em que o EAR caiu abaixo do limiar

        linha_tempo = deque(maxlen=8)
        serie_janelas = []

        cap = cv2.VideoCapture(0)
        try:
            while iniciar:
                inicio = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.error("Não foi possível acessar a webcam.")
                    break

                features, face, ear = analisar_rosto(frame, detector, predictor)
                rosto_valido = (
                    features is not None
                    and face is not None
                    and face.width() >= LARGURA_MIN_ROSTO
                )

                agora = time.time()
                calibrando = ear_base is None and (agora - t_inicio) < CALIBRACAO_SEG
                olhos_fechados_agora = False
                segurando = 0.0
                limiar_fechado = (
                    ear_base * EAR_FRAC_FECHADO if ear_base else EAR_FALLBACK
                )

                if rosto_valido:
                    frames_com_rosto += 1

                    # Coleta o EAR natural durante a calibração e fixa a base
                    if ear is not None and (agora - t_inicio) < CALIBRACAO_SEG:
                        ear_calib.append(ear)
                    if ear_base is None and (agora - t_inicio) >= CALIBRACAO_SEG and ear_calib:
                        ear_base = float(np.median(ear_calib))
                        limiar_fechado = ear_base * EAR_FRAC_FECHADO

                    # Cronômetro do olho fechado: precisa segurar > FECHADO_SEG
                    if not calibrando and ear is not None and ear < limiar_fechado:
                        if t_ear_baixo is None:
                            t_ear_baixo = agora
                        segurando = agora - t_ear_baixo
                    else:
                        t_ear_baixo = None

                    olhos_fechados_agora = segurando >= FECHADO_SEG
                    if olhos_fechados_agora:
                        frames_olhos_fechados += 1

                    proba_raw = modelo.predict_proba([features])[0]
                    proba = np.zeros(4)
                    for i, c in enumerate(classes_modelo):
                        proba[c] = proba_raw[i]
                    janela_proba.append(proba)

                    p_frame = prob_engajado(proba)
                    ema_prob = p_frame if ema_prob is None else \
                        ALPHA_EMA * p_frame + (1 - ALPHA_EMA) * ema_prob

                    # Desenho no frame: verde = olhos abertos, vermelho = fechados
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cor = (0, 0, 255) if olhos_fechados_agora else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                    if olhos_fechados_agora:
                        cv2.putText(
                            frame, "OLHOS FECHADOS", (x, max(y - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
                        )

                # --- Alerta: só aparece quando os olhos estão fechados ---
                if olhos_fechados_agora:
                    alerta_olhos_placeholder.error(
                        f"🙈 OLHOS FECHADOS há {segurando:.1f}s"
                    )
                else:
                    alerta_olhos_placeholder.empty()

                # --- Cronômetro da janela ---
                decorrido = time.time() - janela_t0
                restante = max(0.0, JANELA_SEG - decorrido)
                cronometro_placeholder.progress(
                    min(decorrido / JANELA_SEG, 1.0),
                    text=(
                        f"⏱️ Janela {janela_num} — início {janela_hora0.strftime('%H:%M:%S')} "
                        f"· faltam {restante:04.1f}s"
                    ),
                )

                # IMPORTANTE: usamos `rosto_valido` (se há rosto ESTE frame),
                # e não `ema_prob is not None`. O ema_prob, uma vez calculado,
                # nunca mais volta a ser None — então checar só ele fazia a
                # tela ficar "travada" mostrando o último valor conhecido
                # mesmo depois que a pessoa saía do quadro. Com rosto_valido,
                # a mensagem "aguardando rosto" volta a aparecer sempre que
                # o rosto some do quadro atual.
                if rosto_valido and ema_prob is not None:
                    parcial_placeholder.caption(
                        f"Parcial ao vivo: P(engajado) ≈ {ema_prob:.0%} "
                        f"· {len(janela_proba)} amostra(s) nesta janela"
                    )
                else:
                    parcial_placeholder.caption("Parcial ao vivo: aguardando rosto…")

                status_placeholder.caption(
                    f"{len(janela_proba)} amostra(s) · "
                    f"olhos fechados em {frames_olhos_fechados}/{max(frames_com_rosto, 1)} frames"
                )

                # --- Fechamento da janela ---
                if decorrido >= JANELA_SEG:
                    hora_fim = datetime.now()
                    fim_seg = round(time.time() - t_inicio, 1)
                    inicio_seg = round(janela_t0 - t_inicio, 1)
                    olhos_fechados_janela = (
                        frames_com_rosto > 0
                        and frames_olhos_fechados / frames_com_rosto >= 0.5
                    )

                    if len(janela_proba) >= MIN_AMOSTRAS:
                        media = np.mean(janela_proba, axis=0)
                        p_eng = prob_engajado(media)
                        cat = categorizar(p_eng, olhos_fechados_janela)
                        rotulo = CATEGORIAS[cat]
                        if olhos_fechados_janela:
                            rotulo += " (olhos fechados)"

                        resultado_placeholder.markdown(
                            f"### {rotulo}\nP(engajado) = **{p_eng:.0%}**"
                        )
                        st.session_state["historico"].append({
                            "janela": janela_num,
                            "hora_inicio": janela_hora0.strftime("%H:%M:%S"),
                            "hora_fim": hora_fim.strftime("%H:%M:%S"),
                            "inicio_seg": inicio_seg,
                            "fim_seg": fim_seg,
                            "prob_engajado": round(float(p_eng), 3),
                            "categoria": cat,
                            "amostras": len(janela_proba),
                            "olhos_fechados": bool(olhos_fechados_janela),
                        })
                        serie_janelas.append(
                            {"janela": janela_num, "P(engajado)": round(float(p_eng), 3)}
                        )
                        linha_tempo.appendleft(
                            f"Janela {janela_num} · {janela_hora0.strftime('%H:%M:%S')}"
                            f"–{hora_fim.strftime('%H:%M:%S')} · {rotulo} ({p_eng:.0%})"
                        )
                    else:
                        resultado_placeholder.markdown(
                            f"### ⏳ Janela {janela_num} sem amostras suficientes"
                        )
                        linha_tempo.appendleft(
                            f"Janela {janela_num} · {janela_hora0.strftime('%H:%M:%S')}"
                            f"–{hora_fim.strftime('%H:%M:%S')} · sem dados suficientes"
                        )

                    # Reinicia a janela
                    janela_num += 1
                    janela_proba = []
                    janela_t0 = time.time()
                    janela_hora0 = datetime.now()
                    frames_com_rosto = 0
                    frames_olhos_fechados = 0
                    st.toast(f"🔄 Nova janela iniciada ({janela_hora0.strftime('%H:%M:%S')})")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                if linha_tempo:
                    linha_tempo_placeholder.markdown(
                        "**Linha do tempo das janelas**\n\n"
                        + "\n".join(f"- {linha}" for linha in linha_tempo)
                    )
                if len(serie_janelas) >= 1:
                    grafico_placeholder.altair_chart(
                        grafico_combo(pd.DataFrame(serie_janelas), "janela", "P(engajado)"),
                        use_container_width=True,
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

        st.markdown("**P(engajado) por janela**")
        st.altair_chart(
            grafico_combo(df, "janela", "prob_engajado"),
            use_container_width=True,
        )

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
