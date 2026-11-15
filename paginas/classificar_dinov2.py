"""
Demo ao vivo com os experimentos C e E do TCC (DINOv2 + LSTM, 2 classes).

O modelo devolve UMA probabilidade, P(engajado). As tres zonas mostradas
(Baixo / Indefinido / Alto) sao uma camada de apresentacao sobre ela, com cortes
calibrados na Validacao (calibrar_zonas.py). "Indefinido" significa que o modelo
esta em duvida; nao e um nivel de engajamento aprendido.
"""

import json
import time
from collections import deque
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from features import RAIZ, criar_detector_predictor, analisar_rosto
from motor_dinov2_lstm import (
    VARIANTES, PASTA_MODELO, PASTA_MODELO_FT, criar_motor, recortar_por_caixa,
    MIN_EMBEDDINGS,
)
from paginas.classificar import (
    FPS_ALVO, JANELA_SEG, LARGURA_MIN_ROSTO, ALPHA_EMA, CALIBRACAO_SEG,
    EAR_FRAC_FECHADO, EAR_FALLBACK, FECHADO_SEG,
)

EMB_INTERVALO_SEG = 2.0       # gera 1 embedding DINOv2 a cada 2 s (custo na CPU)

# Qualidade da janela: abaixo disso o resultado aparece como "pouco confiavel"
CONFIAVEL_MIN_EMB = 6         # embeddings na janela (o minimo para calcular e MIN_EMBEDDINGS)
CONFIAVEL_MIN_ROSTO = 0.60    # fracao dos frames da janela com rosto valido

# Tendencia da sessao: inclinacao (em P por janela) acima da qual se diz que sobe/cai
LIMIAR_TENDENCIA = 0.02

VARIANTE_PADRAO = "C"         # experimento aberto por padrao (tambem da para escolher na tela)

ZONA_ROTULO = {
    "baixo": "🔴 Baixo engajamento",
    "indefinido": "🟡 Indefinido (o modelo está em dúvida)",
    "alto": "🟢 Alto engajamento",
}
ZONA_COR = {"baixo": "#e34948", "indefinido": "#eda100", "alto": "#1baf7a"}

DESCRICAO = {
    "C": (
        "cada rosto é transformado em um vetor de 1.536 números pelo **DINOv2 congelado** "
        "(os pesos dele não mudam) e uma rede **LSTM** lê 3 desses vetores em ordem do tempo.",
        "No teste balanceado do TCC (clipes de ~10 s, 2 classes), média de 5 sementes: "
        "F1 macro 0,740 e AUC 0,813.",
    ),
    "E": (
        "igual ao C, mas as **2 últimas camadas do DINOv2 foram ajustadas (fine-tuning parcial)** "
        "com os vídeos do projeto, junto com o LSTM. Usa a melhor semente pela validação.",
        "No teste balanceado do TCC (clipes de ~10 s, 2 classes), média de 5 sementes: "
        "F1 macro 0,753 e AUC 0,826. A diferença para o C é pequena e está dentro da "
        "variação entre sementes.",
    ),
}

ZONAS_PADRAO = {"corte_baixo": 0.40, "corte_alto": 0.65, "pureza_alvo": None,
                "base": "valores padrão, NÃO calibrados"}


def carregar_zonas(variante):
    """Cortes das zonas do experimento (calibrados na Validacao)."""
    pasta = {"C": PASTA_MODELO, "E": PASTA_MODELO_FT}[variante]
    arq = pasta / "zonas.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    return ZONAS_PADRAO


def zona_da_probabilidade(p, zonas, olhos_fechados=False):
    if olhos_fechados or p <= zonas["corte_baixo"]:
        return "baixo"
    if p >= zonas["corte_alto"]:
        return "alto"
    return "indefinido"


@st.cache_resource(show_spinner="Carregando DINOv2 e o LSTM (a primeira vez baixa o DINOv2, ~350 MB)...")
def carregar_recursos(variante):
    detector, predictor = criar_detector_predictor()
    return detector, predictor, criar_motor(variante)


# --------------------------------------------------------------------------
# Item 1: resultado principal = probabilidade, com barra de zonas
# --------------------------------------------------------------------------
def barra_zonas_html(p, zonas):
    """Barra de 0 a 100% com as 3 zonas ao fundo, o marcador da janela e o corte de 50%."""
    b, a = zonas["corte_baixo"] * 100, zonas["corte_alto"] * 100
    pos = min(max(p, 0.0), 1.0) * 100
    return f"""
<div style="position:relative;margin:6px 0 28px 0;height:26px;border-radius:6px;overflow:hidden;display:flex">
  <div style="width:{b}%;background:{ZONA_COR['baixo']};opacity:.35"></div>
  <div style="width:{a - b}%;background:{ZONA_COR['indefinido']};opacity:.35"></div>
  <div style="width:{100 - a}%;background:{ZONA_COR['alto']};opacity:.35"></div>
  <div style="position:absolute;left:50%;top:0;bottom:0;border-left:2px dashed #52514e"></div>
  <div style="position:absolute;left:calc({pos}% - 3px);top:-2px;bottom:-2px;width:6px;background:#0b0b0b;border-radius:3px"></div>
</div>
<div style="position:relative;height:0;font-size:.75rem;color:#52514e">
  <span style="position:absolute;left:0;top:-22px">0%</span>
  <span style="position:absolute;left:calc({b}% - 12px);top:-22px">{b:.0f}%</span>
  <span style="position:absolute;left:calc(50% - 22px);top:-22px">50% (corte do modelo)</span>
  <span style="position:absolute;left:calc({a}% - 12px);top:-22px">{a:.0f}%</span>
  <span style="position:absolute;right:0;top:-22px">100%</span>
</div>
<div style="height:14px"></div>
"""


def mostrar_resultado(placeholder, janela):
    zona = janela["zona"]
    rotulo = ZONA_ROTULO[zona] + (" (olhos fechados)" if janela["olhos_fechados"] else "")
    confianca = ("✅ qualidade boa" if janela["confiavel"]
                 else f"⚠️ pouco confiável: {janela['motivo']}")
    with placeholder.container():
        st.markdown(f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>"
                    f"{janela['prob_engajado']:.0%}</div>"
                    f"<div style='color:#52514e'>P(engajado) da janela {janela['janela']}</div>",
                    unsafe_allow_html=True)
        st.markdown(barra_zonas_html(janela["prob_engajado"], janela["_zonas"]),
                    unsafe_allow_html=True)
        st.markdown(f"**{rotulo}**")
        # Item 3: qualidade do dado
        st.caption(f"{confianca} · {janela['amostras']} embeddings · rosto em "
                   f"{janela['pct_rosto']:.0%} dos frames · olhos fechados em "
                   f"{janela['pct_olhos_fechados']:.0%}")


# --------------------------------------------------------------------------
# Item 2: grafico por janela com faixas das zonas e linha em 50%
# --------------------------------------------------------------------------
def grafico_janelas(df, zonas):
    b, a = zonas["corte_baixo"], zonas["corte_alto"]
    faixas = pd.DataFrame({
        "y0": [0.0, b, a], "y1": [b, a, 1.0],
        "zona": ["Baixo", "Indefinido", "Alto"],
    })
    fundo = alt.Chart(faixas).mark_rect(opacity=0.18).encode(
        y=alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), title="P(engajado)",
                axis=alt.Axis(format="%")),
        y2="y1:Q",
        color=alt.Color("zona:N", title="Zona",
                        scale=alt.Scale(domain=["Baixo", "Indefinido", "Alto"],
                                        range=[ZONA_COR["baixo"], ZONA_COR["indefinido"], ZONA_COR["alto"]])),
    )
    corte = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[5, 4], color="#52514e").encode(y="y:Q")
    d = df.assign(confiança=np.where(df["confiavel"], "boa", "pouco confiável"))
    base = alt.Chart(d).encode(x=alt.X("janela:O", title="Janela", axis=alt.Axis(labelAngle=0)))
    linha = base.mark_line(color="#08519c").encode(y="prob_engajado:Q")
    pontos = base.mark_point(size=110, filled=True, opacity=1).encode(
        y="prob_engajado:Q",
        color=alt.Color("confiança:N", title="Qualidade da janela",
                        scale=alt.Scale(domain=["boa", "pouco confiável"], range=["#08519c", "#9a9992"])),
        tooltip=["janela", alt.Tooltip("prob_engajado:Q", format=".0%", title="P(engajado)"),
                 "amostras", "confiança"],
    )
    return (fundo + corte + linha + pontos).resolve_scale(color="independent").properties(height=320)


# --------------------------------------------------------------------------
# Item 4: resumo da sessao com significado
# --------------------------------------------------------------------------
def tendencia(df):
    """Inclinacao de P(engajado) ao longo das janelas confiaveis."""
    d = df[df["confiavel"]]
    if len(d) < 3:
        return None, "poucas janelas confiáveis para avaliar a tendência (mínimo 3)"
    inc = float(np.polyfit(np.arange(len(d)), d["prob_engajado"].values, 1)[0])
    if inc > LIMIAR_TENDENCIA:
        texto = "subindo"
    elif inc < -LIMIAR_TENDENCIA:
        texto = "caindo"
    else:
        texto = "estável"
    return inc, f"{texto} ({inc:+.3f} de P por janela)".replace(".", ",")


def montar_resumo(historico):
    df = pd.DataFrame(historico)
    if df.empty:
        return df, {}
    ok = df[df["confiavel"]]
    n, n_ok = len(df), len(ok)
    resumo = {
        "Início": df["hora_inicio"].iloc[0],
        "Fim": df["hora_fim"].iloc[-1],
        "Janelas concluídas": n,
        "Janelas com qualidade boa": f"{n_ok} de {n}",
        "Duração total (s)": round(df["fim_seg"].iloc[-1] - df["inicio_seg"].iloc[0], 1),
    }
    if n_ok:
        acima = int((ok["prob_engajado"] >= 0.5).sum())
        resumo["Engajado (P ≥ 50%)"] = f"{acima} de {n_ok} janelas boas ({acima / n_ok:.0%})"
        resumo["P(engajado) média (janelas boas)"] = f"{ok['prob_engajado'].mean():.0%}"
        for chave, rotulo in ZONA_ROTULO.items():
            k = int((ok["zona"] == chave).sum())
            resumo[rotulo] = f"{k} janela(s) · {k / n_ok:.0%}"
    else:
        resumo["Resultado"] = "nenhuma janela com qualidade boa (veja a coluna 'motivo')"
    resumo["Tendência"] = tendencia(df)[1]
    resumo["Janelas com olhos fechados"] = int(df["olhos_fechados"].sum())
    return df, resumo


def exibir():
    st.header("Classificação em Tempo Real — DINOv2 + LSTM")
    chaves = list(VARIANTES)
    variante = st.radio(
        "Experimento do TCC usado no classificador",
        chaves, index=chaves.index(VARIANTE_PADRAO),
        format_func=lambda k: VARIANTES[k][0], key="variante_dino", horizontal=True,
    )
    zonas = carregar_zonas(variante)
    como, metricas = DESCRICAO[variante]
    st.write(
        f"Esta versão usa o modelo do **experimento {variante}** do TCC: {como} "
        f"A decisão é fechada a cada **janela de {JANELA_SEG} segundos**, usando os "
        f"rostos capturados a cada {EMB_INTERVALO_SEG:.0f} s."
    )
    st.warning(
        "O modelo só distingue **engajado** de **não engajado** e devolve uma probabilidade. "
        "As zonas abaixo são uma leitura dessa probabilidade, não níveis de engajamento "
        "aprendidos: **\"Indefinido\" quer dizer que o modelo está em dúvida.** "
        "É uma estimativa feita a partir do rosto, não mede o estado real da pessoa."
    )
    st.info(
        f"{metricas} A demo usa janelas de 20 s e uma webcam, condições diferentes das do "
        "teste: esses valores NÃO se aplicam diretamente a ela."
    )
    with st.expander("Como as zonas foram calibradas"):
        st.write(
            f"Cortes: **Baixo** se P ≤ {zonas['corte_baixo']:.0%}, **Alto** se P ≥ "
            f"{zonas['corte_alto']:.0%}, **Indefinido** entre os dois. "
            f"Base: {zonas['base']}."
        )
        relatorio = RAIZ / "calibracao_zonas" / f"relatorio_{variante}.txt"
        if relatorio.exists():
            st.code(relatorio.read_text(encoding="utf-8"), language=None)

    detector, predictor, motor = carregar_recursos(variante)
    if st.session_state.get("_variante_hist") != variante:
        st.session_state["historico_dino"] = []      # relatório de um modelo não mistura com o do outro
        st.session_state["_variante_hist"] = variante
    st.session_state.setdefault("historico_dino", [])
    iniciar = st.checkbox("Ativar webcam", key="webcam_dino")

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        cronometro_placeholder = st.empty()
        alerta_olhos_placeholder = st.empty()
        parcial_placeholder = st.empty()
        status_placeholder = st.empty()
        resultado_placeholder = st.empty()
    linha_tempo_placeholder = st.empty()
    grafico_placeholder = st.empty()

    if iniciar:
        if not st.session_state.get("_capturando_dino", False):
            st.session_state["historico_dino"] = []
            st.session_state["_capturando_dino"] = True

        t_inicio = time.time()
        janela_num = 1
        janela_emb = []
        janela_t0 = time.time()
        janela_hora0 = datetime.now()
        t_ultimo_emb = 0.0
        frames_total = 0
        frames_com_rosto = 0
        frames_olhos_fechados = 0
        ear_calib, ear_base, t_ear_baixo = [], None, None
        ema_prob = None
        linha_tempo = deque(maxlen=8)
        serie = []

        cap = cv2.VideoCapture(0)
        try:
            while iniciar:
                inicio = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.error("Não foi possível acessar a webcam.")
                    break
                frames_total += 1

                _, face, ear = analisar_rosto(frame, detector, predictor)
                rosto_valido = face is not None and face.width() >= LARGURA_MIN_ROSTO

                agora = time.time()
                calibrando = ear_base is None and (agora - t_inicio) < CALIBRACAO_SEG
                olhos_fechados_agora = False
                segurando = 0.0
                limiar_fechado = ear_base * EAR_FRAC_FECHADO if ear_base else EAR_FALLBACK

                if rosto_valido:
                    frames_com_rosto += 1
                    if ear is not None and (agora - t_inicio) < CALIBRACAO_SEG:
                        ear_calib.append(ear)
                    if ear_base is None and (agora - t_inicio) >= CALIBRACAO_SEG and ear_calib:
                        ear_base = float(np.median(ear_calib))
                        limiar_fechado = ear_base * EAR_FRAC_FECHADO

                    if not calibrando and ear is not None and ear < limiar_fechado:
                        if t_ear_baixo is None:
                            t_ear_baixo = agora
                        segurando = agora - t_ear_baixo
                    else:
                        t_ear_baixo = None
                    olhos_fechados_agora = segurando >= FECHADO_SEG
                    if olhos_fechados_agora:
                        frames_olhos_fechados += 1

                    # Embedding DINOv2 de 1 frame a cada EMB_INTERVALO_SEG
                    if agora - t_ultimo_emb >= EMB_INTERVALO_SEG:
                        recorte = recortar_por_caixa(frame, face)
                        janela_emb.append(motor.embedding(recorte))
                        t_ultimo_emb = agora
                        if len(janela_emb) >= MIN_EMBEDDINGS:
                            p_parcial = motor.p_engajado_janela(janela_emb, n_sorteios=3)
                            ema_prob = p_parcial if ema_prob is None else \
                                ALPHA_EMA * p_parcial + (1 - ALPHA_EMA) * ema_prob

                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cor = (0, 0, 255) if olhos_fechados_agora else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                    if olhos_fechados_agora:
                        cv2.putText(frame, "OLHOS FECHADOS", (x, max(y - 12, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if olhos_fechados_agora:
                    alerta_olhos_placeholder.error(f"🙈 OLHOS FECHADOS há {segurando:.1f}s")
                else:
                    alerta_olhos_placeholder.empty()

                decorrido = time.time() - janela_t0
                restante = max(0.0, JANELA_SEG - decorrido)
                cronometro_placeholder.progress(
                    min(decorrido / JANELA_SEG, 1.0),
                    text=(f"⏱️ Janela {janela_num} — início {janela_hora0.strftime('%H:%M:%S')} "
                          f"· faltam {restante:04.1f}s"),
                )

                if rosto_valido and ema_prob is not None:
                    parcial_placeholder.caption(
                        f"Parcial ao vivo: P(engajado) ≈ {ema_prob:.0%} "
                        f"· {len(janela_emb)} embedding(s) nesta janela")
                else:
                    parcial_placeholder.caption(
                        "Parcial ao vivo: aguardando rosto e embeddings suficientes…")
                # Item 3: qualidade ao vivo
                pct_rosto_agora = frames_com_rosto / max(frames_total, 1)
                status_placeholder.caption(
                    f"Qualidade até agora: {len(janela_emb)} embedding(s) · rosto em "
                    f"{pct_rosto_agora:.0%} dos frames · olhos fechados em "
                    f"{frames_olhos_fechados}/{max(frames_com_rosto, 1)} frames")

                if decorrido >= JANELA_SEG:
                    hora_fim = datetime.now()
                    fim_seg = round(time.time() - t_inicio, 1)
                    inicio_seg = round(janela_t0 - t_inicio, 1)
                    pct_rosto = frames_com_rosto / max(frames_total, 1)
                    pct_olhos = frames_olhos_fechados / max(frames_com_rosto, 1)
                    olhos_fechados_janela = frames_com_rosto > 0 and pct_olhos >= 0.5

                    p_eng = motor.p_engajado_janela(janela_emb)
                    if p_eng is not None:
                        motivos = []
                        if len(janela_emb) < CONFIAVEL_MIN_EMB:
                            motivos.append(f"só {len(janela_emb)} embeddings")
                        if pct_rosto < CONFIAVEL_MIN_ROSTO:
                            motivos.append(f"rosto em apenas {pct_rosto:.0%} dos frames")
                        janela = {
                            "janela": janela_num,
                            "hora_inicio": janela_hora0.strftime("%H:%M:%S"),
                            "hora_fim": hora_fim.strftime("%H:%M:%S"),
                            "inicio_seg": inicio_seg, "fim_seg": fim_seg,
                            "prob_engajado": round(p_eng, 3),
                            "zona": zona_da_probabilidade(p_eng, zonas, olhos_fechados_janela),
                            "amostras": len(janela_emb),
                            "pct_rosto": round(pct_rosto, 3),
                            "pct_olhos_fechados": round(pct_olhos, 3),
                            "olhos_fechados": bool(olhos_fechados_janela),
                            "confiavel": not motivos,
                            "motivo": "; ".join(motivos),
                            "experimento": variante,
                        }
                        mostrar_resultado(resultado_placeholder, {**janela, "_zonas": zonas})
                        st.session_state["historico_dino"].append(janela)
                        serie.append(janela)
                        sufixo = "" if janela["confiavel"] else " ⚠️ pouco confiável"
                        linha_tempo.appendleft(
                            f"Janela {janela_num} · {janela_hora0.strftime('%H:%M:%S')}"
                            f"–{hora_fim.strftime('%H:%M:%S')} · {ZONA_ROTULO[janela['zona']]} "
                            f"({p_eng:.0%}){sufixo}")
                    else:
                        resultado_placeholder.markdown(
                            f"### ⏳ Janela {janela_num} sem embeddings suficientes "
                            f"(mínimo {MIN_EMBEDDINGS})")
                        linha_tempo.appendleft(
                            f"Janela {janela_num} · {janela_hora0.strftime('%H:%M:%S')}"
                            f"–{hora_fim.strftime('%H:%M:%S')} · sem dados suficientes")

                    janela_num += 1
                    janela_emb = []
                    janela_t0 = time.time()
                    janela_hora0 = datetime.now()
                    t_ultimo_emb = 0.0
                    frames_total = frames_com_rosto = frames_olhos_fechados = 0
                    st.toast(f"🔄 Nova janela iniciada ({janela_hora0.strftime('%H:%M:%S')})")

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                if linha_tempo:
                    linha_tempo_placeholder.markdown(
                        "**Linha do tempo das janelas**\n\n"
                        + "\n".join(f"- {linha}" for linha in linha_tempo))
                if serie:
                    grafico_placeholder.altair_chart(
                        grafico_janelas(pd.DataFrame(serie), zonas), width="stretch")

                espera = (1.0 / FPS_ALVO) - (time.time() - inicio)
                if espera > 0:
                    time.sleep(espera)
        finally:
            cap.release()
    else:
        st.session_state["_capturando_dino"] = False

    historico = st.session_state.get("historico_dino", [])
    if historico and not iniciar:
        st.divider()
        st.subheader("Relatório da sessão")
        df, resumo = montar_resumo(historico)
        st.table(pd.DataFrame([(k, str(v)) for k, v in resumo.items()], columns=["Métrica", "Valor"]))
        st.markdown("**P(engajado) por janela** (faixas coloridas = zonas; linha tracejada = corte de 50%; "
                    "pontos cinza = janelas pouco confiáveis)")
        st.altair_chart(grafico_janelas(df, zonas), width="stretch")
        saida = df.drop(columns=[c for c in ("_zonas",) if c in df.columns])
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("⬇️ Baixar relatório (CSV)", data=saida.to_csv(index=False).encode("utf-8"),
                           file_name=f"relatorio_dinov2_lstm_exp{variante}_{carimbo}.csv", mime="text/csv")
        resumo_txt = "\n".join(f"{k}: {v}" for k, v in resumo.items())
        st.download_button("⬇️ Baixar resumo (TXT)", data=resumo_txt.encode("utf-8"),
                           file_name=f"resumo_dinov2_lstm_exp{variante}_{carimbo}.txt", mime="text/plain")
        st.caption("Janelas 'pouco confiáveis' (poucos embeddings ou rosto ausente em boa parte dos "
                   "frames) ficam fora das contagens de engajamento e da tendência.")
        if st.button("Limpar relatório", key="limpar_dino"):
            st.session_state["historico_dino"] = []
            st.rerun()
