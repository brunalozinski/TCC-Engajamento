"""
App Streamlit para deteccao de engajamento em tempo real via webcam.

Usa o modelo Swin fine-tuned (treinado em `finetuning.ipynb` a partir dos
rostos extraidos por `extract_faces.py`) para classificar o rosto detectado
como "engajado" ou "n_engajado".

A captura da webcam usa `streamlit-webrtc`, que abre uma conexao WebRTC
(P2P, via STUN/ICE) entre o navegador e o processo Python, dando um feed de
video continuo de verdade. Isso funciona bem rodando localmente (`localhost`).

Se este app for rodado atras de um proxy HTTP-only (ex: SageMaker Studio),
o WebRTC tende a travar em "connecting..." porque o proxy nao encaminha o
trafego de midia (UDP). Nesse cenario, configure um servidor TURN em
RTC_CONFIGURATION ou troque para `streamlit-camera-input-live` (ver
AGENTS.md para detalhes).

Uso:
    streamlit run app.py

Pre-requisito:
    Ter o modelo salvo (pasta com config.json/model.safetensors/preprocessor_config.json)
    gerado por `trainer.save_model(...)` + `processor.save_pretrained(...)`
    em `finetuning.ipynb`, no mesmo diretorio deste app.py (por padrao
    `swin-engajamento-finetuned/`, ver DEFAULT_MODEL_DIR).
"""

from pathlib import Path

import av
import cv2
import torch
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from transformers import AutoImageProcessor, AutoModelForImageClassification

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "swin-engajamento-finetuned"
FACE_MARGIN = 0.2
MIN_FACE_SIZE = 60
NEGATIVE_LABEL_HINTS = ("n_engajado", "nao_engajado", "nao engajado", "not_engaged", "disengaged")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def resolve_model_dir(model_dir: str) -> Path | None:
    """Tenta achar a pasta do modelo tanto relativa ao diretorio atual (cwd)
    quanto relativa ao diretorio do proprio app.py, ja que `streamlit run`
    pode ser executado de um diretorio diferente de onde o modelo foi salvo."""
    raw = Path(model_dir)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, APP_DIR / raw]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


@st.cache_resource(show_spinner="Carregando modelo de engajamento...")
def load_model(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def crop_largest_face(frame_bgr, face_cascade, margin=FACE_MARGIN, min_size=MIN_FACE_SIZE):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    x0 = max(x - mx, 0)
    y0 = max(y - my, 0)
    x1 = min(x + w + mx, frame_bgr.shape[1])
    y1 = min(y + h + my, frame_bgr.shape[0])
    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


@torch.no_grad()
def classify_face(face_bgr, processor, model, device):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=face_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label = model.config.id2label[pred_id]
    score = float(probs[pred_id])
    return label, score


def is_engaged_label(label: str) -> bool:
    label_lower = label.lower()
    return not any(hint in label_lower for hint in NEGATIVE_LABEL_HINTS)


def annotate_frame(frame_bgr, box, label, score, conf_threshold):
    if box is None:
        cv2.putText(frame_bgr, "Rosto nao detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return frame_bgr

    if label is None or score < conf_threshold:
        return frame_bgr

    x0, y0, x1, y1 = box
    engaged = is_engaged_label(label)
    color = (0, 200, 0) if engaged else (0, 0, 220)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
    text = f"{label} ({score:.0%})"
    text_y = max(y0 - 10, 20)
    cv2.putText(frame_bgr, text, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame_bgr


class EngagementVideoProcessor(VideoProcessorBase):
    """Roda em thread separada, processando cada frame recebido via WebRTC."""

    def __init__(self, processor, model, device, face_cascade, conf_threshold):
        self.processor = processor
        self.model = model
        self.device = device
        self.face_cascade = face_cascade
        self.conf_threshold = conf_threshold

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        face, box = crop_largest_face(frame_bgr, self.face_cascade)
        label, score = (None, 0.0)
        if face is not None and face.size > 0:
            label, score = classify_face(face, self.processor, self.model, self.device)

        annotated = annotate_frame(frame_bgr, box, label, score, self.conf_threshold)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def main():
    st.set_page_config(page_title="Detector de Engajamento", layout="centered")
    st.title("Detector de engajamento em tempo real")
    st.caption(
        "Captura a webcam pelo navegador (WebRTC) e classifica o rosto detectado como "
        "engajado / nao engajado usando o modelo Swin fine-tuned."
    )

    with st.sidebar:
        st.header("Configuracoes")
        model_dir = st.text_input(
            "Pasta do modelo",
            value=DEFAULT_MODEL_DIR,
            help="Pasta gerada por trainer.save_model(...) no finetuning.ipynb.",
        )
        conf_threshold = st.slider("Limiar de confianca para exibir o rotulo", 0.0, 1.0, 0.5, 0.05)

    resolved_model_dir = resolve_model_dir(model_dir)
    if resolved_model_dir is None:
        raw = Path(model_dir)
        checked = (
            [str(raw.resolve())]
            if raw.is_absolute()
            else [str((Path.cwd() / raw).resolve()), str((APP_DIR / raw).resolve())]
        )
        st.error(
            f"Pasta do modelo '{model_dir}' nao encontrada. Verificado em:\n"
            + "\n".join(f"- {c}" for c in checked)
            + "\n\nBaixe/extraia a pasta do modelo treinado para dentro "
            f"de `{APP_DIR}` com o nome `{DEFAULT_MODEL_DIR}`, ou informe o caminho absoluto "
            "correto na barra lateral."
        )
        st.caption(f"Diretorio de trabalho atual (cwd): `{Path.cwd()}`")
        st.caption(f"Diretorio deste app.py: `{APP_DIR}`")
        sibling_dirs = sorted(p.name for p in APP_DIR.iterdir() if p.is_dir())
        if sibling_dirs:
            st.caption(f"Pastas encontradas ao lado do app.py: {', '.join(sibling_dirs)}")
        return

    model_dir = str(resolved_model_dir)
    processor, model, device = load_model(model_dir)
    face_cascade = load_face_detector()

    ctx = webrtc_streamer(
        key="engagement-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: EngagementVideoProcessor(
            processor, model, device, face_cascade, conf_threshold
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Atualiza o limiar de confianca na thread de video sem reiniciar o stream.
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = conf_threshold

    if not ctx.state.playing:
        st.info("Clique em 'START' acima e autorize o acesso a camera no navegador.")


if __name__ == "__main__":
    main()
