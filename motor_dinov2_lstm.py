"""
Motores da demo com os experimentos C e E do TCC (2 classes, DAiSEE + EngageNet).

Cadeia: recorte do rosto (224x224) -> embedding DINOv2 de cada frame (token CLS +
media dos patches = 1536 numeros) -> 3 embeddings em ordem -> LSTM bidirecional ->
P(engajado).

  C  MotorDinoLSTM      DINOv2 CONGELADO + LSTM. O modelo e a normalizacao ficam em
                        modelo_dinov2_lstm/ (lstm.pt = melhor semente pela validacao;
                        normalizacao.npz = media e desvio dos embeddings de TREINO).
  E  MotorDinoFineTune  DINOv2 com FINE-TUNING PARCIAL (2 ultimos blocos ajustados) +
                        LSTM. Pesos em modelo_dinov2_finetune/modelo.pt (semente 0, a
                        melhor pela validacao): os 2 ultimos blocos do DINOv2 e a
                        layernorm final, a LayerNorm de entrada, o LSTM e a camada final.

Os treinos (TCC/kaggle/combinado_lstm.py, engagenet_lstm.py e dinov2_finetune.py)
usaram exatamente estes passos; aqui eles sao repetidos para o uso ao vivo.
Use criar_motor("C") ou criar_motor("E").
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parent
PASTA_MODELO = RAIZ / "modelo_dinov2_lstm"
PASTA_MODELO_FT = RAIZ / "modelo_dinov2_finetune"
NOME_DINOV2 = "facebook/dinov2-base"

TAMANHO_RECORTE = 224      # mesmo tamanho dos recortes usados no treino
MARGEM_RECORTE = 0.2       # 20% de cada lado, igual ao face_crop.py do treino
FRAMES_POR_VIDEO = 3       # o modelo foi treinado com 3 frames por video
MIN_EMBEDDINGS = 4         # minimo de embeddings na janela para classificar


class ModeloLSTM(nn.Module):
    """Mesma arquitetura do treino (combinado_lstm.py)."""

    def __init__(self, d_in=1536, h=128, p=0.4, n_classes=2):
        super().__init__()
        self.drop_in = nn.Dropout(p)
        self.lstm = nn.LSTM(d_in, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * h, n_classes)

    def forward(self, x):
        o, _ = self.lstm(self.drop_in(x))
        return self.fc(self.drop(o.mean(1)))


def recortar_por_caixa(frame_bgr, face, margem=MARGEM_RECORTE, tamanho=TAMANHO_RECORTE):
    """Recorte quadrado em torno da caixa do rosto (dlib), com margem, igual ao treino."""
    h, w = frame_bgr.shape[:2]
    lado_rosto = max(face.width(), face.height())
    cx = (face.left() + face.right()) / 2.0
    cy = (face.top() + face.bottom()) / 2.0
    lado = min(int(round(lado_rosto * (1 + 2 * margem))), h, w)
    x0 = min(max(int(round(cx - lado / 2.0)), 0), w - lado)
    y0 = min(max(int(round(cy - lado / 2.0)), 0), h - lado)
    recorte = frame_bgr[y0:y0 + lado, x0:x0 + lado]
    return cv2.resize(recorte, (tamanho, tamanho), interpolation=cv2.INTER_AREA)


class MotorDinoLSTM:
    def __init__(self):
        from transformers import AutoModel

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 so na GPU (como no treino); na CPU usa fp32
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.encoder = AutoModel.from_pretrained(
            NOME_DINOV2, dtype=self.dtype).to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

        self.lstm = ModeloLSTM().to(self.dev)
        self.lstm.load_state_dict(
            torch.load(PASTA_MODELO / "lstm.pt", map_location=self.dev, weights_only=True))
        self.lstm.eval()
        norm = np.load(PASTA_MODELO / "normalizacao.npz")
        self.mu, self.sd = norm["mu"], norm["sd"]

    @torch.no_grad()
    def embedding(self, recorte_bgr):
        """Recorte 224x224 (BGR) -> vetor de 1536 numeros."""
        rgb = recorte_bgr[:, :, ::-1].copy()
        x = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1)[None].float() / 255
        h = self.encoder(pixel_values=((x - self.mean) / self.std).to(self.dtype)).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()[0]

    @torch.no_grad()
    def p_engajado_sequencia(self, tres_embeddings):
        """3 embeddings em ordem cronologica (3, 1536) -> P(engajado) do LSTM."""
        x = (np.asarray(tres_embeddings, dtype=np.float32) - self.mu) / self.sd
        x = torch.from_numpy(x.astype(np.float32))[None].to(self.dev)
        return float(self.lstm(x).softmax(-1)[0, 1].cpu())

    def p_engajado_janela(self, embeddings, n_sorteios=7, semente=0):
        """
        Probabilidade de engajamento de uma janela de tempo.

        O modelo foi treinado com 3 frames por video. A janela tem varios
        embeddings (um a cada poucos segundos), entao ela e dividida em 3 trechos
        iguais e sorteia-se 1 embedding de cada trecho, mantendo a ordem do
        tempo. Isso e repetido n_sorteios vezes e a media das probabilidades e o
        resultado da janela.
        """
        e = np.asarray(embeddings, dtype=np.float32)
        if len(e) < MIN_EMBEDDINGS:
            return None
        trechos = np.array_split(np.arange(len(e)), FRAMES_POR_VIDEO)
        rng = np.random.default_rng(semente)
        probs = []
        for _ in range(n_sorteios):
            idx = [int(rng.choice(t)) for t in trechos]
            probs.append(self.p_engajado_sequencia(e[idx]))
        return float(np.mean(probs))


def _nome_transformers_novo(nome):
    """
    Os pesos do experimento E foram salvos no Kaggle com a versao antiga do
    transformers (attention.attention.query ...). A versao instalada aqui usa
    outros nomes (attention.q_proj ...); so os nomes mudam, os pesos sao os mesmos.
    """
    for velho, novo in (("attention.attention.query", "attention.q_proj"),
                        ("attention.attention.key", "attention.k_proj"),
                        ("attention.attention.value", "attention.v_proj"),
                        ("attention.output.dense", "attention.o_proj")):
        nome = nome.replace(velho, novo)
    return nome


class ModeloFineTune(nn.Module):
    """Cabeca do experimento E (dinov2_finetune.py): LayerNorm de entrada + LSTM."""

    def __init__(self, d_in=1536, h=128, p=0.4, n_classes=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.drop_in = nn.Dropout(p)
        self.lstm = nn.LSTM(d_in, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * h, n_classes)

    def forward(self, e):                      # e: (B, 3, 1536)
        o, _ = self.lstm(self.drop_in(self.norm(e)))
        return self.fc(self.drop(o.mean(1)))


class MotorDinoFineTune(MotorDinoLSTM):
    """Experimento E: mesma interface do MotorDinoLSTM, com o DINOv2 ajustado."""

    def __init__(self):
        from transformers import AutoModel

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.encoder = AutoModel.from_pretrained(NOME_DINOV2, dtype=torch.float32)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

        ckpt = torch.load(PASTA_MODELO_FT / "modelo.pt", map_location="cpu", weights_only=True)
        enc = {_nome_transformers_novo(k[len("enc."):]): v
               for k, v in ckpt.items() if k.startswith("enc.")}
        faltando, inesperadas = self.encoder.load_state_dict(enc, strict=False)
        assert not inesperadas, inesperadas
        self.encoder = self.encoder.to(self.dev).to(self.dtype).eval()

        self.lstm = ModeloFineTune().to(self.dev)
        self.lstm.load_state_dict({k: v for k, v in ckpt.items() if not k.startswith("enc.")})
        self.lstm.eval()

    @torch.no_grad()
    def p_engajado_sequencia(self, tres_embeddings):
        """3 embeddings do DINOv2 ajustado (3, 1536) -> P(engajado). A normalizacao e a LayerNorm do modelo."""
        x = torch.from_numpy(np.asarray(tres_embeddings, dtype=np.float32))[None].to(self.dev)
        return float(self.lstm(x).softmax(-1)[0, 1].cpu())


VARIANTES = {
    "C": ("Experimento C: DINOv2 congelado + LSTM", MotorDinoLSTM),
    "E": ("Experimento E: DINOv2 com fine-tuning parcial + LSTM", MotorDinoFineTune),
}


def criar_motor(variante="C"):
    return VARIANTES[variante][1]()
