# %% [markdown]
# # Swin-tiny congelado + LSTM, 2 classes, DAiSEE + EngageNet (mesmos videos do experimento C)
# Comparacao com o DINOv2 congelado + LSTM: mudam so os embeddings. Mesmos videos, mesmos
# 3 frames, mesmas divisoes, mesmo LSTM, mesmas 5 sementes. Swin-tiny sem treino (congelado).

# %% Setup
import subprocess, sys
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers"], check=False)

# %% Embeddings do Swin-tiny (saida pooled, 768 numeros por frame)
import numpy as np, pandas as pd, torch, cv2, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression

IMG = next(Path('/kaggle/input').rglob('manifesto_imagens.csv')).parent
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NOME = 'swin-tiny-congelado'
enc = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224',
                                dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
man = pd.read_csv(IMG / 'manifesto_imagens.csv', dtype={'pessoa': str, 'clipe': str})
print(man.groupby(['divisao', 'fonte', 'y']).size().unstack(fill_value=0))

class Clipes(Dataset):
    def __init__(self, m): self.m = m.reset_index(drop=True)
    def __len__(self): return len(self.m)
    def __getitem__(self, i):
        r = self.m.iloc[i]
        imgs = [cv2.imread(str(IMG / r.divisao / r.pasta / f'{k:02d}.jpg'))[:, :, ::-1] for k in range(3)]
        return torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))

@torch.no_grad()
def embutir(m, lote=32):
    out = []
    for x in tqdm(DataLoader(Clipes(m), batch_size=lote, num_workers=4),
                  total=(len(m) + lote - 1) // lote):
        b = x.shape[0]
        x = x.to(dev).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        h = enc(pixel_values=((x - MEAN) / STD).to(enc.dtype)).pooler_output
        out.append(h.float().cpu().numpy().reshape(b, 3, -1))
    return np.concatenate(out)

D = {}
for div in ('Train', 'Validation', 'Test'):
    m = man[man.divisao == div].reset_index(drop=True)
    D[div] = dict(X=embutir(m).astype(np.float16), y=m.y.values, y_orig=m.y_orig.values,
                  fonte=m.fonte.values, pessoa=m.pessoa.values, clipe=m.clipe.values,
                  balanceado=m.balanceado.values)
    print(div, D[div]['X'].shape, flush=True)

# %% Dados
