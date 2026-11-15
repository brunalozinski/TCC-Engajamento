# %% [markdown]
# # EngageNet: embeddings (DINOv2) + BiLSTM, com tratamento de desbalanceamento
# Entrada: dataset Kaggle `engagenet-seq` (recortes de rosto, 16 frames por video,
# gerados por engagenet_sequencias.py). Divisao oficial Train/Validation/Test.

# %% Setup
import subprocess, sys
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers"], check=False)

# %% Embeddings
import numpy as np, pandas as pd, torch, cv2
from pathlib import Path
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

ENTRADA = next(p for p in Path('/kaggle/input').rglob('manifesto_seq.csv')).parent
SAIDA = Path('/kaggle/working/emb'); SAIDA.mkdir(parents=True, exist_ok=True)
dev = 'cuda'
NOME = ENTRADA.name   # prefixo dos arquivos de saida (engagenet-seq, daisee-seq3, ...)
enc = AutoModel.from_pretrained('facebook/dinov2-base', dtype=torch.float16).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

man = pd.read_csv(ENTRADA / 'manifesto_seq.csv', dtype={'pessoa': str, 'clipe': str})
man = man[man.status == 'ok'].reset_index(drop=True)
N_FRAMES = len(list((ENTRADA / man.divisao[0] / man.clipe[0]).glob('*.jpg')))
print('frames por video:', N_FRAMES)
print(man.groupby(['divisao', 'engajamento']).size().unstack(fill_value=0))

class Clipes(Dataset):
    def __init__(self, m): self.m = m.reset_index(drop=True)
    def __len__(self): return len(self.m)
    def __getitem__(self, i):
        r = self.m.iloc[i]
        imgs = [cv2.imread(str(ENTRADA / r.divisao / r.clipe / f'{k:02d}.jpg'))[:, :, ::-1]
                for k in range(N_FRAMES)]
        return torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))

@torch.no_grad()
def embutir(m, lote=16):
    out = []
    for x in tqdm(DataLoader(Clipes(m), batch_size=lote, num_workers=4),
                  total=(len(m) + lote - 1) // lote):
        b = x.shape[0]
        x = x.to(dev).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        h = enc(pixel_values=((x - MEAN) / STD).half()).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()
        out.append(e.reshape(b, N_FRAMES, -1))
    return np.concatenate(out)

for div in ('Train', 'Validation', 'Test'):
    arq = SAIDA / f'{div}.npz'
    m = man[man.divisao == div].reset_index(drop=True)
    if arq.exists() or len(m) == 0:
        continue
    X = embutir(m)
    np.savez(arq, X=X.astype(np.float16), y=m.engajamento.values,
             pessoa=m.pessoa.astype(str).values, clipe=m.clipe.values)
    print(div, X.shape, flush=True)

# %% LSTM
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score

NC = 4
def carregar(div):
    d = np.load(SAIDA / f'{div}.npz', allow_pickle=True)
    return d['X'].astype(np.float32), d['y'].astype(int)
Xtr, ytr = carregar('Train'); Xva, yva = carregar('Validation'); Xte, yte = carregar('Test')
mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]
print('treino por classe:', np.bincount(ytr, minlength=NC))

class ModeloLSTM(nn.Module):
    def __init__(self, d_in, h=128, p=0.4):
        super().__init__()
        self.drop_in = nn.Dropout(p)
        self.lstm = nn.LSTM(d_in, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * h, NC)
    def forward(self, x):
        o, _ = self.lstm(self.drop_in(x))
        return self.fc(self.drop(o.mean(1)))

def pesos_classe(y, modo):
    n = np.bincount(y, minlength=NC).astype(float)
    w = {'inverso': n.sum() / (NC * n), 'sqrt': np.sqrt(n.sum() / (NC * n))}.get(modo, np.ones(NC))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)

def prever(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.tensor(X[i:i + 512], device=dev)).softmax(-1).cpu()
                          for i in range(0, len(X), 512)]).numpy()

def treinar(pesos='nenhum', sampler=False, epocas=60, paciencia=12, seed=0, foco=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModeloLSTM(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    w = pesos_classe(ytr, pesos)
    if sampler:
        n = np.bincount(ytr, minlength=NC); p = (1 / np.sqrt(n))[ytr]; p = p / p.sum()
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    melhor, estado, sem_ganho = -1, None, 0
    for ep in range(epocas):
        model.train()
        idx = np.random.choice(len(ytr), len(ytr), p=p) if sampler else np.random.permutation(len(ytr))
        for i in range(0, len(idx), 64):
            b = torch.tensor(idx[i:i + 64], device=dev)
            x = Xt[b] + 0.1 * torch.randn_like(Xt[b])
            x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=dev) > 0.1)
            logit = model(x)
            ce = F.cross_entropy(logit, yt[b], weight=w, label_smoothing=0.05, reduction='none')
            if foco:
                pt = torch.exp(-F.cross_entropy(logit, yt[b], reduction='none'))
                ce = ce * (1 - pt) ** foco
            opt.zero_grad(); ce.mean().backward(); opt.step()
        f1 = f1_score(yva, prever(model, Xva).argmax(1), average='macro')
        if f1 > melhor:
            melhor, sem_ganho = f1, 0
            estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sem_ganho += 1
            if sem_ganho >= paciencia: break
    model.load_state_dict(estado)
    return model, melhor

# %% Estrategias contra o desbalanceamento, 5 sementes cada
configs = {
    'sem tratamento':        dict(pesos='nenhum'),
    'pesos sqrt':            dict(pesos='sqrt'),
    'sampler + pesos sqrt':  dict(pesos='sqrt', sampler=True),
    'focal(2) + pesos sqrt': dict(pesos='sqrt', foco=2.0),
}
linhas, melhor_modelo, melhor_f1 = [], None, -1
for nome, cfg in configs.items():
    for s in range(5):
        m, f1v = treinar(seed=s, **cfg)
        pr = prever(m, Xte); pt = pr.argmax(1)
        linhas.append(dict(config=nome, semente=s, f1_val=f1v,
                           f1_test=f1_score(yte, pt, average='macro'),
                           bacc_test=balanced_accuracy_score(yte, pt), acc_test=(pt == yte).mean(),
                           auc_test=roc_auc_score(yte, pr, multi_class='ovr', average='macro',
                                                  labels=range(NC))))
        if f1v > melhor_f1:
            melhor_f1, melhor_modelo, melhor_nome = f1v, m, nome
    print(nome, 'ok', flush=True)
df = pd.DataFrame(linhas)
df.to_csv(f'/kaggle/working/resultados_{NOME}.csv', index=False)
print(df.groupby('config')[['f1_val', 'f1_test', 'bacc_test', 'acc_test', 'auc_test']].agg(['mean', 'std']).round(3).to_string())

# %% Relatorio do melhor modelo (escolhido pela validacao)
pr = prever(melhor_modelo, Xte); pt = pr.argmax(1)
print('Melhor (pela validacao):', melhor_nome)
print(classification_report(yte, pt, labels=range(NC), zero_division=0,
                            target_names=['Not-Engaged', 'Barely', 'Engaged', 'Highly']))
print(confusion_matrix(yte, pt, labels=range(NC)))
print('AUC macro (OvR):', round(roc_auc_score(yte, pr, multi_class='ovr', average='macro', labels=range(NC)), 3))
print('AUC por classe:', [round(roc_auc_score(yte == c, pr[:, c]), 3) for c in range(NC)])
torch.save(melhor_modelo.state_dict(), f'/kaggle/working/lstm_{NOME}.pt')
