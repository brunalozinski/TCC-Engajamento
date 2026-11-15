# %% [markdown]
# # Swin-tiny com fine-tuning parcial + LSTM, 2 classes, DAiSEE + EngageNet
# Mesmos videos, mesmos 3 frames, mesmas divisoes, mesmas metricas e mesmos hiperparametros do
# experimento do DINOv2 com fine-tuning parcial. Aqui o Swin-tiny nao fica congelado: o ULTIMO
# ESTAGIO (2 blocos, o analogo dos 2 blocos finais do DINOv2) e a layernorm final sao ajustados
# junto com o LSTM, de ponta a ponta. Os 3 primeiros estagios ficam congelados.

# %% Setup
import subprocess, sys
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers"], check=False)

# %% Configuracao e imagens na memoria
import time, copy
import numpy as np, pandas as pd, torch, cv2, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torchvision.ops import roi_align
from transformers import AutoModel
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

SEMENTES = [0, 1, 2, 3, 4]
DIM = 768               # saida pooled do Swin-tiny (ultimo estagio)
LR_EXTRATOR, LR_CABECA, WD = 2e-5, 1e-3, 1e-2
EPOCAS, PACIENCIA, LOTE = 8, 3, 16     # LOTE = videos (x3 frames = imagens)
NOME = 'swin-finetune-ultimo-estagio'
dev = 'cuda'
IMG = next(Path('/kaggle/input').rglob('manifesto_imagens.csv')).parent
man = pd.read_csv(IMG / 'manifesto_imagens.csv', dtype={'pessoa': str, 'clipe': str})
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def ler(r):
    return np.stack([cv2.imread(str(IMG / r.divisao / r.pasta / f'{k:02d}.jpg'))[:, :, ::-1]
                     for k in range(3)])

t0 = time.time()
DADOS = {}
with ThreadPoolExecutor(8) as ex:
    for div in ('Train', 'Validation', 'Test'):
        m = man[man.divisao == div].reset_index(drop=True)
        X = np.stack(list(ex.map(ler, m.itertuples())))          # (n, 3, 224, 224, 3) uint8
        DADOS[div] = dict(m=m, X=torch.from_numpy(X), y=torch.tensor(m.y.values))
        print(div, tuple(X.shape), f'{time.time() - t0:.0f}s', flush=True)

# %% Modelo
class Modelo(nn.Module):
    def __init__(self, h=128, p=0.4):
        super().__init__()
        self.enc = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        for q in self.enc.parameters():
            q.requires_grad = False
        for q in self.enc.encoder.layers[-1].parameters():
            q.requires_grad = True
        for q in self.enc.layernorm.parameters():
            q.requires_grad = True
        self.norm = nn.LayerNorm(DIM)
        self.drop_in = nn.Dropout(p)
        self.lstm = nn.LSTM(DIM, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * h, 2)

    def frames(self, x):                      # x: (B*3, 3, 224, 224) ja normalizado
        return self.enc(pixel_values=x).pooler_output

    def forward(self, x, b):
        e = self.norm(self.frames(x).float()).view(b, 3, -1)
        o, _ = self.lstm(self.drop_in(e))
        return self.fc(self.drop(o.mean(1)))

def treinavel(k):
    return k.startswith('enc.encoder.layers.3.') or k.startswith('enc.layernorm')

def preparar(xb, treino):
    """xb: (B,3,224,224,3) uint8 na CPU -> (B*3,3,224,224) normalizado, com aumento se treino."""
    b = xb.shape[0]
    x = xb.to(dev, non_blocking=True).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
    if treino:
        # mesma janela de recorte/espelho para os 3 frames de um video
        lado = torch.empty(b, device=dev).uniform_(0.8, 1.0) * 224
        x0 = torch.rand(b, device=dev) * (224 - lado)
        y0 = torch.rand(b, device=dev) * (224 - lado)
        cx = torch.stack([x0, y0, x0 + lado, y0 + lado], 1).repeat_interleave(3, 0)
        idx = torch.arange(3 * b, device=dev, dtype=torch.float32).view(-1, 1)
        x = roi_align(x, torch.cat([idx, cx], 1), output_size=224, aligned=True)
        flip = (torch.rand(b, device=dev) < 0.5).repeat_interleave(3, 0).view(-1, 1, 1, 1)
        x = torch.where(flip, x.flip(3), x)
        brilho = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3).repeat_interleave(3, 0)
        contraste = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3).repeat_interleave(3, 0)
        mu = x.mean((1, 2, 3), keepdim=True)
        x = ((x - mu) * contraste + mu) * brilho
        x = x.clamp(0, 1)
    return (x - MEAN) / STD

@torch.no_grad()
def prever(model, X, lote=32):
    model.eval(); out = []
    for i in range(0, len(X), lote):
        xb = X[i:i + lote]
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(preparar(xb, False), xb.shape[0]).float().softmax(-1).cpu())
    return torch.cat(out).numpy()

# verificacao rapida de formas antes de treinar
_m = Modelo().to(dev)
with torch.autocast('cuda', dtype=torch.float16):
    _y = _m(preparar(DADOS['Train']['X'][:2], True), 2)
print('teste de forma:', tuple(_y.shape), '| parametros treinaveis:',
      sum(q.numel() for q in _m.parameters() if q.requires_grad))
del _m, _y; torch.cuda.empty_cache()

# %% Treino
def treinar(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Modelo().to(dev)
    ext = [q for q in model.enc.parameters() if q.requires_grad]
    cab = [q for n, q in model.named_parameters() if not n.startswith('enc.')]
    opt = torch.optim.AdamW([dict(params=ext, lr=LR_EXTRATOR), dict(params=cab, lr=LR_CABECA)],
                            weight_decay=WD)
    scaler = torch.amp.GradScaler()
    Xtr, ytr = DADOS['Train']['X'], DADOS['Train']['y']
    yva = DADOS['Validation']['y'].numpy()
    melhor, estado, sem_ganho, hist = -1, None, 0, []
    for ep in range(EPOCAS):
        model.train(); t = time.time(); perda = 0; nb = 0
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), LOTE):
            ib = perm[i:i + LOTE]
            xb, yb = Xtr[ib], ytr[ib].to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                logit = model(preparar(xb, True), xb.shape[0])
            loss = F.cross_entropy(logit.float(), yb, label_smoothing=0.05)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            perda += loss.item(); nb += 1
        pv = prever(model, DADOS['Validation']['X'])
        f1 = f1_score(yva, pv.argmax(1), average='macro')
        hist.append(dict(semente=seed, epoca=ep, perda_treino=perda / nb, f1_val=f1,
                         auc_val=roc_auc_score(yva, pv[:, 1]), segundos=time.time() - t))
        print(hist[-1], flush=True)
        if f1 > melhor:
            melhor, sem_ganho = f1, 0
            estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if not k.startswith('enc.') or treinavel(k)}
        else:
            sem_ganho += 1
            if sem_ganho >= PACIENCIA: break
    model.load_state_dict(estado, strict=False)
    return model, melhor, hist

def metricas(y, pr):
    p = pr.argmax(1)
    return dict(f1=f1_score(y, p, average='macro'), bacc=balanced_accuracy_score(y, p),
                acc=(p == y).mean(), auc=roc_auc_score(y, pr[:, 1]))

yte = DADOS['Test']['y'].numpy(); mte = DADOS['Test']['m']
bal, fonte = mte.balanceado.values.astype(bool), mte.fonte.values.astype(str)
linhas, historico, melhor_f1 = [], [], -1
for s in SEMENTES:
    t = time.time()
    model, f1v, hist = treinar(s)
    historico += hist
    pr = prever(model, DADOS['Test']['X'])
    r = dict(semente=s, f1_val=f1v, segundos_total=time.time() - t)
    for nome, sel in (('bal', bal), ('bal_DAiSEE', bal & (fonte == 'DAiSEE')),
                      ('bal_EngageNet', bal & (fonte == 'EngageNet')), ('natural', np.ones_like(bal))):
        r.update({f'{k}_{nome}': v for k, v in metricas(yte[sel], pr[sel]).items()})
    linhas.append(r); print(r, flush=True)
    if f1v > melhor_f1:
        melhor_f1, melhor_pr, melhor_s = f1v, pr, s
        torch.save({k: v.cpu() for k, v in model.state_dict().items()
                    if not k.startswith('enc.') or treinavel(k)}, f'/kaggle/working/modelo_{NOME}.pt')
    del model; torch.cuda.empty_cache()
df = pd.DataFrame(linhas)
df.to_csv(f'/kaggle/working/resultados_{NOME}.csv', index=False)
pd.DataFrame(historico).to_csv(f'/kaggle/working/historico_{NOME}.csv', index=False)
print(df.drop(columns='semente').agg(['mean', 'std']).T.round(3).to_string())

# %% Relatorio do melhor modelo (escolhido pela validacao)
pr = melhor_pr
print('Semente escolhida (pela validacao):', melhor_s)
for nome, sel in (('teste balanceado', bal), ('DAiSEE', bal & (fonte == 'DAiSEE')),
                  ('EngageNet', bal & (fonte == 'EngageNet')), ('teste completo (natural)', np.ones_like(bal))):
    print('\n==', nome, '(n=%d)' % sel.sum())
    print(classification_report(yte[sel], pr[sel].argmax(1), target_names=['Nao engajado', 'Engajado'], digits=3))
    print(confusion_matrix(yte[sel], pr[sel].argmax(1)))
    print('AUC:', round(roc_auc_score(yte[sel], pr[sel][:, 1]), 3))
pd.DataFrame(dict(fonte=fonte, pessoa=mte.pessoa, clipe=mte.clipe, y_orig=mte.y_orig, y=yte,
                  balanceado=bal, p_engajado=pr[:, 1])).to_csv(f'/kaggle/working/predicoes_{NOME}.csv', index=False)
