# %% [markdown]
# # DAiSEE + EngageNet, 2 classes (nao engajado x engajado), classes iguais
# Entrada: dataset Kaggle `daisee-engagenet-binario` (embeddings DINOv2 ja calculados,
# 3 frames por video, gerado por combinar_binario.py). Nao engajado = originais 0 e 1,
# engajado = originais 2 e 3. Train/Validation/Test oficiais, classes com o mesmo
# numero de videos e a mesma mistura de fontes nas duas classes.

# %% Dados
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression

ENTRADA = next(p for p in Path('/kaggle/input').rglob('Train.npz')).parent
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NOME = ENTRADA.name
D = {div: dict(np.load(ENTRADA / f'{div}.npz', allow_pickle=True)) for div in ('Train', 'Validation', 'Test')}
for div, d in D.items():
    d['X'] = d['X'].astype(np.float32)
    print(div, d['X'].shape, 'balanceados:', int(d['balanceado'].sum()))
    print(pd.crosstab(d['fonte'][d['balanceado']], d['y'][d['balanceado']]))

Xtr, ytr = D['Train']['X'], D['Train']['y'].astype(int)
Xva, yva = D['Validation']['X'], D['Validation']['y'].astype(int)
mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
norm = lambda a: (a - mu) / sd
Xtr, Xva = norm(Xtr), norm(Xva)
T = D['Test']; Xte = norm(T['X']); yte = T['y'].astype(int)
bal, fonte = T['balanceado'].astype(bool), T['fonte'].astype(str)
NC = 2

# %% Modelo
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

def prever(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.tensor(X[i:i + 512], device=dev)).softmax(-1).cpu()
                          for i in range(0, len(X), 512)]).numpy()

def treinar(epocas=60, paciencia=12, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModeloLSTM(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    melhor, estado, sem_ganho = -1, None, 0
    for ep in range(epocas):
        model.train()
        idx = np.random.permutation(len(ytr))
        for i in range(0, len(idx), 64):
            b = torch.tensor(idx[i:i + 64], device=dev)
            x = Xt[b] + 0.1 * torch.randn_like(Xt[b])
            x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=dev) > 0.1)
            loss = F.cross_entropy(model(x), yt[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = f1_score(yva, prever(model, Xva).argmax(1), average='macro')
        if f1 > melhor:
            melhor, sem_ganho = f1, 0
            estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sem_ganho += 1
            if sem_ganho >= paciencia: break
    model.load_state_dict(estado)
    return model, melhor

def metricas(y, pr):
    p = pr.argmax(1)
    return dict(f1=f1_score(y, p, average='macro'), bacc=balanced_accuracy_score(y, p),
                acc=(p == y).mean(), auc=roc_auc_score(y, pr[:, 1]))

# %% 5 sementes: teste balanceado (principal), por fonte e teste completo (distribuicao natural)
linhas, melhor_modelo, melhor_f1 = [], None, -1
for s in range(5):
    m, f1v = treinar(seed=s)
    pr = prever(m, Xte)
    r = dict(semente=s, f1_val=f1v)
    for nome, sel in (('bal', bal), ('bal_DAiSEE', bal & (fonte == 'DAiSEE')),
                      ('bal_EngageNet', bal & (fonte == 'EngageNet')), ('natural', np.ones_like(bal))):
        r.update({f'{k}_{nome}': v for k, v in metricas(yte[sel], pr[sel]).items()})
    linhas.append(r)
    if f1v > melhor_f1:
        melhor_f1, melhor_modelo, melhor_s = f1v, m, s
    print('semente', s, 'ok', flush=True)
df = pd.DataFrame(linhas)
df.to_csv(f'/kaggle/working/resultados_{NOME}.csv', index=False)
res = df.drop(columns='semente').agg(['mean', 'std']).T.round(3)
print(res.to_string())

# %% Referencia simples: regressao logistica nos embeddings (media dos frames)
lr = LogisticRegression(max_iter=2000, C=0.01).fit(Xtr.mean(1), ytr)
pl = lr.predict_proba(Xte.mean(1))
print('Regressao logistica (teste balanceado):', {k: round(v, 3) for k, v in metricas(yte[bal], pl[bal]).items()})

# %% Relatorio do melhor modelo (escolhido pela validacao)
pr = prever(melhor_modelo, Xte)
print('Melhor semente (pela validacao):', melhor_s)
for nome, sel in (('teste balanceado', bal), ('DAiSEE', bal & (fonte == 'DAiSEE')),
                  ('EngageNet', bal & (fonte == 'EngageNet')), ('teste completo (natural)', np.ones_like(bal))):
    print('\n==', nome, '(n=%d)' % sel.sum())
    print(classification_report(yte[sel], pr[sel].argmax(1), target_names=['Nao engajado', 'Engajado'], digits=3))
    print(confusion_matrix(yte[sel], pr[sel].argmax(1)))
    print('AUC:', round(roc_auc_score(yte[sel], pr[sel][:, 1]), 3))
pd.DataFrame(dict(fonte=fonte, pessoa=T['pessoa'], clipe=T['clipe'], y_orig=T['y_orig'], y=yte,
                  balanceado=bal, p_engajado=pr[:, 1])).to_csv(f'/kaggle/working/predicoes_{NOME}.csv', index=False)
torch.save(melhor_modelo.state_dict(), f'/kaggle/working/lstm_{NOME}.pt')
