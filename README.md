# Classificação de Engajamento a partir de Vídeo Facial
### DINOv2 / Swin-tiny + LSTM, com DAiSEE e EngageNet

> Trabalho de Conclusão de Curso — Ciência da Computação  
> Universidade Paulista – UNIP | 2026

**Autores:** Beatriz Santos Sousa · Bruna Lozinski · Haniel Ribeiro · Yasmin Sampaio  
**Orientador:** Prof. Edy Hayashida

---

## Sobre o projeto

O trabalho avalia se é possível estimar, a partir de vídeos do rosto, se uma pessoa está **engajada** ou **não engajada** durante uma atividade em frente a uma tela, em **contexto educacional**.

Foram comparados modelos de visão pré-treinados (**DINOv2** e **Swin-tiny**), congelados ou com ajuste fino parcial, ligados a uma rede **LSTM** que lê a sequência de rostos do vídeo. Os dados vêm de dois datasets públicos: **DAiSEE** e **EngageNet**. O projeto também inclui uma **demo em Streamlit** que usa a webcam.

---

## Pipeline

```
vídeo → frames → detecção do rosto (dlib) → recorte 224×224 (margem de 20%)
      → encoder (DINOv2-base: 1.536 números | Swin-tiny: 768 números)
      → LSTM bidirecional (128 unidades, dropout 0,4)
      → P(engajado)
```

- **Tarefa:** classificação binária. *Não engajado* = níveis originais 0 e 1; *engajado* = níveis 2 e 3.
- **Frames por vídeo:** 3 nos experimentos C a F, distribuídos ao longo da duração do vídeo.
- **Divisões:** as divisões oficiais de treino, validação e teste de cada dataset, separadas por pessoa.
- **Sementes:** 5 por experimento; os resultados são média ± desvio padrão.
- **Teste balanceado:** 1.564 vídeos (782 por classe).

---

## Experimentos

| Exp. | Dataset | Método | Tarefa |
|---|---|---|---|
| A | DAiSEE | referência com 4 classes | 4 classes |
| B | EngageNet | referência com 4 classes | 4 classes |
| **C** | DAiSEE + EngageNet | **DINOv2 congelado** + LSTM | engajado × não engajado |
| **D** | DAiSEE + EngageNet | **Swin-tiny congelado** + LSTM | engajado × não engajado |
| **E** | DAiSEE + EngageNet | **DINOv2 com ajuste fino parcial** (últimos 2 de 12 blocos + layernorm) + LSTM | engajado × não engajado |
| **F** | DAiSEE + EngageNet | **Swin-tiny com ajuste fino parcial** (último estágio + layernorm) + LSTM | engajado × não engajado |

Os experimentos A e B usam 4 classes e outro conjunto de teste, então **não são comparáveis diretamente** com C a F.

Detalhes do ajuste fino (E e F): taxa de aprendizado de 2e-5 no encoder e 1e-3 na cabeça, até 8 épocas com parada antecipada (paciência 3) e aumento de dados.

### Parâmetros

| Modelo | Parâmetros do encoder | Treinável no ajuste fino |
|---|---|---|
| DINOv2-base | ~86,6 milhões | ~14,2 milhões (últimos 2 blocos + layernorm) |
| Swin-tiny | ~27,5 milhões | ~14,2 milhões (último estágio + layernorm) |
| LSTM (cabeça) | ~1,7 milhão (entrada 1.536) ou ~0,9 milhão (entrada 768) | treinado em todos |

---

## Resultados

Teste balanceado (1.564 vídeos), média ± desvio de 5 sementes:

| Exp. | Método | F1 macro | Acurácia balanceada | AUC |
|---|---|---|---|---|
| C | DINOv2 congelado + LSTM | 0,740 ± 0,013 | 0,741 ± 0,013 | 0,813 ± 0,006 |
| E | DINOv2 ajuste fino parcial + LSTM | 0,753 ± 0,007 | 0,753 ± 0,008 | 0,826 ± 0,022 |
| D | Swin-tiny congelado + LSTM | 0,612 ± 0,021 | 0,613 ± 0,020 | 0,701 ± 0,024 |
| F | Swin-tiny ajuste fino parcial + LSTM | 0,669 ± 0,029 | 0,669 ± 0,029 | 0,734 ± 0,027 |

Referências com 4 classes (média de 5 sementes, F1 macro): **A** (DAiSEE) 0,320 e **B** (EngageNet) 0,484.

**Leitura dos resultados**
- C e E foram os melhores. A diferença entre eles (+0,013 de F1 e +0,013 de AUC) é do tamanho da variação entre sementes, então **não é um ganho comprovado**.
- A escolha do modelo (DINOv2 em vez de Swin-tiny) pesou mais do que o ajuste fino.
- O Swin-tiny melhorou com o ajuste fino (F1 0,612 → 0,669), mas continuou abaixo do DINOv2.
- O teste só do DAiSEE (176 vídeos, 88 por classe) é mais incerto e teve resultado pior que o do EngageNet.
- No teste completo, de distribuição natural (4.030 vídeos, 81% engajados), o F1 macro cai (C 0,665; E 0,681) e o AUC se mantém (C 0,801; E 0,826).

As tabelas por fonte de dados, as matrizes de confusão, as curvas ROC e o relatório por classe estão em [`TCC/`](TCC/):

- `tabela_consolidada_experimentos_binarios.csv`
- `figuras_tcc/` (matrizes de confusão, curvas ROC, barras de F1 e AUC e `relatorio_por_classe.csv`)

---

## Demo em tempo real (Streamlit)

A demo abre a webcam, classifica o engajamento em **janelas de 20 segundos** e mostra:

- a **probabilidade de engajamento** (P(engajado)) com uma barra de zonas;
- uma **zona de leitura**: *Baixo*, *Indefinido* (o modelo está em dúvida) ou *Alto*;
- a **qualidade da janela** (embeddings, % de frames com rosto e % de olhos fechados), com aviso de "pouco confiável";
- o **gráfico por janela** e o **relatório da sessão**, com resumo e tendência, que pode ser baixado em CSV e TXT.

Na página é possível escolher entre o **experimento C** e o **experimento E**.

O modelo só distingue engajado de não engajado. As zonas são uma leitura da probabilidade, não níveis aprendidos. Os cortes foram **calibrados na validação** (790 vídeos balanceados, pureza mínima de 80% nas zonas extremas) e avaliados no teste:

| | Baixo se P ≤ | Alto se P ≥ | Pureza no teste balanceado (Baixo / Alto) |
|---|---|---|---|
| Experimento C | 13% | 78% | 90,7% / 79,6% |
| Experimento E | 22% | 60% | 86,7% / 81,5% |

**Privacidade:** a demo processa o rosto localmente, em tempo real, e não grava quadros, imagens nem vídeo em disco. Só é gravado o que o usuário baixa (o CSV e o TXT do relatório, que têm números, sem imagens).

---

## Como executar

### Pré-requisitos

1. **Python 3.11** (64 bits).
2. **Visual Studio Build Tools** com o componente "Desenvolvimento para desktop com C++", necessário apenas para compilar o `dlib`.
3. **Modelo de 68 pontos do dlib** (`shape_predictor_68_face_landmarks.dat`), usado na detecção do rosto. [Download](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2); extraia o `.dat` na raiz do projeto.
4. Uma **webcam**.

### Instalação

```bash
pip install -r requirements.txt
```

Na primeira execução o DINOv2 é baixado automaticamente (cerca de 350 MB). A demo roda em CPU.

### Rodar a demo

```bash
python -m streamlit run app.py
```

A aplicação abre em `http://localhost:8501`, com três páginas: **Início**, **Classificar em Tempo Real (DINOv2 + LSTM)** e **Resultados do Treinamento**.

---

## Estrutura do repositório

```
app.py                         # menu da aplicação Streamlit
paginas/
  classificar_dinov2.py        # demo em tempo real (experimentos C e E)
  classificar.py               # constantes da janela e da detecção de olhos fechados
  resultados.py                # comparação dos experimentos
motor_dinov2_lstm.py           # motor de inferência (DINOv2 + LSTM)
features.py                    # detecção do rosto e dos olhos fechados
calibrar_zonas.py              # calibração das zonas na validação
modelo_dinov2_lstm/            # modelo do experimento C (+ zonas.json)
modelo_dinov2_finetune/        # modelo do experimento E (+ zonas.json)
fine tuning/modelos/           # detector de reserva (BlazeFace)
TCC/
  face_crop.py                 # recorte do rosto
  daisee_sequencias.py         # extração de quadros do DAiSEE
  engagenet_sequencias.py      # extração de quadros do EngageNet
  kaggle/                      # scripts dos experimentos (treino no Kaggle, GPU T4) e resultados por experimento
  tabela_consolidada_experimentos_binarios.csv
  figuras_tcc/                 # matrizes, ROC, barras, relatório por classe
requirements.txt
```

Os datasets **não** estão versionados: DAiSEE e EngageNet têm licença e termos próprios e contêm rostos de pessoas. Para reproduzir, obtenha-os nas fontes oficiais. O EngageNet (7.983 vídeos de treino, 1.071 de validação e 2.257 de teste, 127 participantes) é liberado mediante pedido por e-mail aos autores, conforme o [repositório oficial](https://github.com/engagenet/engagenet_baselines).

---

## Limitações

- Cada vídeo é representado por apenas **3 frames**, o que limita a informação temporal.
- No DAiSEE, as classes de menor engajamento são muito raras: no teste binário há só 88 vídeos de "não engajado".
- Os resultados são de clipes de ~10 s; as janelas de 20 s da demo e a webcam são condições diferentes.
- A diferença entre C e E está dentro da variação entre sementes.
- A demo estima o engajamento a partir do rosto; não mede o estado real da pessoa.

---

## Referências

- GUPTA, A. et al. DAiSEE: Towards User Engagement Recognition in the Wild. *arXiv*, 2016.
- SINGH, M. et al. Do I Have Your Attention: A Large Scale Engagement Prediction Dataset and Baselines. *ICMI*, 2023. Código e baselines: <https://github.com/engagenet/engagenet_baselines>. O acesso ao dataset é solicitado por e-mail aos autores.
- OQUAB, M. et al. DINOv2: Learning Robust Visual Features without Supervision. *arXiv*, 2023.
- LIU, Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*, 2021.

---

## Conformidade legal

O projeto usa datasets públicos de terceiros e processa a webcam localmente, sem gravar imagens. Ao usar a demo com outras pessoas, obtenha o consentimento delas, conforme a **LGPD (Lei nº 13.709/2018)**.
