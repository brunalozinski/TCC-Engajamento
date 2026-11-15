# Classificação de níveis de engajamento em ambientes digitais de aprendizagem

Trabalho de Conclusão de Curso (Ciência da Computação, UNIP, 2026).

O projeto estima se um estudante está **engajado** ou **não engajado** a partir do rosto em vídeo. Cada vídeo é representado por três quadros. O rosto é recortado, transformado em um *embedding* por um modelo pré-treinado de visão computacional (DINOv2 ou Swin Transformer) e a sequência é classificada por uma **LSTM bidirecional**. Os dados vêm de dois conjuntos públicos: **DAiSEE** e **EngageNet**.

```
vídeo → 3 quadros → rosto (dlib) → recorte 224×224 → DINOv2 / Swin → 3 embeddings → LSTM → P(engajado)
```

## Experimentos

| Exp. | Dados | Classes | Extrator | Modo |
|---|---|---|---|---|
| A | DAiSEE | 4 | DINOv2 | congelado |
| B | EngageNet | 4 | DINOv2 | congelado |
| C | DAiSEE + EngageNet | 2 | DINOv2 | congelado |
| D | DAiSEE + EngageNet | 2 | Swin-tiny | congelado |
| E | DAiSEE + EngageNet | 2 | DINOv2 | ajuste fino parcial |
| F | DAiSEE + EngageNet | 2 | Swin-tiny | ajuste fino parcial |

Resultados binários no teste balanceado (1.564 vídeos, média de cinco sementes):

| Exp. | F1 macro | AUC |
|---|---|---|
| C | 0,740 | 0,813 |
| E | 0,753 | 0,826 |
| D | 0,612 | 0,701 |
| F | 0,669 | 0,734 |
| Regressão logística (referência) | 0,703 | 0,786 |

A diferença entre C e E está dentro da variação entre sementes. A escolha do extrator (DINOv2 ou Swin) teve mais efeito que o ajuste fino parcial. O acaso é 0,50.

## Demo (Streamlit)

A demo usa a webcam e fecha a decisão a cada **janela de 20 segundos**. O resultado é a probabilidade de engajamento, apresentada em três zonas (Baixo, Indefinido e Alto), calibradas na validação. A página permite escolher o modelo do experimento C ou E e mostra um resumo da sessão com download em CSV. As métricas do teste foram obtidas em clipes de cerca de 10 segundos e **não se aplicam diretamente** à demo.

### Requisitos

- Python 3.11 (64 bits)
- Webcam
- Visual Studio Build Tools com "Desenvolvimento para desktop com C++" (apenas para compilar o `dlib` no Windows)
- Modelo de 68 pontos do dlib, que **não está neste repositório** (99,7 MB): baixe [shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2), extraia e coloque o `.dat` na **raiz do projeto**.

### Instalação e execução

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

A aplicação abre em `http://localhost:8501`. Na primeira execução o DINOv2 (`facebook/dinov2-base`, cerca de 350 MB) é baixado automaticamente. Em CPU, cada embedding leva cerca de 0,3 s. Se houver GPU, ela é usada.

## Estrutura

```
app.py                          menu da aplicação
paginas/
  classificar_dinov2.py         classificação em tempo real
  classificar.py                constantes da janela e da detecção de olhos fechados
  resultados.py                 página de resultados
motor_dinov2_lstm.py            motores dos experimentos C e E
features.py                     detector dlib e cálculo do EAR
calibrar_zonas.py               calibração das zonas Baixo / Indefinido / Alto
modelo_dinov2_lstm/             modelo do experimento C (lstm.pt, normalizacao.npz, zonas.json)
modelo_dinov2_finetune/         modelo do experimento E (modelo.pt, zonas.json)
fine tuning/modelos/            detector de reserva (BlazeFace)
TCC/
  face_crop.py                  recorte do rosto
  daisee_sequencias.py          extração de quadros do DAiSEE
  engagenet_sequencias.py       extração de quadros do EngageNet
  tabela_consolidada_experimentos_binarios.csv
  figuras_tcc/                  gráficos, matrizes de confusão e curvas ROC
  kaggle/                       scripts de treino e resultados por experimento (saida_*)
```

## Treinamento

O pré-processamento foi feito localmente (`TCC/face_crop.py`, `TCC/daisee_sequencias.py` e `TCC/engagenet_sequencias.py`). A geração de embeddings e o treino foram feitos no Kaggle, com GPU NVIDIA Tesla T4. Os scripts estão em `TCC/kaggle/` e os resultados de cada experimento, em `TCC/kaggle/saida_*`.

Os conjuntos DAiSEE e EngageNet **não estão** neste repositório e devem ser obtidos pelos canais oficiais, respeitando seus termos de uso.

## Limitações

- O modelo distingue apenas engajado de não engajado, e a saída é uma estimativa feita a partir do rosto, não o estado real da pessoa.
- Foram usados três quadros por vídeo, o que limita a informação temporal.
- Dos vídeos de treino do EngageNet, apenas parte estava disponível localmente.
- As zonas foram calibradas em clipes de cerca de 10 s, e não com a demo ao vivo.
- A demo não passou por avaliação formal com usuários.

## Autores

Beatriz Santos Sousa, Bruna Lozinski, Haniel Ribeiro e Yasmin Sampaio. Orientador: Prof. Edy Hayashida.
