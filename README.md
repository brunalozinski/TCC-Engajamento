# Classificação de Níveis de Engajamento em Ambientes Digitais
### Utilizando Técnicas de Machine Learning

> Trabalho de Conclusão de Curso — Ciência da Computação  
> Universidade Paulista – UNIP | 2026

**Autores:** Beatriz Santos Sousa · Bruna Lozinski · Haniel Ribeiro · Yasmin Sampaio

---

## Sobre o Projeto

Este projeto desenvolve e avalia modelos supervisionados de Machine Learning para classificação automática de **níveis de engajamento** em ambientes corporativos digitais simulados, utilizando exclusivamente indicadores visuais extraídos de imagens — expressões faciais, pontos de referência facial e pose da cabeça.

---

## Ambiente de Desenvolvimento

| Componente | Especificação |
|---|---|
| Sistema Operacional | Windows 11  |
| Processador | Intel Core i5-1135G7 11ª Gen @ 2.40GHz |
| Memória RAM | 8 GB |
| Placa de Vídeo | Intel Iris Xe Graphics (integrada) |
| Python | 3.11.9 (64-bit) |

---

## Estrutura do Projeto

```
C:/TCC/
│
├── shape_predictor_68_face_landmarks.dat   # modelo pré-treinado dlib (68 pontos)
│
├── archive/
│   └── Student-engagement-dataset/         # dataset original do Kaggle
│       ├── Engaged/
│       │   ├── Confused/
│       │   ├── Focused/
│       │   └── Frustrated/
│       └── Not Engaged/
│           ├── Bored/
│           ├── Drowsy/
│           └── Looking Away/
│
├── ER_Dataset/                             # dataset reorganizado em 4 níveis
│   ├── 0/                                  # Muito Baixo (686 imagens)
│   ├── 1/                                  # Baixo       (358 imagens)
│   ├── 2/                                  # Alto        (729 imagens)
│   └── 3/                                  # Muito Alto  (347 imagens)
│
├── organizar_dataset.py                    # reorganiza o dataset em 4 níveis
├── treinar.py                              # extrai features e treina os modelos
├── mcnemar.py                              # teste estatístico de McNemar
│
├── model_SVM.pkl                           # modelo SVM treinado
├── model_Random_Forest.pkl                 # modelo Random Forest treinado
├── model_MLP.pkl                           # modelo MLP treinado
│
├── X_test.npy                              # features do conjunto de teste
├── y_test.npy                              # labels do conjunto de teste
│
├── confusion_SVM.png                       # matriz de confusão SVM
├── confusion_Random_Forest.png             # matriz de confusão Random Forest
├── confusion_MLP.png                       # matriz de confusão MLP
│
└── requirements.txt                        # dependências do projeto
```

---

## Instalação

### Pré-requisitos

1. **Python 3.11.9** — [download](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)  
   ⚠️ Marcar **"Add Python to PATH"** durante a instalação

2. **Visual Studio Build Tools** — [download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)  
   ⚠️ Selecionar o componente **"Desenvolvimento para desktop com C++"**  
   Necessário para compilar o dlib (não significa que o projeto usa C++)

3. **Modelo de 68 pontos do dlib** — [download](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)  
   Extrair o `.bz2` com 7-Zip e colocar o `.dat` em `C:\TCC\`

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

---

## Dataset

**Student Engagement Dataset** — disponível no [Kaggle](https://www.kaggle.com/datasets/deepramazumder/student-engagement-dataset)

### Remapeamento para 4 Níveis

O dataset original usa classificação binária (Engaged / Not Engaged). Foi remapeado para 4 níveis com base na literatura (D'Mello e Graesser, 2012; Dewan, Murshed e Lin, 2019):

| Nível | Nome | Subcategorias Originais | Imagens |
|---|---|---|---|
| 0 | Muito Baixo | Drowsy + Looking Away | 686 |
| 1 | Baixo | Bored | 358 |
| 2 | Alto | Confused + Frustrated | 729 |
| 3 | Muito Alto | Focused | 347 |
| | **Total** | | **2.120** |

---

## Execução

### 1. Organizar o Dataset
```bash
python C:\TCC\organizar_dataset.py
```
Copia as imagens do dataset original para pastas numeradas (0, 1, 2, 3) em `C:\TCC\ER_Dataset\`.

### 2. Treinar os Modelos
```bash
python C:\TCC\treinar.py
```
- Carrega as imagens do `ER_Dataset`
- Detecta rostos e extrai 68 pontos faciais (dlib)
- Gera vetor de 136 features por imagem
- Treina SVM, Random Forest e MLP
- Salva os modelos `.pkl` e as matrizes de confusão `.png`

### 3. Teste de McNemar
```bash
python C:\TCC\mcnemar.py
```
Compara estatisticamente os três modelos para verificar se as diferenças de desempenho são significativas.

---

## Resultados

### Desempenho dos Modelos

| Modelo | Acurácia | Precisão Macro | Recall Macro | F1 Macro |
|---|---|---|---|---|
| SVM | 79% | 76% | 77% | 76% |
| Random Forest | **95%** | **94%** | **93%** | **94%** |
| MLP | 94% | 94% | 92% | 93% |

### F1-Score por Classe

| Classe | SVM | Random Forest | MLP |
|---|---|---|---|
| 0 — Muito Baixo | 85% | 92% | 92% |
| 1 — Baixo | 57% | 83% | 83% |
| 2 — Alto | 83% | 100% | 99% |
| 3 — Muito Alto | 81% | 99% | 99% |

> **Observação:** A classe Baixo (nível 1) apresentou maior dificuldade em todos os modelos, consistente com o reportado na literatura para classes intermediárias de engajamento.

### Teste de McNemar

| Comparação | chi² | p-valor | Resultado |
|---|---|---|---|
| SVM vs Random Forest | 53.1571 | 0.0000 | Diferença **SIGNIFICATIVA** |
| SVM vs MLP | 54.3906 | 0.0000 | Diferença **SIGNIFICATIVA** |
| Random Forest vs MLP | 0.0625 | 0.8026 | Diferença **NÃO significativa** |

**Conclusão:** O SVM é estatisticamente inferior aos demais. Random Forest e MLP são estatisticamente equivalentes — a diferença de 1% entre eles pode ter ocorrido por acaso.

---

## Pipeline Técnico

```
Imagem
  → Detectar rosto (dlib HOG+SVM)
  → Extrair 68 pontos faciais (landmarks)
  → Normalizar coordenadas (÷ largura do rosto)
  → Vetor de 136 valores [x1,y1, x2,y2, ..., x68,y68]
  → StandardScaler (padroniza a escala)
  → Modelo (SVM / Random Forest / MLP)
  → Predição: 0, 1, 2 ou 3
```

---

## Dependências Principais

```
dlib==20.0.1
opencv-python==4.13.0.92
scikit-learn==1.8.0
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.9
seaborn==0.13.2
mlxtend==0.24.0
mediapipe==0.10.35
joblib==1.5.3
tqdm==4.67.3
```

---

## Próximos Passos

- [ ] Capturar imagens dos 4 integrantes (~800 fotos) simulando os 4 níveis
- [ ] Retreinar os modelos com o dataset combinado
- [ ] Avaliar comparativamente os resultados antes e depois
- [ ] Implementar demo em tempo real via webcam
- [ ] Discussão e escrita final do TCC

---

## Referências

- D'MELLO, S. K.; GRAESSER, A. C. Dynamics of affective states during complex learning. *Learning and Instruction*, 2012.
- DEWAN, M. A. A.; MURSHED, M.; LIN, F. Engagement detection in online learning: a review. *Smart Learning Environments*, 2019.
- GUPTA, A. et al. DAiSEE: Towards User Engagement Recognition in the Wild. *arXiv*, 2016.
- GÉRON, A. *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly, 2019.
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016.

---

## Conformidade Legal

Este projeto utiliza exclusivamente datasets públicos, sem coleta direta de dados de participantes reais, garantindo conformidade com a **Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018)**.