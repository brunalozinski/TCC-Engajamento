# BASE DE CONHECIMENTO — TCC "Classificação de Níveis de Engajamento em Ambientes Digitais"

Documento de handoff consolidado a partir do arquivo do projeto e de conversas anteriores (chat "Passo a passo desenvolvimento de modelos" e chat "Identificando características de conteúdo gerado por IA"). Use este arquivo para continuar o trabalho em outra conta sem perder contexto.

---

## 1. Identificação do projeto

- **Tema:** Classificação de níveis de engajamento em ambientes corporativos digitais simulados, usando indicadores visuais (landmarks faciais) e Machine Learning supervisionado.
- **Instituição:** Universidade Paulista (UNIP) — Santana de Parnaíba, SP — 2026.
- **Curso:** Bacharelado em Ciência da Computação (TCC).
- **Autores:** Beatriz Santos Sousa (G810414), Bruna Lozinski (G80BCG6), Haniel Ribeiro (N0904A2), Yasmin Bonfim (N0740A8).
- **Orientador:** campo em branco — **pendente de preenchimento**.
- **Repositório GitHub:** https://github.com/brunalozinski/TCC-Engajamento (raiz do repo = `C:\TCC\` na máquina local, não uma subpasta).

---

## 2. Decisão crítica que mudou o rumo técnico: troca de dataset

O TCC foi **redigido** (capítulos 1–3) planejando usar o **DAiSEE** (Gupta et al., 2016), dataset de vídeos, com pipeline de extração de frames a 5 fps + HOG/LBP + 68 landmarks.

Na prática, isso **não foi o que aconteceu**:

1. O ZIP do DAiSEE baixado deu erro `Bad CRC-32` (arquivo corrompido) e não pôde ser extraído.
2. Foi testado o repositório `omidmnezami/Engagement-Recognition` (GitHub) — continha só dados de exemplo (4 amostras de uma classe), não o dataset real.
3. **Dataset efetivamente usado:** "Student Engagement Dataset" do Kaggle (`kaggle.com/datasets/deepramazumder/student-engagement-dataset`) — **imagens estáticas**, organizadas em pastas `Engaged/` e `Not Engaged/`, cada uma com subpastas por categoria.
4. Reorganização em 4 níveis (script `organizar_dataset.py`), salva em `ER_Dataset/`:
   - **Nível 0 – Muito Baixo:** Not Engaged/Drowsy + Not Engaged/Looking Away → 686 imagens
   - **Nível 1 – Baixo:** Not Engaged/Bored → 358 imagens
   - **Nível 2 – Alto:** Engaged/Confused + Engaged/Frustrated → 729 imagens
   - **Nível 3 – Muito Alto:** Engaged/Focused → 347 imagens
   - **Total: 2.120 imagens**, divisão 80% treino / 20% teste.

⚠️ **Consequência importante:** o pipeline real não faz extração de frames (são imagens diretas) e o vetor de características usado no treinamento final é só de **136 valores geométricos** (coordenadas x,y normalizadas dos 68 landmarks) — **sem HOG/LBP**, diferente do que o Capítulo 3 do documento atual descreve.

---

> ⚠️ **Atualização 03/09/2026:** o pipeline de extração de características foi corrigido e os modelos foram retreinados. Os números desta seção 3 e da seção 4 abaixo são os da **primeira** versão. Para os dados atuais (usar no TCC), ver a **seção 9 — Atualização de setembro/2026**.

## 3. Pipeline técnico efetivamente implementado

- **Detecção facial:** `dlib.get_frontal_face_detector()` (HOG+SVM).
- **Landmarks:** modelo `shape_predictor_68_face_landmarks.dat` (~95 MB, baixado de dlib.net, mantido fora do Git por tamanho).
- **Vetor de características:** 136 valores (68 pontos × x,y), normalizados pela largura do rosto detectado.
- **Divisão treino/teste:** `train_test_split`, 80/20, `stratify=y`, `random_state=42`.
- **Modelos treinados (scikit-learn):**
  - SVM — kernel RBF, `class_weight="balanced"`, `probability=True`, dentro de `Pipeline` com `StandardScaler`.
  - Random Forest.
  - MLP (Multilayer Perceptron).
- **Scripts no projeto:**
  - `organizar_dataset.py` — reorganiza o dataset baixado do Kaggle nos 4 níveis.
  - `treinar.py` — extrai features, treina os 3 modelos, salva `.pkl` e gera matrizes de confusão `.png`.
  - `mcnemar.py` — carrega modelos/dados de teste salvos, gera predições e aplica teste de McNemar par a par.
- **Artefatos gerados** (mantidos só localmente, fora do Git por tamanho): `model_SVM.pkl`, `model_Random_Forest.pkl`, `model_MLP.pkl`, `X_test.npy`, `y_test.npy`, `confusion_SVM.png`, `confusion_Random_Forest.png`, `confusion_MLP.png`.
- Ambiente: Python 3.10 (obrigatório por causa do dlib), Windows, pasta `C:\TCC\`.

---

## 4. Resultados já obtidos (PRIMEIRA versão — ver seção 9 para os atuais)

**Desempenho geral** (396 amostras de teste):

| Modelo | Acurácia | Precisão Macro | Recall Macro | F1 Macro |
|---|---|---|---|---|
| SVM | 79% | 76% | 77% | 76% |
| Random Forest | **95%** | 94% | 93% | 94% |
| MLP | 94% | 94% | 92% | 93% |

**F1-Score por nível de engajamento:**

| Classe | SVM | Random Forest | MLP |
|---|---|---|---|
| 0 — Muito Baixo | 85% | 92% | 92% |
| 1 — Baixo | 57% | 83% | 83% |
| 2 — Alto | 83% | 100% | 99% |
| 3 — Muito Alto | 81% | 99% | 99% |

**Teste de McNemar:**

| Comparação | chi² | p-valor | Conclusão |
|---|---|---|---|
| SVM vs Random Forest | 53,1571 | 0,0000 | Diferença significativa |
| SVM vs MLP | 54,3906 | 0,0000 | Diferença significativa |
| Random Forest vs MLP | 0,0625 | 0,8026 | Diferença NÃO significativa |

**Principais achados (já discutidos, reaproveitáveis no texto):**
- Random Forest e MLP são estatisticamente equivalentes; SVM é significativamente inferior.
- O nível "Baixo" é o mais difícil de classificar em todos os modelos — confunde-se visualmente com desengajamento total e engajamento passivo. Consistente com Dewan, Murshed e Lin (2019).
- Classes extremas (Muito Baixo, Alto, Muito Alto) são classificadas quase perfeitamente por RF e MLP.

Também já foram redigidos em conversa (mas **não confirmados como inseridos no .docx real**):
- Nova versão da seção 3.2.3 (pré-processamento sem extração de frames).
- Capítulo 4 completo (Resultados) com subseções: estrutura do projeto/scripts, desempenho geral, análise por nível, matrizes de confusão, teste de McNemar.
- Capítulo 5 (Discussão).
- Capítulo 6 (Conclusão e Trabalhos Futuros).

---

## 5. Estado atual do arquivo .docx do projeto

Arquivo de referência: `CLASSIFICAÇÃO_DE_NÍVEIS_DE_ENGAJAMENTO_CORRIGIDO.docx` (versão mais recente disponível no projeto).

**O que já está pronto:**
- Estrutura pré-textual completa: capa, folha de rosto, folha de aprovação, resumo/abstract, lista de abreviaturas e siglas, sumário.
- Capítulos 1 (Introdução), 2 (Revisão da Literatura) e 3 (Materiais e Métodos, até 3.2.5) completos.
- Referências bibliográficas em formato ABNT, completas.
- Já passou por uma revisão específica para reduzir "marcas de texto gerado por IA" (linguagem repetitiva/formulaica, verbos ajustados para futuro, referências fora de escopo removidas) — revisão registrada em 21/04/2026.

**O que ainda está como placeholder/pendente:**
- Campo **"Orientador:"** em branco.
- **Dedicatória, Agradecimentos e Epígrafe** — marcados como "[a ser preenchido pelos autores]".
- **Banca examinadora** com nomes fictícios ("Prof. Xxxxx Xxxxxxxxx") — precisa dos nomes reais.
- **Lista de Figuras e Lista de Tabelas** — vazias, pois ainda não há figuras/tabelas inseridas no corpo do texto.
- **Capítulos 4 (Resultados), 5 (Discussão) e 6 (Conclusão e Trabalhos Futuros)** — ainda não existem no documento.
- **Apêndices com o código-fonte** — não existem.

**⚠️ Inconsistência a resolver:**
O Capítulo 3 (metodologia) ainda descreve o pipeline **planejado originalmente** (vídeos DAiSEE, extração de frames a 5 fps, HOG+LBP) — isso está **desatualizado** em relação ao que foi de fato implementado (imagens estáticas do Kaggle, sem extração de frames, sem HOG/LBP, apenas 136 landmarks geométricos). O texto precisa ser revisado para bater com a implementação real, ou então o documento precisa deixar claro — de forma metodologicamente honesta — que houve uma mudança de dataset/pipeline durante a execução, com justificativa (o esboço de resposta a "por que não usar o DAiSEE original" já existe em conversa anterior).

---

## 6. Lista de pendências (o que falta fazer)

1. **Atualizar o Capítulo 3** para refletir o pipeline realmente usado (dataset Kaggle, sem extração de frames, sem HOG/LBP, apenas landmarks geométricos), com justificativa da troca de dataset.
2. **Escrever e inserir o Capítulo 4 (Resultados)** com as tabelas de desempenho, análise por classe e as 3 matrizes de confusão como figuras (`confusion_SVM.png`, `confusion_Random_Forest.png`, `confusion_MLP.png`).
3. **Escrever e inserir o Capítulo 5 (Discussão)**.
4. **Escrever e inserir o Capítulo 6 (Conclusão e Trabalhos Futuros)**.
5. **Adicionar apêndices** com o código-fonte completo (`organizar_dataset.py`, `treinar.py`, `mcnemar.py`).
6. **Preencher Lista de Figuras e Lista de Tabelas** depois de inserir as figuras/tabelas no corpo do texto.
7. **Capturar as fotos dos 4 integrantes** (script `capturar_imagens.py` já existe: ~50 fotos × 4 níveis × 4 integrantes = ~800 fotos) e decidir se serão incorporadas ao dataset para um retreinamento comparativo (antes/depois).
8. ✅ **Feito (03/09/2026)** — demo em tempo real via webcam (Streamlit) usando o modelo treinado, com janela temporal e exportação de relatório. Ver seção 9.7.
9. **Preencher dados pessoais/institucionais pendentes:** nome do orientador, dedicatória, agradecimentos, epígrafe, nomes reais da banca examinadora.
10. **Revisar resumo/abstract e introdução** para garantir coerência entre o dataset citado (decidir entre manter DAiSEE como "dataset originalmente planejado" com nota metodológica sobre a substituição, ou reescrever citando o Student Engagement Dataset como definitivo) e os resultados reais obtidos.
11. **Conferir se o texto do resumo/abstract não afirma indevidamente** que um modelo é superior a outro sem checar contra o teste de McNemar (Random Forest teve acurácia numericamente maior, mas é estatisticamente equivalente ao MLP — não usar frases que apontem o MLP como "o melhor" sem qualificar).
12. **Finalizar o README.md** do repositório GitHub e confirmar quais arquivos grandes devem permanecer fora do controle de versão via `.gitignore` (`.pkl`, `.npy`, `.dat`, pastas de dataset).

---

## 7. Arquivos e caminhos relevantes (ambiente local do Windows)

```
C:\TCC\                              ← raiz do repositório Git
├── archive\                         ← dataset original do Kaggle (não versionado)
├── doc\                             ← documentos do TCC
├── Engagement-Recognition\          ← repo clonado (só continha dados de exemplo)
├── ER_Dataset\                      ← dataset reorganizado em 4 níveis (2.120 imagens)
├── testes\
├── backup_2026-09-02\                   ← backup completo pré-mudanças de setembro (não versionado)
├── .streamlit\config.toml              ← tema da demo Streamlit
├── paginas\
│   ├── classificar.py                  ← classificação ao vivo pela webcam
│   └── resultados.py                   ← métricas, matrizes de confusão, McNemar
├── .gitignore
├── app.py                              ← app Streamlit (menu + roteamento)
├── features.py                         ← extração de características (compartilhada por treinar.py e a demo)
├── organizar_dataset.py
├── treinar.py
├── mcnemar.py
├── requirements.txt
├── README.md
├── confusion_SVM.png
├── confusion_Random_Forest.png
├── confusion_MLP.png
├── model_SVM.pkl
├── model_Random_Forest.pkl
├── model_MLP.pkl
├── X_test.npy
├── y_test.npy
└── shape_predictor_68_face_landmarks.dat   ← ~95MB, fora do Git
```

---

## 8. Prompt de continuação (para colar em outra conta/sessão do Claude)

Copie o bloco abaixo integralmente na primeira mensagem da nova conversa, junto com o arquivo `.docx` mais recente do TCC e, se possível, este arquivo de base de conhecimento.

```
Estou continuando um TCC (UNIP, Ciência da Computação) sobre classificação automática de
níveis de engajamento em ambientes corporativos digitais usando Machine Learning
supervisionado (SVM, Random Forest e MLP) a partir de landmarks faciais extraídos com dlib.

CONTEXTO DO QUE JÁ FOI FEITO:
- O documento .docx já tem os Capítulos 1 (Introdução), 2 (Revisão da Literatura) e 3
  (Materiais e Métodos) prontos, além de toda a estrutura pré-textual (capa, resumo,
  abstract, lista de siglas, sumário, referências ABNT).
- O dataset planejado originalmente era o DAiSEE (vídeos), mas o ZIP baixado estava
  corrompido. Por isso, o dataset REALMENTE usado foi o "Student Engagement Dataset" do
  Kaggle (imagens estáticas), reorganizado em 4 níveis de engajamento (2.120 imagens no
  total: Nível 0 = 686, Nível 1 = 358, Nível 2 = 729, Nível 3 = 347).
- O pipeline real: detecção facial com dlib (HOG+SVM) → 68 landmarks faciais → vetor de
  136 características geométricas normalizadas (SEM extração de frames de vídeo e SEM
  HOG/LBP, diferente do que o Capítulo 3 atual descreve — isso precisa ser corrigido).
- Os três modelos já foram treinados e avaliados, com estes resultados sobre 396 amostras
  de teste:
    SVM: 79% acurácia | Random Forest: 95% acurácia | MLP: 94% acurácia
  F1 por classe, teste de McNemar (RF vs MLP sem diferença estatística significativa,
  p=0,8026; SVM significativamente inferior aos outros dois) e matrizes de confusão já
  foram gerados e estão disponíveis como imagens (.png) e nos scripts do repositório
  GitHub: https://github.com/brunalozinski/TCC-Engajamento

O QUE PRECISO QUE VOCÊ FAÇA AGORA:
1. Revisar e atualizar o Capítulo 3 (Materiais e Métodos) para descrever corretamente o
   pipeline REAL (dataset Kaggle, sem extração de frames, sem HOG/LBP — apenas landmarks
   geométricos), incluindo uma justificativa honesta da troca de dataset em relação ao
   DAiSEE originalmente planejado.
2. Redigir o Capítulo 4 (Resultados) com as tabelas de desempenho, análise por classe e
   as matrizes de confusão como figuras — vou te passar os arquivos.
3. Redigir o Capítulo 5 (Discussão).
4. Redigir o Capítulo 6 (Conclusão e Trabalhos Futuros).
5. Montar os apêndices com o código-fonte (organizar_dataset.py, treinar.py, mcnemar.py).
6. Gerar a Lista de Figuras e Lista de Tabelas depois de tudo inserido.
7. Revisar o resumo e o abstract para garantir coerência com os resultados reais (não
   afirmar que um modelo é "o melhor" sem qualificar, já que RF e MLP são estatisticamente
   equivalentes pelo teste de McNemar).
8. Me ajudar a decidir e, se eu topar, implementar a captura de fotos dos integrantes do
   grupo para um retreinamento comparativo (script capturar_imagens.py já existe).
9. Sinalizar tudo que ainda depende de mim: nome do orientador, dedicatória, agradecimentos,
   epígrafe e nomes reais da banca examinadora.

Vou anexar o arquivo .docx mais atual do TCC. Comece confirmando o que você consegue ver
no documento anexado e me diga qual pendência da lista acima faz mais sentido atacar
primeiro.
```

---

---

## 9. Atualização de setembro/2026 — correção das características, retreino e demo

> Esta seção substitui as seções 3 e 4 para efeito de dados a inserir no TCC.
> Todo o estado anterior do projeto foi copiado para `C:\TCC\backup_2026-09-02\`
> (backup completo, inclusive `.git`) antes das mudanças.

### 9.1. Problema identificado

Ao testar a demo ao vivo (webcam), uma pessoa visivelmente **engajada** era
classificada como **"Baixo"**, com alta "confiança" do modelo. Investigação
apontou duas causas somadas:

1. **Características faciais não invariantes à posição/escala.** A função de
   extração usava as coordenadas `(x, y)` **absolutas em pixels** dos 68 pontos,
   divididas apenas pela largura do rosto (`coords /= face.width()`). Isso remove
   escala parcialmente, mas **não remove a posição**: o vetor de 136 valores
   dependia de *onde* o rosto estava no quadro e da resolução da câmera. As
   imagens do dataset Kaggle (rostos centralizados, recorte e resolução
   consistentes) geram vetores numa faixa; a webcam (resolução, distância e
   enquadramento diferentes) gera vetores em outra região → o modelo extrapola.
2. **Rótulo "Baixo" = "Bored".** Fora da distribuição de treino, um rosto atento
   porém neutro é geometricamente o vizinho mais próximo de "entediado".

### 9.2. O que foi mudado

**Novo módulo `features.py` (raiz do projeto)** — passou a existir **uma única**
função de extração de características, importada **tanto pelo `treinar.py` quanto
pela demo (`paginas/classificar.py`)**. Antes a função estava duplicada nos dois
arquivos, o que permitiu que treino e uso real divergissem.

**Nova normalização dos 68 pontos** (dentro de `features.py`, função
`normalizar_pontos`):

1. **Posição:** transladar todos os pontos de modo que a **ponta do nariz
   (ponto 30)** fique na origem `(0, 0)` — `pts -= pts[30]`.
2. **Tamanho:** dividir todas as coordenadas pela **distância entre os cantos
   externos dos olhos (pontos 36 e 45)** — `pts /= ||pts[36] - pts[45]||`.

Assim o mesmo rosto gera aproximadamente o mesmo vetor independentemente de
onde está no quadro e da distância até a câmera. **Rotação da cabeça não é
corrigida** (decisão deliberada: acrescenta trigonometria e o ganho é pequeno
para alguém sentado olhando a tela) — pode ficar como trabalho futuro.

**O que NÃO mudou:** continua sendo 68 landmarks do dlib → vetor de **136
valores geométricos** (sem HOG/LBP, sem extração de frames); mesmos 3 modelos
(SVM RBF com `StandardScaler`, Random Forest 200 árvores, MLP 256-128); mesma
divisão 80/20 estratificada com `random_state=42`; mesmos 4 níveis de
engajamento.

**Limpeza:** caminhos fixos `C:/TCC/...` substituídos por caminho relativo à
raiz do projeto (`pathlib`) em `features.py`, `treinar.py`, `mcnemar.py`,
`paginas/classificar.py` e `paginas/resultados.py`.

### 9.3. Retreino — desempenho geral (396 amostras de teste, 1.980 no total)

| Modelo | Acurácia | Precisão Macro | Recall Macro | F1 Macro |
|---|---|---|---|---|
| SVM | 95% | 96% | 93% | 94% |
| Random Forest | 95% | 95% | 94% | 95% |
| MLP | **96%** | 95% | 95% | 95% |

Suporte por classe no teste: Nível 0 = 124, Nível 1 = 66, Nível 2 = 137,
Nível 3 = 69 (total 396). Total de amostras com rosto detectado: 1.980
(imagens em que o dlib não detecta rosto são descartadas).

### 9.4. Retreino — F1-Score por nível

| Classe | SVM | Random Forest | MLP |
|---|---|---|---|
| 0 — Muito Baixo | 0,95 | 0,94 | 0,94 |
| 1 — Baixo | 0,86 | 0,86 | 0,88 |
| 2 — Alto | 0,99 | 0,99 | 1,00 |
| 3 — Muito Alto | 0,96 | 0,99 | 0,99 |

O nível "Baixo" continua sendo o mais difícil (confunde-se com desengajamento
total e com engajamento passivo — consistente com Dewan, Murshed e Lin, 2019),
mas deixou de ser um ponto fraco isolado do SVM: subiu de **0,57 → 0,86** nesse
modelo e ficou equilibrado entre os três.

### 9.5. Retreino — Teste de McNemar

| Comparação | chi² | p-valor | Conclusão |
|---|---|---|---|
| SVM vs Random Forest | 0,0000 | 1,0000 | Diferença NÃO significativa |
| SVM vs MLP | 0,2353 | 0,6276 | Diferença NÃO significativa |
| Random Forest vs MLP | 0,2500 | 0,6171 | Diferença NÃO significativa |

### 9.6. Comparação antes × depois

| Métrica | Antes | Depois |
|---|---|---|
| Acurácia SVM | 79% | 95% |
| Acurácia Random Forest | 95% | 95% |
| Acurácia MLP | 94% | 96% |
| F1 do nível "Baixo" (SVM) | 0,57 | 0,86 |
| McNemar | SVM significativamente inferior a RF e MLP; RF ≡ MLP | **os 3 modelos são estatisticamente equivalentes** |

**Conclusão a registrar no TCC:** a normalização das características (translação
para o nariz + escala pela distância interocular) eliminou a desvantagem do SVM,
que era o modelo mais sensível a características mal escaladas. Após a correção,
**nenhum dos três modelos pode ser considerado superior aos demais** neste
conjunto de dados. Não usar frases que apontem um modelo como "o melhor".

### 9.7. Aplicação de demonstração (Streamlit)

Estrutura:

```
app.py                      ← menu lateral e roteamento das 3 seções
features.py                 ← extração de características (compartilhada com treinar.py)
paginas/
├── classificar.py          ← classificação ao vivo pela webcam
└── resultados.py           ← tabelas de métricas, matrizes de confusão e McNemar
.streamlit/config.toml      ← tema
```

Rodar: `python -m streamlit run app.py` (Python 3.11, com `streamlit` no
`requirements.txt`).

**Como a demo classifica o engajamento (`paginas/classificar.py`):**

1. A webcam é amostrada a **2 quadros por segundo** (não na taxa máxima — o
   engajamento não muda em milissegundos e amostrar rápido demais só gera ruído).
2. Cada quadro válido passa por `extrair_features` → o Random Forest devolve
   `predict_proba` (probabilidade de cada um dos 4 níveis).
3. As probabilidades dos últimos **20 segundos** ficam numa janela deslizante;
   o resultado usado é a **média** dessa janela (remove oscilação quadro a
   quadro).
4. Os 4 níveis são reduzidos a **2 para a demo**: `P(engajado) = P(nível 2) +
   P(nível 3)`. Essa probabilidade ainda passa por uma **suavização exponencial
   (EMA)**.
5. Decisão final:
   - se os **olhos ficam fechados por ~2 s** (razão de aspecto do olho / *EAR*
     abaixo de 0,15 em 4 amostras seguidas) → "Não engajado (olhos fechados)";
   - senão, "Engajado" se `P(engajado) ≥ 0,5`, caso contrário "Não engajado".
6. Quadros sem rosto frontal ou com rosto muito pequeno (< 100 px de largura)
   são ignorados e não entram na janela.

> Importante para a honestidade metodológica: **a demo ao vivo usa 2 classes;
> o TCC (treino, tabelas, matrizes de confusão, McNemar) continua com os 4
> níveis.** A redução para 2 classes vale só para a visualização em tempo real,
> onde a divisão em 4 é menos confiável.

**Relatório da sessão (novo):** enquanto a webcam está ligada, cada amostra é
registrada. Ao desligar a webcam, a página mostra um resumo e oferece dois
downloads:

- **CSV** (`relatorio_engajamento_AAAAMMDD_HHMMSS.csv`) — uma linha por amostra,
  colunas: `horario`, `segundos` (desde o início), `prob_engajado`, `engajado`
  (V/F), `olhos_fechados` (V/F), `ear`.
- **TXT** (`resumo_engajamento_...txt`) — resumo com: horário de início e fim,
  duração em segundos, nº de amostras, **% do tempo engajado**, P(engajado)
  média, mínima e máxima.

Também há um gráfico de `P(engajado)` ao longo do tempo (na tela e no relatório).

### 9.7.1. Sessão de teste registrada (exemplo real para o TCC)

Primeira sessão de demonstração gravada com o modelo retreinado (03/09/2026).
Arquivos em `C:\TCC\relatorio-teste-demo\`
(`relatorio_engajamento_20260903_001501.csv` e o resumo `.txt`).

**Configuração:** uma pessoa, webcam de notebook, iluminação de ambiente
interno; modelo Random Forest; janela de 20 s; amostragem de 2/s.

| Indicador | Valor |
|---|---|
| Duração | 422,4 s (≈ 7 min) |
| Amostras registradas | 836 |
| % do tempo classificado como "Engajado" | 34% |
| P(engajado) — média | 0,42 |
| P(engajado) — mínimo / máximo | 0,18 / 0,68 |
| P(engajado) — desvio padrão | 0,14 |
| Trocas de rótulo (Engajado ↔ Não engajado) na sessão | 8 |
| Amostras com "olhos fechados" detectado | 20 (≈ 2,4%) |
| Amostras sem rosto detectado (campo `ear` vazio) | 49 (≈ 5,9%) |
| EAR — média / mínimo / máximo | 0,25 / 0,08 / 0,47 |

**Leitura dos resultados (para a discussão do TCC):**

- Apenas **8 trocas de rótulo em 7 minutos** (uma a cada ~53 s em média)
  confirmam que a janela temporal de 20 s + suavização cumpriram o objetivo:
  a classificação deixou de oscilar quadro a quadro, ao contrário da versão
  inicial (que reclassificava a cada quadro, em frações de segundo).
- `P(engajado)` ficou concentrado na faixa **0,18–0,68** (nunca próximo de 0
  nem de 1), coerente com um cenário real onde a pessoa alterna atenção plena,
  atenção parcial e distração — diferente das imagens de dataset, que são
  extremos posados.
- O valor médio 0,42 e os 34% de tempo "Engajado" descrevem uma sessão de
  engajamento **intermediário/oscilante**, não um extremo.
- A verificação de olhos fechados disparou em ~2,4% das amostras (piscadas
  longas / olhar para baixo), sem dominar o resultado — comportamento esperado
  do limiar conservador (EAR < 0,15 por ~2 s).
- ~5,9% das amostras sem rosto (pessoa saiu do enquadramento ou virou muito):
  essas amostras são ignoradas e não contaminam a janela.

> ⚠️ É **um** registro, de **uma** pessoa. Serve como ilustração do
> funcionamento da aplicação em tempo real (Capítulo 4 ou apêndice), **não**
> como avaliação quantitativa do modelo — a avaliação formal é a das seções
> 9.3–9.5, sobre o conjunto de teste rotulado.

### 9.8. Arquivos alterados / regerados

- **Novos:** `features.py`, `app.py`, `paginas/classificar.py`,
  `paginas/resultados.py`, `.streamlit/config.toml`.
- **Editados:** `treinar.py`, `mcnemar.py` (uso de `features.py` + caminhos
  relativos); `.gitignore` (reescrito em UTF-8; ignora `*.pkl`, `*.npy`,
  `archive/`, `ER_Dataset/`, `__pycache__/`, `backup_*/`); `requirements.txt`
  (`streamlit==1.57.0`).
- **Regerados pelo retreino:** `model_SVM.pkl`, `model_Random_Forest.pkl`,
  `model_MLP.pkl`, `X_test.npy`, `y_test.npy`, `confusion_SVM.png`,
  `confusion_Random_Forest.png`, `confusion_MLP.png`. As versões anteriores
  estão em `backup_2026-09-02/`.
- **Saída da demo:** `relatorio-teste-demo/` — CSV e TXT da primeira sessão de
  teste gravada (ver 9.7.1).

### 9.9. Impacto no texto do TCC (pendências atualizadas)

1. **Capítulo 3** — descrever a extração de características com a normalização
   nova (nariz na origem + escala pela distância interocular); justificar que
   sem isso o vetor dependia da posição do rosto no quadro. Continua valendo:
   imagens estáticas do Kaggle, sem extração de frames, sem HOG/LBP, 136 valores
   geométricos.
2. **Capítulo 4** — usar as tabelas 9.3, 9.4 e 9.5 e as **novas** matrizes de
   confusão. Registrar a comparação antes/depois (9.6) como parte da discussão
   metodológica (uma primeira versão das características se mostrou frágil a
   mudança de domínio; a correção elevou o SVM ao mesmo nível dos outros).
   Incluir a **sessão de teste da aplicação (9.7.1)** como demonstração do uso
   em tempo real — deixando claro que é ilustrativa, não uma métrica do modelo.
3. **Capítulo 5 (Discussão)** — o "Baixo" segue como a classe mais ambígua;
   os 3 modelos ficaram equivalentes (McNemar); a fragilidade a domínio mostra
   o limite de usar coordenadas geométricas puras.
4. **Capítulo 6 / Trabalhos Futuros** — (a) modelagem **temporal** de verdade
   (LSTM/GRU sobre sequências de landmarks, exige dataset de vídeo tipo DAiSEE);
   (b) **pose da cabeça** (pitch/yaw/roll via `solvePnP`) e direção do olhar como
   características; (c) **calibração por pessoa** (linha de base individual de
   "atento" e "distraído"); (d) correção de rotação da cabeça na normalização.
5. **Apêndices** — incluir `features.py` além de `organizar_dataset.py`,
   `treinar.py` e `mcnemar.py`; opcionalmente `app.py` e `paginas/`.
6. **Resumo / Abstract** — não afirmar que um modelo é superior (os três são
   equivalentes pelo McNemar).

### 9.10. Ambiente confirmado

- Python **3.11** (`C:\Users\lozin\AppData\Local\Programs\Python\Python311`) —
  é o interpretador com `dlib`, `streamlit`, `opencv`, `scikit-learn`, `pandas`
  instalados. O Python 3.14 que está no PATH **não** tem as bibliotecas.
- `shape_predictor_68_face_landmarks.dat` (~95 MB) permanece fora do Git.

---

*Documento gerado a partir da consolidação de conversas anteriores do projeto. Sempre confira os dados numéricos (acurácias, contagens de imagens) contra os arquivos reais antes de usá-los como fato definitivo no TCC final.*
