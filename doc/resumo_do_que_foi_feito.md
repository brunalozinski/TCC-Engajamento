# O que foi feito no projeto até agora

*Texto escrito em linguagem simples, para servir de base na documentação do TCC.*

---

## 1. Qual é o projeto?

Um sistema que olha para o **rosto** de uma pessoa (pela webcam ou por fotos) e diz **o quanto ela está engajada** — por exemplo, numa aula ou reunião online.

Como ele "enxerga" o engajamento?

1. Uma biblioteca chamada **dlib** encontra o rosto e marca **68 pontinhos** nele (contorno do rosto, sobrancelhas, olhos, nariz, boca).
2. Esses pontinhos viram uma lista de **136 números** (x e y de cada ponto).
3. **Três modelos de Machine Learning** (SVM, Random Forest e MLP) aprendem a relacionar esses números com um **nível de engajamento**:

| Nível | Nome | O que representa (categorias originais do Kaggle) |
|---|---|---|
| 0 | Muito Baixo | Sonolento (Drowsy) + Desviando o olhar (Looking Away) |
| 1 | Baixo | Entediado (Bored) |
| 2 | Alto | Confuso (Confused) + Frustrado (Frustrated) |
| 3 | Muito Alto | Focado (Focused) |

Os dados de treino vêm do dataset público **Student Engagement Dataset** (Kaggle), reorganizado em 4 pastas (`ER_Dataset/0` a `3`, 2.120 imagens).

---

## 2. Etapa 1 — Construção da base

- Reorganização do dataset do Kaggle nos 4 níveis (`organizar_dataset.py`).
- Treino dos 3 modelos, geração das matrizes de confusão e teste estatístico de McNemar para comparar os modelos (`treinar.py`, `mcnemar.py`).
- Primeiros resultados: **mais de 90% de acerto**. Parecia ótimo… mas era bom demais para ser verdade (ver etapa 4).

---

## 3. Etapa 2 — O aplicativo (demo ao vivo)

Foi criado um app em **Streamlit** (`app.py` + pasta `paginas/`) com duas telas: **Classificar** (webcam) e **Resultados** (relatórios).

A ideia principal é a **janela temporal**:

- Olhar um único frame é instável (a pessoa pisca, vira a cabeça por um instante). Então o app **não decide a cada foto**.
- Ele coleta amostras (2 por segundo) durante **20 segundos** e, no final de cada janela, dá **um veredito**: 🔴 Baixo, 🟠 Moderado ou 🟢 Alto engajamento.
- O score da janela é `P(engajado) = P(nível 2) + P(nível 3)`. Abaixo de 40% = Baixo; de 40% a 65% = Moderado; acima de 65% = Alto.
- **Detecção de olhos fechados** (EAR — razão de abertura do olho): o app mede o olho aberto natural da pessoa nos 3 primeiros segundos e só considera "fechado" se o olho ficar bem fechado por mais de 2 segundos seguidos. Se estiver fechado, a janela vira "Baixo".
- Rostos pequenos demais (baixa qualidade) são ignorados, e uma janela só vale com pelo menos 5 amostras.
- Ao final, o app gera um relatório CSV e um resumo em texto.

---

## 4. Etapa 3 — Normalização dos pontos do rosto (`features.py`)

Problema: se a pessoa estiver mais perto da câmera ou mais para o canto da imagem, os números mudam, mesmo que a expressão seja a mesma.

Solução: antes de usar os 68 pontos, eles são **normalizados**:

- **Posição:** a ponta do nariz vira a origem (0, 0).
- **Tamanho:** tudo é dividido pela distância entre os cantos externos dos olhos.

Assim, o mesmo rosto gera os mesmos números em qualquer posição ou distância. Esse mesmo arquivo é usado **tanto no treino quanto na demo**, para garantir que os dois "falam a mesma língua". Também foi criada a função `analisar_rosto`, que detecta o rosto **uma vez só** por frame e já devolve features + retângulo + EAR (antes eram duas detecções).

---

## 5. Etapa 4 — A descoberta do vazamento de dados (a parte mais importante)

### O problema
O dataset do Kaggle não é feito de fotos independentes: são **frames de vídeos**. Frames vizinhos (ex.: `0020`, `0021`, `0022`) são praticamente a mesma imagem. O treino original sorteava imagens **uma a uma** para treino e teste, então o teste tinha "gêmeos" das imagens de treino. O modelo parecia ótimo, mas só estava **reconhecendo o que já tinha visto**. Isso se chama **vazamento de dados (data leakage)**.

*Analogia:* é como dar ao aluno a prova com as mesmas questões que ele já estudou, só com a ordem das alternativas trocada — a nota alta não prova que ele aprendeu a matéria.

### Como foi provado
1. **Teste do quase-gêmeo:** 393 de 396 imagens de teste (99,2%) tinham um frame quase idêntico no treino.
2. **Teste do 1-NN:** um classificador "burro", que só copia o rótulo da imagem mais parecida, acertou **96,21%** — sem aprender nada sobre engajamento. Logo, o teste não era independente.

### Como foi corrigido
- Cada imagem foi colocada num **grupo** (blocos de frames consecutivos, inferidos pela numeração do arquivo).
- A divisão treino/teste passou a usar `StratifiedGroupKFold`: **o grupo inteiro vai para o treino OU para o teste, nunca para os dois**, mantendo a proporção de classes.
- Prova de que funcionou: o 1-NN caiu de 96,21% para 54,45%.

### Resultados depois da correção (honestos)

| Modelo | Antes (contaminado) | Depois (corrigido) |
|---|---|---|
| SVM | ~90%+ | 52% |
| Random Forest | ~90%+ | 50% |
| MLP | ~90%+ | 56% |

A queda **não é o modelo piorando** — é a medição ficando honesta. Para o TCC, achar e corrigir esse problema é uma contribuição metodológica em si.

### Otimização de hiperparâmetros (`otimizar_hiperparametros.py`)
Foi feita uma busca em grade (GridSearchCV, também com validação por grupo) nos "botões" dos modelos. Ganho pequeno: SVM 52→54,8%, MLP 56→56,6%, Random Forest sem ganho. Conclusão: **o gargalo não é o algoritmo, é a informação que os 68 pontos carregam.**

### Onde os modelos erram (matrizes de confusão)
- Eles **separam bem "engajado" de "não engajado"** (quase nunca confundem Muito Baixo com Alto/Muito Alto).
- O erro está **dentro de cada grupo**: Alto × Muito Alto (focado vs. confuso/frustrado) e Muito Baixo × Baixo (sonolento vs. entediado). Essas diferenças são sutis demais para apenas a geometria dos pontos.

> Isso justifica, na demo, juntar os 4 níveis num score só de "engajado vs. não engajado" (níveis 2+3), que é justamente a distinção que os modelos fazem bem.

---

## 6. Etapa 5 — Descobrindo que o dataset tem só 5 pessoas

Ao revisar as imagens manualmente (`mapeamento_pessoas_kaggle.py`), descobriu-se que **as 2.120 imagens do Kaggle mostram apenas 5 pessoas reais** (A, B, C, E, F). A mesma pessoa aparece em categorias diferentes e em vários grupos. Ou seja, mesmo o split por grupo ainda pode misturar a mesma pessoa no treino e no teste.

Limitação conhecida: em alguns trechos de "Drowsy", duas pessoas se alternam frame a frame, então o mapeamento por grupo é uma aproximação. Para ser exato seria preciso reconhecimento facial automático (embeddings) — ainda não feito.

---

## 7. Etapa 6 — Teste externo com pessoas novas

Para medir generalização de verdade, foi gravado um **conjunto externo** (`ER_Dataset_Externo/`): **1.402 imagens de 3 pessoas** que nunca apareceram no treino. Os frames são extraídos de vídeos com `extrair_frames_video.py`, e o nome do arquivo já guarda pessoa, sessão e classe (assim não é preciso "adivinhar" nada).

Avaliação (`avaliar_externo.py`, relatório em `relatorio_avaliacao_externa.txt`):

| Modelo | Acurácia no externo |
|---|---|
| SVM | 26% |
| Random Forest | 24% |
| MLP | 23% |

São quase o nível de chute (25% com 4 classes). Os modelos tendem a prever "Muito Baixo" para quase tudo. **Interpretação:** os modelos treinados com apenas 5 pessoas **não generalizam para pessoas novas**. Esse é o resultado real e deve ser reportado de forma transparente como limitação.

---

## 8. Etapa 7 — Próximos passos já preparados

- **Dataset unificado** (`montar_dataset_unificado.py` → `ER_Dataset_Unificado/`): junta o Kaggle (com o ID real de pessoa no nome) e o externo, totalizando 3.522 imagens (991 / 681 / 1.055 / 795 por nível), para que o split seja **por pessoa**. Nada foi apagado ou movido dos datasets originais.
- **Modelo final** (`treinar_final.py`): treina com 100% do `ER_Dataset` e avalia no externo (script pronto, sem sobrescrever os modelos atuais).
- **Teste prático da demo** (`teste_demo_pratico.py`): liga a webcam e compara a predição do modelo novo com a do modelo antigo (contaminado).

---

## 9. Organização e segurança do trabalho

- Foram feitos **backups numerados** (`backup_v1` a `backup_v4`) antes de cada rodada de mudanças, para poder voltar atrás.
- Os scripts novos **não sobrescrevem** resultados anteriores (usam sufixos como `_externo` e `_final`).
- Documentos de apoio: `doc/relatorio_experimentos_vazamento_dados.md` (relatório técnico detalhado) e `doc/BASE CONHECIMENTO TCC ENGAJAMENTO.md`.

---

## 10. Resumo em 5 frases (para falar na apresentação)

1. Criamos um sistema que classifica engajamento a partir de 68 pontos do rosto, com SVM, Random Forest e MLP.
2. Construímos uma demo ao vivo que decide o engajamento a cada **janela de 20 segundos**, com detecção de olhos fechados.
3. Nossos primeiros resultados (>90%) estavam inflados por **vazamento de dados**; provamos isso (1-NN com 96%) e corrigimos dividindo por grupos, chegando a **50–56%** reais.
4. Descobrimos que o dataset tem só **5 pessoas**, e o teste com 3 pessoas novas deu ~**25%** — os modelos ainda não generalizam.
5. Os próximos passos são treinar com mais pessoas (dataset unificado), dividir por pessoa e, se possível, usar features melhores que só a geometria dos pontos.

---

## Observação sobre o estado do repositório

Há muitas alterações **ainda não commitadas** (scripts novos, datasets externo/unificado, modelos retreinados). O último commit (`3ce93d3`) cobre apenas a normalização dos landmarks e o relatório da demo.
