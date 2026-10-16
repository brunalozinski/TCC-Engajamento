"""
Mapeamento de GRUPO REAL DE PESSOA para o ER_Dataset (Kaggle), feito por
revisao visual manual de uma imagem representativa de cada um dos 45 grupos
identificados pela heuristica de nome de arquivo (ver treinar.py,
identificar_grupo_de_cada_imagem).

Por que isso existe: a heuristica de agrupamento por salto de numeracao no
nome do arquivo (treinar.py) NAO identifica pessoa real -- ela so agrupa
frames consecutivos de um mesmo "pedaco" de video. A revisao visual descobriu
que:
  1) A mesma pessoa real as vezes aparece em MULTIPLOS grupos da heuristica
     dentro da mesma categoria (ex: Drowsy tem 2 pessoas intercaladas nos
     13 grupos).
  2) A mesma pessoa real aparece em CATEGORIAS E CLASSES DIFERENTES (ex: a
     pessoa "E" aparece em Bored=1, Frustrated=2 e Focused=3).

Resultado da revisao: o ER_Dataset inteiro (2120 imagens, 45 grupos pela
heuristica) tem APENAS 5 PESSOAS REAIS DISTINTAS (A, B, C, E, F; "D" e "G"
foram identificadas inicialmente mas confirmadas como a mesma pessoa que
"C" e "B", respectivamente), identificadas por caracteristicas visuais
(roupa, oculos, cabelo, fundo/cortina).

ATENCAO - limitacao conhecida (confirmada por revisao manual do usuario):
  - "Pessoa C" e "Pessoa D" sao A MESMA PESSOA (fundidas abaixo em "C").
  - Dentro da categoria "Not Engaged_Drowsy", Pessoa A e Pessoa B se
    ALTERNAM FRAME A FRAME em alguns trechos, inclusive DENTRO de um
    mesmo grupo da heuristica de numeracao (ex: o grupo 12, frames
    458-469, tem o frame 460 = Pessoa B misturado com o resto = Pessoa A).
    Isso foi descoberto conferindo arquivos especificos (21-24, 36, 92-94,
    460), nao estava visivel na amostragem de 1 frame por grupo.
  - Por causa disso, o mapeamento abaixo (por GRUPO) e uma APROXIMACAO,
    nao uma verdade por imagem -- pode haver frames individuais mal
    classificados dentro de grupos atribuidos a A ou B. Para precisao real
    por imagem seria necessario clustering automatico por reconhecimento
    facial (embeddings), nao feito ainda.
  - O arquivo Not Engaged_Drowsy_0036.jpg (inicialmente visto como uma 7a
    pessoa "G") foi confirmado pelo usuario como sendo a Pessoa B.

Uso: este modulo exporta PESSOA_POR_GRUPO, um dicionario que mapeia o rotulo
de grupo gerado por identificar_grupo_de_cada_imagem() (formato
"{categoria}_grupo{N}", igual ao de treinar.py) para um ID de pessoa real
(string: "A", "B", "C", "D", "E" ou "F").

Um script de unificacao (ex: treinar_unificado.py) pode usar isto assim:

    from treinar import identificar_grupo_de_cada_imagem
    from mapeamento_pessoas_kaggle import PESSOA_POR_GRUPO

    grupos_heuristica = identificar_grupo_de_cada_imagem(nomes_arquivo)
    pessoa_real = [PESSOA_POR_GRUPO[g] for g in grupos_heuristica]
    # usar pessoa_real como 'groups' no StratifiedGroupKFold, em vez de
    # grupos_heuristica
"""

PESSOA_POR_GRUPO = {
    # --- Classe 0 (Muito Baixo) ---------------------------------------
    # Not Engaged_Drowsy: 2 pessoas intercaladas nos 13 grupos da heuristica
    "Not Engaged_Drowsy_grupo0": "A",
    "Not Engaged_Drowsy_grupo1": "B",
    "Not Engaged_Drowsy_grupo2": "A",
    "Not Engaged_Drowsy_grupo3": "A",
    "Not Engaged_Drowsy_grupo4": "B",
    "Not Engaged_Drowsy_grupo5": "A",
    "Not Engaged_Drowsy_grupo6": "B",
    "Not Engaged_Drowsy_grupo7": "A",
    "Not Engaged_Drowsy_grupo8": "A",
    "Not Engaged_Drowsy_grupo9": "A",
    "Not Engaged_Drowsy_grupo10": "A",
    "Not Engaged_Drowsy_grupo11": "A",
    "Not Engaged_Drowsy_grupo12": "A",

    # Not Engaged_Looking Away: pessoa C nos grupos 0-3, pessoa D no grupo 4
    "Not Engaged_Looking Away_grupo0": "C",
    "Not Engaged_Looking Away_grupo1": "C",
    "Not Engaged_Looking Away_grupo2": "C",
    "Not Engaged_Looking Away_grupo3": "C",
    "Not Engaged_Looking Away_grupo4": "C",  # fundida com C (confirmado pelo usuario)

    # --- Classe 1 (Baixo) ----------------------------------------------
    # Not Engaged_Bored: 3 pessoas (A, E, C) intercaladas nos 10 grupos
    # ("D" fundida com "C" -- ver nota de limitacao acima)
    "Not Engaged_Bored_grupo0": "A",
    "Not Engaged_Bored_grupo1": "E",
    "Not Engaged_Bored_grupo2": "E",
    "Not Engaged_Bored_grupo3": "E",
    "Not Engaged_Bored_grupo4": "C",
    "Not Engaged_Bored_grupo5": "C",
    "Not Engaged_Bored_grupo6": "C",
    "Not Engaged_Bored_grupo7": "A",
    "Not Engaged_Bored_grupo8": "A",
    "Not Engaged_Bored_grupo9": "C",

    # --- Classe 2 (Alto) -------------------------------------------------
    # Engaged_Confused: pessoa F em todos os 7 grupos
    "Engaged_Confused_grupo0": "F",
    "Engaged_Confused_grupo1": "F",
    "Engaged_Confused_grupo2": "F",
    "Engaged_Confused_grupo3": "F",
    "Engaged_Confused_grupo4": "F",
    "Engaged_Confused_grupo5": "F",
    "Engaged_Confused_grupo6": "F",

    # Engaged_Frustrated: pessoa E em todos os 6 grupos
    "Engaged_Frustrated_grupo0": "E",
    "Engaged_Frustrated_grupo1": "E",
    "Engaged_Frustrated_grupo2": "E",
    "Engaged_Frustrated_grupo3": "E",
    "Engaged_Frustrated_grupo4": "E",
    "Engaged_Frustrated_grupo5": "E",

    # --- Classe 3 (Muito Alto) ------------------------------------------
    # Engaged_Focused: pessoa E em todos os 4 grupos (classe inteira = 1 pessoa)
    "Engaged_Focused_grupo0": "E",
    "Engaged_Focused_grupo1": "E",
    "Engaged_Focused_grupo2": "E",
    "Engaged_Focused_grupo3": "E",
}

# Resumo de quais classes cada pessoa real aparece (para referencia rapida).
# "D" e "G" foram fundidas em "C" e "B" respectivamente -- so 5 pessoas reais
# distintas ficam no Kaggle (A, B, C, E, F).
CLASSES_POR_PESSOA = {
    "A": [0, 1],
    "B": [0],
    "C": [0, 1],
    "E": [1, 2, 3],
    "F": [2],
}
