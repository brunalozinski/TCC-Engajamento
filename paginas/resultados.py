import pandas as pd
import streamlit as st

from features import RAIZ

PASTA_JEAN = RAIZ / "TCC"
TABELA = PASTA_JEAN / "tabela_consolidada_experimentos_binarios.csv"
FIGURAS = PASTA_JEAN / "figuras_tcc"
ORDEM = ["DINOv2 congelado + LSTM", "DINOv2 fine-tuning parcial + LSTM",
         "Swin-tiny congelado + LSTM", "Swin-tiny fine-tuning parcial + LSTM"]
LETRA = {ORDEM[0]: "C", ORDEM[1]: "E", ORDEM[2]: "D", ORDEM[3]: "F"}


def _celula(linha):
    return f"{linha.media:.3f} ± {linha.desvio:.3f}".replace(".", ",")


def _tabela(t, conjunto, metricas):
    """Linhas = métodos, colunas = métricas (média ± desvio de 5 sementes)."""
    sub = t[t.conjunto_teste == conjunto]
    linhas = []
    for metodo in ORDEM:
        linha = {"Exp.": LETRA[metodo], "Método": metodo}
        for m in metricas:
            r = sub[(sub.metodo == metodo) & (sub.metrica == m)].iloc[0]
            linha[m] = _celula(r)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("Exp.")


def _comparacao_experimentos():
    st.subheader("Comparação dos experimentos: DAiSEE + EngageNet, 2 classes")
    st.write(
        "Tarefa: dizer se o vídeo mostra uma pessoa **engajada** ou **não engajada**. "
        "Os quatro métodos usam os **mesmos vídeos, os mesmos 3 frames por vídeo, as "
        "mesmas divisões de treino/validação/teste (por pessoa)** e 5 repetições do "
        "treino (sementes). Os números são a média ± desvio entre as 5 sementes, no "
        "conjunto de **teste**. O acaso vale 0,50."
    )
    try:
        t = pd.read_csv(TABELA, sep=";", decimal=",")
    except Exception as erro:
        st.error(f"Não foi possível ler a tabela consolidada: {erro}")
        return

    st.markdown("**Teste balanceado (1.564 vídeos, 782 por classe)**")
    st.table(_tabela(t, "Teste balanceado (1564)", ["F1 macro", "Acuracia balanceada", "AUC"])
             .rename(columns={"Acuracia balanceada": "Acurácia balanceada"}))

    st.markdown("**Por fonte e no teste completo (distribuição natural, 81% engajados)**")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Só DAiSEE (176 vídeos)")
        st.table(_tabela(t, "Balanceado so DAiSEE (176)", ["F1 macro", "AUC"]))
        st.caption("Teste completo natural (4.030 vídeos)")
        st.table(_tabela(t, "Teste completo natural (4030)", ["F1 macro", "AUC"]))
    with col2:
        st.caption("Só EngageNet (1.388 vídeos)")
        st.table(_tabela(t, "Balanceado so EngageNet (1388)", ["F1 macro", "AUC"]))

    graf = _tabela(t, "Teste balanceado (1564)", ["F1 macro", "AUC"])
    num = pd.DataFrame({
        m: [float(v.split(" ")[0].replace(",", ".")) for v in graf[m]] for m in ("F1 macro", "AUC")
    }, index=list(graf.index))
    st.caption("F1 macro e AUC no teste balanceado (barras lado a lado; acaso = 0,50). "
               "C: DINOv2 congelado · E: DINOv2 fine-tuning parcial · D: Swin congelado · F: Swin fine-tuning parcial")
    st.bar_chart(num, horizontal=True, stack=False)

    st.markdown("**Como ler**")
    st.markdown(
        "- **C** (DINOv2 congelado) e **E** (DINOv2 com ajuste fino parcial) foram os melhores; a diferença entre eles (+0,013 em F1 e em AUC) é do tamanho da variação entre sementes, então **não é um ganho comprovado**.\n"
        "- **D** e **F** usam o Swin-tiny. O ajuste fino ajudou mais nele (F1 0,612 → 0,669), mas ele continuou abaixo do DINOv2.\n"
        "- A **escolha do modelo** que gera os embeddings pesou mais do que ajustá-lo.\n"
        "- O teste do DAiSEE tem só 88 vídeos por classe (incerteza grande) e, em todos os métodos, o resultado nele é pior que no EngageNet."
    )

    st.markdown("**Figuras (melhor semente de cada método pela validação, limiar 0,5)**")
    for arquivo, legenda in (
        ("matrizes_confusao_balanceado.png", "Matrizes de confusão, teste balanceado"),
        ("curvas_roc_balanceado.png", "Curvas ROC, teste balanceado"),
        ("matrizes_confusao_natural.png", "Matrizes de confusão, teste completo (distribuição natural)"),
    ):
        caminho = FIGURAS / arquivo
        if caminho.exists():
            st.image(str(caminho), caption=legenda, width=720)

    st.markdown("**Experimentos anteriores, 4 classes (média de 5 sementes, teste)**")
    st.table(pd.DataFrame({
        "Experimento": ["A — DAiSEE, 4 classes", "B — EngageNet, 4 classes"],
        "F1 macro (maior média no teste)": ["0,320", "0,484"],
        "Observação": [
            "Classes raras praticamente não aprendidas (34 vídeos da classe 0 no treino).",
            "Classes intermediárias difíceis; AUC macro 0,765.",
        ],
    }))
    st.caption(
        "Os experimentos A e B não são comparáveis diretamente com C a F: mudam o número "
        "de classes, de frames e o conjunto de teste."
    )


def exibir():
    st.header("Resultados do Treinamento")
    _comparacao_experimentos()
