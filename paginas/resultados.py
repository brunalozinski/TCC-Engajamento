import streamlit as st

from features import RAIZ


def exibir():
    st.header("Resultados do Treinamento")

    # Tabela de métricas gerais
    st.subheader("Desempenho dos Modelos")
    dados_metricas = {
        "Modelo": ["SVM", "Random Forest", "MLP"],
        "Acurácia": ["95%", "95%", "96%"],
        "Precision Macro": ["96%", "95%", "95%"],
        "Recall Macro": ["93%", "94%", "95%"],
        "F1-Score Macro": ["94%", "95%", "95%"],
    }
    st.table(dados_metricas)

    st.subheader("F1-Score por nível de engajamento")
    dados_f1 = {
        "Nível": ["0 — Muito Baixo", "1 — Baixo", "2 — Alto", "3 — Muito Alto"],
        "SVM": ["0.95", "0.86", "0.99", "0.96"],
        "Random Forest": ["0.94", "0.86", "0.99", "0.99"],
        "MLP": ["0.94", "0.88", "1.00", "0.99"],
    }
    st.table(dados_f1)
    st.caption(
        "O nível \"Baixo\" continua sendo o mais difícil de classificar — "
        "confunde-se com desengajamento total e com engajamento passivo."
    )

    st.divider()

    # Matrizes de confusão
    st.subheader("Matrizes de Confusão")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**SVM**")
        st.image(str(RAIZ / "confusion_SVM.png"))

    with col2:
        st.markdown("**Random Forest**")
        st.image(str(RAIZ / "confusion_Random_Forest.png"))

    with col3:
        st.markdown("**MLP**")
        st.image(str(RAIZ / "confusion_MLP.png"))

    st.divider()

    # Teste de McNemar
    st.subheader("Teste de McNemar")
    dados_mcnemar = {
        "Comparação": ["SVM vs Random Forest", "SVM vs MLP", "Random Forest vs MLP"],
        "chi²": ["0.0000", "0.2353", "0.2500"],
        "p-valor": ["1.0000", "0.6276", "0.6171"],
        "Conclusão": ["Não significativa", "Não significativa", "Não significativa"],
    }
    st.table(dados_mcnemar)

    st.info("""
    **Interpretação:** com as características faciais normalizadas (posição e
    tamanho), os três modelos passaram a ter desempenho estatisticamente
    equivalente (todos os p-valores > 0,05). Nenhum modelo pode ser considerado
    superior aos demais neste conjunto de dados.
    """)
