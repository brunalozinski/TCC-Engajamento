import streamlit as st
from PIL import Image

def exibir():
    st.header("Resultados do Treinamento")

    # Tabela de métricas gerais
    st.subheader("Desempenho dos Modelos")
    dados_metricas = {
        "Modelo": ["SVM", "Random Forest", "MLP"],
        "Acurácia": ["79%", "95%", "94%"],
        "Precision Macro": ["76%", "94%", "94%"],
        "Recall Macro": ["77%", "93%", "92%"],
        "F1-Score": ["76%", "94%", "93%"]
    }
    st.table(dados_metricas)

    st.divider()

    # Matrizes de confusão
    st.subheader("Matrizes de Confusão")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**SVM**")
        st.image("C:/TCC/confusion_SVM.png")

    with col2:
        st.markdown("**Random Forest**")
        st.image("C:/TCC/confusion_Random_Forest.png")

    with col3:
        st.markdown("**MLP**")
        st.image("C:/TCC/confusion_MLP.png")

    st.divider()

    # Teste de McNemar
    st.subheader("Teste de McNemar")
    dados_mcnemar = {
        "Comparação": ["SVM vs Random Forest", "SVM vs MLP", "Random Forest vs MLP"],
        "chi²": ["53.1571", "54.3906", "0.0625"],
        "p-valor": ["0.0000", "0.0000", "0.8026"],
        "Conclusão": ["Significativa", "Significativa", "Não significativa"]
    }
    st.table(dados_mcnemar)

    st.info("""
    **Interpretação:** O SVM é estatisticamente inferior aos outros dois modelos. 
    Random Forest e MLP são estatisticamente equivalentes — a diferença de 1% 
    de acurácia entre eles pode ter ocorrido por acaso.
    """)