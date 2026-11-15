import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Classificador de Engajamento",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Classificação de Níveis de Engajamento")
st.markdown("Trabalho de Conclusão de Curso - UNIP 2026")
st.divider()

# Menu lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Escolha uma seção:",
    ["Início", "Classificar em Tempo Real (DINOv2 + LSTM)", "Resultados do Treinamento"]
)

# Roteamento simples
if pagina == "Início":
    st.header("Bem-vindo")
    st.write("""
    Esta aplicação demonstra um sistema de classificação automática de 
    níveis de engajamento a partir de imagens faciais, desenvolvido como 
    parte do TCC de classificação de engajamento em contexto educacional
    (vídeos de alunos, datasets DAiSEE e EngageNet) por meio de modelos
    supervisionados de Machine Learning. O uso em ambientes corporativos é
    uma possibilidade futura, ainda não testada.

    **Como usar:**
    - **Classificar em Tempo Real (DINOv2 + LSTM):** abre a webcam e classifica seu engajamento ao vivo, com o modelo do experimento C (DINOv2 congelado + LSTM) ou do E (DINOv2 com fine-tuning parcial + LSTM)
    - **Resultados do Treinamento:** compara os experimentos (DAiSEE + EngageNet) (métricas, matrizes de confusão e curvas ROC)
    """)

elif pagina == "Classificar em Tempo Real (DINOv2 + LSTM)":
    from paginas import classificar_dinov2
    classificar_dinov2.exibir()

elif pagina == "Resultados do Treinamento":
    from paginas import resultados
    resultados.exibir()