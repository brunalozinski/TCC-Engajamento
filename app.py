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
    ["Início", "Classificar em Tempo Real", "Resultados do Treinamento"]
)

# Roteamento simples
if pagina == "Início":
    st.header("Bem-vindo")
    st.write("""
    Esta aplicação demonstra um sistema de classificação automática de 
    níveis de engajamento a partir de imagens faciais, desenvolvido como 
    parte do TCC de classificação de padrões comportamentais em ambientes 
    corporativos simulados por meio de modelos supervisionados de Machine Learning.

    **Como usar:**
    - **Classificar em Tempo Real:** abre a webcam e classifica seu engajamento ao vivo
    - **Resultados do Treinamento:** mostra as métricas, matrizes de confusão e teste de McNemar
    """)

elif pagina == "Classificar em Tempo Real":
    from paginas import classificar
    classificar.exibir()

elif pagina == "Resultados do Treinamento":
    from paginas import resultados
    resultados.exibir()