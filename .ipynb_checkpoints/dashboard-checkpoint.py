import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurações gerais
st.set_page_config(page_title="Análise Fisiológica - Lesão Medular", layout="wide")


# Função para carregar dados
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# Título principal
st.title("Dashboard de Análise de Parâmetros Fisiológicos em Modelo Animal")

# Abas
tab1, tab2 = st.tabs(["Contexto e Objetivos", "Análise Descritiva"])

with tab1:
    st.header("Contexto do Estudo")
    
    st.image("imagem_choque_neuro.png", caption="Representação do choque neurogênico", use_container_width=True)

    st.markdown("""
    **Fonte dos dados:**  
    Estudo realizado em macacos com lesão medular, com dados coletados por 4 diferentes aparelhos
    que posteriormente foram consolidados em um único conjunto de dados.

    **Parâmetros monitorados:**
    - Pressão arterial (sistólica, diastólica e média)
    - Frequência respiratória
    - Índices de perfusão (R1, R2, R3)
    - Pulsação cardíaca (R3)
    """)

    st.header("Objetivo da Análise")
    st.markdown("""
    **Variável target selecionada:** `systolic_mmhg` (Pressão Arterial Sistólica)  

    **Justificativa:**  
    Existe uma relação comprovada entre lesões medulares e alterações na pressão arterial sistólica.
    Animais com este tipo de lesão frequentemente apresentam:
    - Hipotensão ortostática
    - Disreflexia autonômica
    - Instabilidade cardiovascular

    **Objetivo principal:**  
    Acompanhar a evolução dos parâmetros fisiológicos para entender como diferentes intervenções
    (farmacológicas, de reabilitação ou terapia celular) podem contribuir para o tratamento de
    lesões medulares em modelos animais.
    """)

with tab2:
    st.header("Distribuição das Variáveis")
    st.info("Esta seção mostra a distribuição estatística de cada variável do dataset")

    try:
        df = load_data("vitals_mais_rs.csv")
    except:
        st.error("Arquivo não encontrado! Por favor, verifique o caminho do arquivo.")
        st.stop()

    columns = [
        'systolic_mmhg', 'diastolic_mmhg', 'map_mmhg',
        'respiration_bpm', 'r1_pi_value', 'r2_pi_value',
        'r3_pi_value', 'r3_pr_bpm_value'
    ]

    for col in columns:
        with st.expander(f"Análise da variável: **{col}**", expanded=False):
            c1, c2 = st.columns([4, 2])

            with c1:
                st.subheader(f"Distribuição de {col}")

                # Criar figura
                fig, ax = plt.subplots(figsize=(10, 4))

                # Plotar histograma e KDE
                sns.histplot(df[col], kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribuição de {col}")
                ax.set_xlabel("Valores")
                ax.set_ylabel("Frequência")

                # Adicionar linhas de média e mediana
                mean_val = df[col].mean()
                median_val = df[col].median()
                ax.axvline(mean_val, color='r', linestyle='--', label=f'Média: {mean_val:.2f}')
                ax.axvline(median_val, color='g', linestyle='-', label=f'Mediana: {median_val:.2f}')

                plt.legend()
                st.pyplot(fig)

            with c2:
                st.subheader("Estatísticas Descritivas")

                # Estatísticas básicas
                stats = df[col].describe()
                st.write(stats)

                # Informações adicionais
                st.metric("Valores Faltantes", df[col].isnull().sum())
                st.metric("Valores Únicos", df[col].nunique())

                # Boxplot compacto
                st.subheader("Detecção de Outliers")
                fig2, ax2 = plt.subplots(figsize=(4, 2))
                sns.boxplot(x=df[col], ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard desenvolvido para análise de dados fisiológicos**  
*Laboratório de Pesquisa em Lesões Medulares*  
Versão 1.0 - Junho 2025
""")