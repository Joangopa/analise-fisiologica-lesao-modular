import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.header("Análise Série Temporal das Variáveis")
    st.info("Selecione uma variável ou grupo abaixo para visualizar seus valores ao longo das observações")
    
    try:
        df = load_data("vitals_mais_rs.csv")
        # Resetar o índice para começar em 1
        df.index = df.index + 1
    except:
        st.error("Arquivo não encontrado! Por favor, verifique o caminho do arquivo.")
        st.stop()

    # Lista de variáveis para seleção
    variables = [
        'systolic_mmhg', 'diastolic_mmhg', 'map_mmhg',
        'respiration_bpm', 'r3_pr_bpm_value', 'Índices de Perfusão'
    ]
    
    # Widget de seleção
    selected_var = st.selectbox(
        "Selecione a variável/grupo para análise:",
        options=variables,
        index=0,
        key="var_selector"
    )
    
    st.subheader(f"Análise: **{selected_var}**")
    
    # Layout em colunas
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Gráfico principal
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if selected_var == 'Índices de Perfusão':
            # Plotar todas as séries de perfusão juntas
            perfusion_cols = ['r1_pi_value', 'r2_pi_value', 'r3_pi_value']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Laranja, Verde
            
            for col, color in zip(perfusion_cols, colors):
                sns.lineplot(
                    x=df.index,
                    y=df[col],
                    ax=ax,
                    marker='o',
                    markersize=4,
                    linewidth=1,
                    label=col.replace('_pi_value', '').upper(),
                    color=color
                )
            
            # Configurações do gráfico
            ax.set_title("Índices de Perfusão por Observação")
            ax.set_ylabel("Valor do Índice")
            
        else:
            # Plotar variável individual
            sns.lineplot(
                x=df.index,
                y=df[selected_var],
                ax=ax,
                marker='o',
                markersize=4,
                linewidth=1,
                color='steelblue',
                label=selected_var
            )
            
            # Linha de referência para variáveis individuais
            mean_val = df[selected_var].mean()
            ax.axhline(mean_val, color='r', linestyle='--', label=f'Média: {mean_val:.2f}')
            
            # Configurações do gráfico
            ax.set_title(f"Valores de {selected_var} por Observação")
            ax.set_ylabel(selected_var)
        
        # Configurações comuns a ambos os casos
        ax.set_xlabel("Número da Observação (Índice)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Estatísticas descritivas
        st.subheader("Estatísticas Descritivas")
        
        if selected_var == 'Índices de Perfusão':
            # Mostrar estatísticas para todos os índices
            perfusion_stats = df[['r1_pi_value', 'r2_pi_value', 'r3_pi_value']].describe().T
            st.dataframe(perfusion_stats.style.format("{:.2f}"), height=250)
        else:
            # Estatísticas para variável individual
            stats = df[selected_var].describe().to_frame().T
            st.dataframe(stats.style.format("{:.2f}"), height=250)
        
        # Métricas rápidas
        st.subheader("Informações Adicionais")
        
        if selected_var == 'Índices de Perfusão':
            st.metric(label="Valores Faltantes Totais", 
                      value=df[['r1_pi_value', 'r2_pi_value', 'r3_pi_value']].isnull().sum().sum())
            
            # Criar colunas para métricas individuais
            cols = st.columns(3)
            for i, col in enumerate(['r1_pi_value', 'r2_pi_value', 'r3_pi_value']):
                with cols[i]:
                    st.metric(label=f"Média {col.replace('_pi_value', '')}", 
                             value=f"{df[col].mean():.2f}")
        else:
            st.metric(label="Valores Faltantes", value=df[selected_var].isnull().sum())
            st.metric(label="Valores Únicos", value=df[selected_var].nunique())
            st.metric(label="Média", value=f"{df[selected_var].mean():.2f}")
            st.metric(label="Mediana", value=f"{df[selected_var].median():.2f}")
        
        # Gráfico complementar
        st.subheader("Distribuição" if selected_var != 'Índices de Perfusão' else "Distribuição Comparada")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        
        if selected_var == 'Índices de Perfusão':
            # Boxplot comparado para os índices
            perfusion_df = df[['r1_pi_value', 'r2_pi_value', 'r3_pi_value']].melt(var_name='Índice', value_name='Valor')
            sns.boxplot(data=perfusion_df, x='Índice', y='Valor', ax=ax2, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax2.set_title("Distribuição dos Índices de Perfusão")
            ax2.set_ylabel("Valor")
            ax2.set_xlabel("")
        else:
            # Histograma para variável individual
            sns.histplot(df[selected_var], kde=True, bins=30, ax=ax2, color='seagreen')
            ax2.set_title(f"Distribuição de {selected_var}")
            ax2.set_xlabel(selected_var)
            ax2.set_ylabel("Frequência")
        
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard desenvolvido para análise de dados fisiológicos**  
*Laboratório de Pesquisa em Lesões Medulares*  
Versão 1.0 - Junho 2025
""")