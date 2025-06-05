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
tab1, tab2 = st.tabs(["Contexto e Objetivos", "Apresentação dos Dados"])

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
    **Variável target selecionada:** Pressão Arterial Sistólica  

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

    # Dicionário de mapeamento de variáveis para nomes em português
    var_names = {
        'respiration_bpm': 'Frequência Respiratória (rpm)',
        'r3_pr_bpm_value': 'Frequência Cardíaca (bpm)',
        'r1_pi_value': 'Índice de Perfusão R1',
        'r2_pi_value': 'Índice de Perfusão R2',
        'r3_pi_value': 'Índice de Perfusão R3',
        'systolic_mmhg': 'Pressão Sistólica (mmHg)',
        'diastolic_mmhg': 'Pressão Diastólica (mmHg)',
        'map_mmhg': 'Pressão Arterial Média (mmHg)'
    }

    # Lista de variáveis para seleção (com nomes em português)
    variables = [
        'Frequência Respiratória (rpm)',
        'Frequência Cardíaca (bpm)',
        'Índices de Perfusão', 
        'Pressões Arteriais'
    ]
    
    # Widget de seleção
    selected_var = st.selectbox(
        "Selecione a variável/grupo para análise:",
        options=variables,
        index=0,
        key="var_selector"
    )
    
    # Gráfico principal
    fig, ax = plt.subplots(figsize=(12, 6))
        
    if selected_var == 'Índices de Perfusão':
        # Plotar todas as séries de perfusão juntas
        perfusion_cols = ['r1_pi_value', 'r2_pi_value', 'r3_pi_value']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = [var_names[col] for col in perfusion_cols]
        
        for col, color, label in zip(perfusion_cols, colors, labels):
            sns.lineplot(
                x=df.index,
                y=df[col],
                ax=ax,
                marker='o',
                markersize=4,
                linewidth=1,
                label=label,
                color=color
            )
        
        ax.set_title("Índices de Perfusão por Observação")
        ax.set_ylabel("Valor do Índice")
        
    elif selected_var == 'Pressões Arteriais':
        # Plotar todas as pressões juntas
        pressure_cols = ['systolic_mmhg', 'diastolic_mmhg', 'map_mmhg']
        colors = ['#d62728', '#1f77b4', '#9467bd']
        labels = [var_names[col] for col in pressure_cols]
        
        for col, color, label in zip(pressure_cols, colors, labels):
            sns.lineplot(
                x=df.index,
                y=df[col],
                ax=ax,
                marker='o',
                markersize=4,
                linewidth=1.5,
                label=label,
                color=color
            )
        
        ax.set_title("Pressões Arteriais por Observação")
        ax.set_ylabel("mmHg")
        
    else:
        # Encontrar o nome original da variável selecionada
        original_var = [k for k, v in var_names.items() if v == selected_var][0]
        
        # Plotar variável individual
        sns.lineplot(
            x=df.index,
            y=df[original_var],
            ax=ax,
            marker='o',
            markersize=4,
            linewidth=1,
            color='steelblue',
            label=selected_var
        )
        
        # Linha de referência
        mean_val = df[original_var].mean()
        ax.axhline(mean_val, color='r', linestyle='--', label=f'Média: {mean_val:.2f}')
        
        ax.set_title(f"{selected_var} por Observação")
        ax.set_ylabel(selected_var.split('(')[-1].replace(')', '') if '(' in selected_var else "")
    
    ax.set_xlabel("Número da Observação (Índice)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Gráfico complementar
    st.subheader("Distribuição" if selected_var not in ['Índices de Perfusão', 'Pressões Arteriais'] else "Distribuição Comparada")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    
    if selected_var == 'Índices de Perfusão':
        perfusion_df = df[['r1_pi_value', 'r2_pi_value', 'r3_pi_value']].melt(var_name='Índice', value_name='Valor')
        perfusion_df['Índice'] = perfusion_df['Índice'].map(var_names)
        sns.boxplot(data=perfusion_df, x='Índice', y='Valor', ax=ax2, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title("Distribuição dos Índices de Perfusão")
        ax2.set_ylabel("Valor")
        ax2.set_xlabel("")
        plt.xticks(rotation=45)
        
    elif selected_var == 'Pressões Arteriais':
        pressure_df = df[['systolic_mmhg', 'diastolic_mmhg', 'map_mmhg']].melt(var_name='Pressão', value_name='Valor')
        pressure_df['Pressão'] = pressure_df['Pressão'].map(var_names)
        sns.boxplot(data=pressure_df, x='Pressão', y='Valor', ax=ax2, palette=['#d62728', '#1f77b4', '#9467bd'])
        ax2.set_title("Distribuição das Pressões Arteriais")
        ax2.set_ylabel("mmHg")
        ax2.set_xlabel("")
        plt.xticks(rotation=45)
        
    else:
        original_var = [k for k, v in var_names.items() if v == selected_var][0]
        sns.histplot(df[original_var], kde=True, bins=30, ax=ax2, color='seagreen')
        ax2.set_title(f"Distribuição de {selected_var}")
        ax2.set_xlabel(selected_var)
        ax2.set_ylabel("Frequência")
    
    st.pyplot(fig2)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard desenvolvido para análise de dados fisiológicos**  
*Laboratório de Pesquisa em Lesões Medulares*  
Versão 1.0 - Junho 2025
""")