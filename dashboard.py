import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import joblib
from PIL import Image

import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

# Configurações gerais
st.set_page_config(page_title="Análise Fisiológica - Lesão Medular", layout="wide")

# Função para carregar dados
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Título principal
st.title("Dashboard de Análise de Parâmetros Fisiológicos em Modelo Animal")



try:
    dados1 = pd.read_csv('vitals_mais_rs.csv', sep=',', encoding='utf-8')
    dados1 = dados1.drop(columns=['r1_events', 'r2_events', 'r3_events'])
    dados1 = dados1.rename(columns={'r3_pr_bpm_value':'pr_bpm'})

    dados2 = pd.read_csv('vitals_mais_rs_2.csv', sep=',', encoding='utf-8')
    dados2 = dados2.drop(columns=['r1_events', 'r2_events', 'r3_events'])
    dados2 = dados2.rename(columns={'pr_bpm_value':'pr_bpm'})

    dados = pd.concat([dados1, dados2], ignore_index=True)


    dados = dados.rename(columns={
        'pr_bpm': 'fc_bpm',
        'r1_pi_value': 'r1_ip',
        'r2_pi_value': 'r2_ip',
        'r3_pi_value': 'r3_ip',
        'systolic_mmhg': 'sistolica_mmhg'
    })


    dados = dados.loc[dados['sistolica_mmhg'] < 201]

    # Remover registros com índices de perfusão (r1_ip, r2_ip, r3_ip) maiores que 5
    dados = dados.loc[
        (dados['r1_ip'] <= 5) &
        (dados['r2_ip'] <= 5) &
        (dados['r3_ip'] <= 5)
    ]

    # Resetar o índice para começar em 1
    dados.index = dados.index + 1
except:
    st.error("Arquivo não encontrado! Por favor, verifique o caminho do arquivo.")
    st.stop()



# Abas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Contexto e Objetivos",
    "Apresentação dos Dados",
    "Classificador de Pressão Arterial Sistólica",
    "Modelo Preditivo da Pressão Arterial Sistólica",
    "Conclusões" 
])

with tab1:
    st.header("Contexto do Estudo")
    
    st.image("imagem_choque_neuro.png", caption="Representação do choque neurogênico", width=1000)

    st.markdown("""
    **Fonte dos dados:**  
    Estudo realizado em macacos com lesão medular, com dados coletados por 4 diferentes aparelhos
    que posteriormente foram consolidados em um único conjunto de dados.

    **Parâmetros monitorados:**
    - Pressão arterial (sistólica, diastólica e média) = força exercida pelo sangue contra as paredes das artérias
    - Índices de perfusão (R1, R2, R3) = medida que reflete a qualidade da circulação sanguínea, especialmente em regiões periféricas do corpo
    - Pulsação cardíaca (R3) = número de batimentos do coração por minuto
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

    # Dicionário de mapeamento de variáveis para nomes em português
    var_names = {
        'fc_bpm': 'Frequência Cardíaca (bpm)',
        'r1_ip': 'Índice de Perfusão R1',
        'r2_ip': 'Índice de Perfusão R2',
        'r3_ip': 'Índice de Perfusão R3',
        'sistolica_mmhg': 'Pressão Sistólica (mmHg)',
        'diastolic_mmhg': 'Pressão Diastólica (mmHg)',
        'map_mmhg': 'Pressão Arterial Média (mmHg)'
    }

    # Lista de variáveis para seleção (com nomes em português)
    variables = [
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
        perfusion_cols = ['r1_ip', 'r2_ip', 'r3_ip']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = [var_names[col] for col in perfusion_cols]
        
        for col, color, label in zip(perfusion_cols, colors, labels):
            sns.lineplot(
                x=dados.index,
                y=dados[col],
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
        pressure_cols = ['sistolica_mmhg', 'diastolic_mmhg', 'map_mmhg']
        colors = ['#d62728', '#1f77b4', '#9467bd']
        labels = [var_names[col] for col in pressure_cols]
        
        for col, color, label in zip(pressure_cols, colors, labels):
            sns.lineplot(
                x=dados.index,
                y=dados[col],
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
            x=dados.index,
            y=dados[original_var],
            ax=ax,
            marker='o',
            markersize=4,
            linewidth=1,
            color='steelblue',
            label=selected_var
        )
        
        # Linha de referência
        mean_val = dados[original_var].mean()
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
        perfusion_dados = dados[['r1_ip', 'r2_ip', 'r3_ip']].melt(var_name='Índice', value_name='Valor')
        perfusion_dados['Índice'] = perfusion_dados['Índice'].map(var_names)
        sns.boxplot(data=perfusion_dados, x='Índice', y='Valor', ax=ax2, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title("Distribuição dos Índices de Perfusão")
        ax2.set_ylabel("Valor")
        ax2.set_xlabel("")
        plt.xticks(rotation=45)
        
    elif selected_var == 'Pressões Arteriais':
        pressure_dados = dados[['sistolica_mmhg', 'diastolic_mmhg', 'map_mmhg']].melt(var_name='Pressão', value_name='Valor')
        pressure_dados['Pressão'] = pressure_dados['Pressão'].map(var_names)
        sns.boxplot(data=pressure_dados, x='Pressão', y='Valor', ax=ax2, palette=['#d62728', '#1f77b4', '#9467bd'])
        ax2.set_title("Distribuição das Pressões Arteriais")
        ax2.set_ylabel("mmHg")
        ax2.set_xlabel("")
        plt.xticks(rotation=45)
        
    else:
        original_var = [k for k, v in var_names.items() if v == selected_var][0]
        sns.histplot(dados[original_var], kde=True, bins=30, ax=ax2, color='seagreen')
        ax2.set_title(f"Distribuição de {selected_var}")
        ax2.set_xlabel(selected_var)
        ax2.set_ylabel("Frequência")
    
    st.pyplot(fig2)

with tab3:
        # Carrega o scaler e o modelo
    scaler_arvore = joblib.load('scaler_arvore.pkl')
    modelo_arvore = joblib.load('modelo_arvore.pkl')

    mapa_classes = {
        0: 'Hipotensão',
        1: 'Normotenso',
        2: 'Hipertensão'
    }


    # Carregar modelo salvo
    modelo: DecisionTreeClassifier = joblib.load("modelo_arvore.pkl")

    # Checar se tem feature_names e class_names no modelo
    feature_names = getattr(modelo, 'feature_names_in_', [f'feature_{i}' for i in range(modelo.n_features_in_)])
    class_names = getattr(modelo, 'classes_', [str(i) for i in range(len(modelo.classes_))])

    # Função auxiliar para organizar a posição dos nós
    def hierarchy_pos(tree_, node_id=0, x=0., y=0., dx=1.):
        children_left = tree_.children_left
        children_right = tree_.children_right

        def _hierarchy_pos(node_id, x, y, dx):
            if children_left[node_id] == children_right[node_id]:
                return {node_id: (x, y)}
            else:
                left = _hierarchy_pos(children_left[node_id], x - dx, y - 1, dx / 2)
                right = _hierarchy_pos(children_right[node_id], x + dx, y - 1, dx / 2)
                return {node_id: (x, y), **left, **right}

        return _hierarchy_pos(node_id, x, y, dx)

    # Função para criar o gráfico interativo com Plotly
    def plot_tree_interactive(clf, feature_names, class_names):
        tree_ = clf.tree_
        labels = []

        for i in range(tree_.node_count):
            if tree_.children_left[i] == tree_.children_right[i]:
                label = f"Leaf: class={np.argmax(tree_.value[i])}\nsamples={tree_.n_node_samples[i]}"
            else:
                label = (f"{feature_names[tree_.feature[i]]} <= {tree_.threshold[i]:.2f}\n"
                        f"samples={tree_.n_node_samples[i]}")
            labels.append(label)

        pos = hierarchy_pos(tree_)
        xs = [pos[i][0] for i in range(len(pos))]
        ys = [pos[i][1] for i in range(len(pos))]

        edge_x, edge_y = [], []
        for i in range(len(labels)):
            if tree_.children_left[i] != tree_.children_right[i]:
                for child in [tree_.children_left[i], tree_.children_right[i]]:
                    edge_x += [pos[i][0], pos[child][0], None]
                    edge_y += [pos[i][1], pos[child][1], None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            text=labels,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='lightblue',
                size=30,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Árvore de Decisão Interativa (Plotly)',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                        ))
        return fig




    st.title('Classificador de Pressão Arterial Sistólica')



    # 1. Exibir a imagem do agrupamento
    st.header("Matriz de Agrupamento")
    st.image("imagem_agrupamento.png", caption="Agrupamento de dados", use_container_width=True)

    # 2. Introdução à árvore de decisão
    st.header("Introdução à Árvore de Decisão")
    st.markdown("""
    Uma **árvore de decisão** é um algoritmo de aprendizado supervisionado utilizado para problemas de classificação e regressão.
    Ela funciona dividindo os dados em subconjuntos com base em atributos que maximizam a separação entre as classes.

    Cada nó da árvore representa uma decisão baseada em um atributo, e as folhas representam os rótulos finais de classificação.
    É uma técnica simples, porém poderosa, com ótima capacidade de interpretação.
    """)

    # 3. Mostrar o reporte da classificação e matriz de confusão
    st.header("Resultados da Classificação")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Relatório de Classificação")
        st.image("reporte_classificacao.png", caption="Métricas de avaliação", use_container_width=True)

    with col2:
        st.subheader("Matriz de Confusão")
        st.image("matriz_classificacao.png", caption="Desempenho do classificador", use_container_width=True)

    # 4. Exibir o gráfico da árvore
    st.header("Visualização da Árvore de Decisão")

    fig = plot_tree_interactive(modelo, feature_names, class_names)
    st.plotly_chart(fig, use_container_width=True)


    # Entradas do usuário
    pr_bpm = st.number_input('Frequência Cardíaca (bpm)', min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f", key="pr_bpm_tab3")
    r1_pi = st.number_input('R1 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.2f", key="r1_pi_tab3")
    r2_pi = st.number_input('R2 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.1, step=0.1, format="%.2f", key="r2_pi_tab3")
    r3_pi = st.number_input('R3 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f", key="r3_pi_tab3")
    # Quando o botão for clicado
    if st.button('Classificar'):
        # Dados de entrada do usuário (exemplo)
        valores = [[pr_bpm, r1_pi, r2_pi, r3_pi]]

        # Usa os mesmos nomes de colunas que no treino
        colunas = ['fc_bpm', 'r1_ip', 'r2_ip', 'r3_ip']
        df_input = pd.DataFrame(valores, columns=colunas)

        # Aplica o scaler e o modelo
        # dados_escalados = scaler.transform(df_input)
        predicao = modelo_arvore.predict(df_input)[0]
        
        # Exibe a resposta mapeada
        st.success(f'Resultado: {mapa_classes[predicao]}')

with tab4:
    # -----------------------
    # Título e imagem introdutória
    # -----------------------
    st.title('🧠 Modelo Preditivo da Pressão Arterial Sistólica com XGBoost')
    st.image('imagem_xgb.png', caption='Explicação do algoritmo XGBoost', use_container_width=True)

    # -----------------------
    # Carregamento do modelo
    # -----------------------
    scaler_regressao = joblib.load('scaler_regressao.pkl')
    modelo_xgb = joblib.load('modelo_xgb.pkl')

    # -----------------------
    # Interface de entrada
    # -----------------------

    # -----------------------
    # Tabela com métricas
    # -----------------------
    st.subheader("📈 Avaliação do Modelo de Regressão")
    try:
        tabela_metricas = pd.read_csv('metricas_modelo_regressao.csv')
        st.table(tabela_metricas.style.format({'Valor': '{:.2f}'}))
    except FileNotFoundError:
        st.warning("Arquivo 'metricas_modelo_regressao.csv' não encontrado.")

    # -----------------------
    # Imagens Real vs Predito e Erros
    # -----------------------
    st.subheader("🔎 Análise de Desempenho")

    col1, col2 = st.columns(2)
    with col1:
        st.image("imagem_real_vs_pred_regressao.png", caption="Valores Reais vs Preditos", use_container_width=True)
    with col2:
        st.image("imagem_histograma_erros_regressao.png", caption="Distribuição dos Erros", use_container_width=True)

    # -----------------------
    # Imagens de Importância e Efeitos das Variáveis
    # -----------------------
    st.subheader("🧬 Interpretação do Modelo")

    col3, col4 = st.columns(2)
    with col3:
        st.image("imagem_importancia_features_regressao.png", caption="Importância das Variáveis", use_container_width=True)
    with col4:
        st.image("imagem_efeito_features_regressao.png", caption="Efeito das Variáveis", use_container_width=True)

    # -----------------------
    # Código fonte (opcional)
    # -----------------------
    st.subheader("💻 Implementação do Modelo ")


    pr_bpm = st.number_input('Frequência Cardíaca (bpm)', min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f", key="pr_bpm_tab4")
    r1_pi = st.number_input('R1 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.2f", key="r1_pi_tab4")
    r2_pi = st.number_input('R2 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.1, step=0.1, format="%.2f", key="r2_pi_tab4")
    r3_pi = st.number_input('R3 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f", key="r3_pi_tab4")
    # -----------------------
    # Previsão
    # -----------------------
    if st.button('🔍 Prever Pressão Sistólica'):
        valores = [[pr_bpm, r1_pi, r2_pi, r3_pi]]
        colunas = ['fc_bpm', 'r1_ip', 'r2_ip', 'r3_ip']
        df_input = pd.DataFrame(valores, columns=colunas)

        try:
            dados_escalados = scaler_regressao.transform(df_input)
            predicao = round(modelo_xgb.predict(dados_escalados)[0])
            st.success(f'🩺 Pressão Sistólica Prevista: **{predicao} mmHg**')
        except Exception as e:
            st.error(f'Erro ao fazer previsão: {e}')

with tab5:
    st.header("Conclusões")
    st.markdown("""
- O modelo indica que os índices de perfusão e de frequência cardíaca podem ser variáveis preditoras da pressão arterial sistólica;
- É possível reduzir equipamentos de monitoramento dos parâmetros fisiológicos, consequentemente isso reduz os custos de manutenção, operação e armazenamento do equipamento;
- Modelos apresentados podem ser usados para traçar um prognóstico em paciente com LM com base na previsão de parâmetros fisiológicos (FC e IP) que tenham influência no trauma medular.
    """)



# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard desenvolvido para análise de dados fisiológicos**  
*Laboratório de Pesquisa em Lesões Medulares*  
Versão 1.0 - Junho 2025
""")
