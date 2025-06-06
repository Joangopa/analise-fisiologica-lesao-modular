import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image


import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier



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
pr_bpm = st.number_input('Frequência Cardíaca (bpm)', min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
r1_pi = st.number_input('R1 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.2f")
r2_pi = st.number_input('R2 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.1, step=0.1, format="%.2f")
r3_pi = st.number_input('R3 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")

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

