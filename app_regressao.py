import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image

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


pr_bpm = st.number_input('Frequência Cardíaca (bpm)', min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
r1_pi = st.number_input('R1 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.2f")
r2_pi = st.number_input('R2 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.1, step=0.1, format="%.2f")
r3_pi = st.number_input('R3 Índice de Perfusão', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")

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
