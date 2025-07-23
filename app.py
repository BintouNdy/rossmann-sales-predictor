import streamlit as st
from src.ui import render_form

st.set_page_config(page_title="Prédiction des ventes Rossmann", page_icon="📈", layout="centered")
st.title("📈 Prédiction des ventes Rossmann")
st.markdown("Bienvenue sur l'outil de prédiction des ventes basé sur XGBoost.")

render_form()