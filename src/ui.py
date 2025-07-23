import streamlit as st
import pandas as pd
from src.features import load_model
from src.predict import predict_sales

model = load_model()

def render_form():
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            DayOfWeek = st.selectbox("Jour de la semaine (1=Lun ... 7=Dim)", list(range(1, 8)))
            Day = st.slider("Jour du mois", 1, 31, 15)
            WeekOfYear = st.slider("Semaine de l'année", 1, 52, 20)
            Promo = st.selectbox("Promo active aujourd’hui ?", [0, 1])
            Promo2Since = st.number_input("Semaines depuis Promo2", min_value=0, value=50)
            IsPromo2Active = st.selectbox("Promo2 active ?", [0, 1])
            PromoDuringHoliday = st.selectbox("Promo pendant vacances ?", [0, 1])
            IsBeforePromo = st.selectbox("Veille de promo ?", [0, 1])
            IsAfterPromo = st.selectbox("Après promo ?", [0, 1])

        with col2:
            CompetitionDistance = st.number_input("Distance concurrente (m)", min_value=0.0, value=500.0)
            CompetitionOpenSinceMonth = st.slider("Mois ouverture concurrent", 1, 12, 6)
            CompetitionOpenSince = st.number_input("Ancienneté concurrente (mois)", min_value=0, value=24)
            CompetitionIntensity = st.number_input("Intensité concurrentielle", min_value=0.0, value=12.5)
            StoreType_b = st.selectbox("Type magasin B", [0, 1])
            StoreType_c = st.selectbox("Type magasin C", [0, 1])
            StoreType_d = st.selectbox("Type magasin D", [0, 1])
            DaysSinceStart = st.number_input("Jours depuis début", min_value=0, value=1200)

        st.subheader("🔁 Ventes historiques")
        Sales_lag_1 = st.number_input("Vente J-1", value=6500)
        Sales_lag_7 = st.number_input("Vente J-7", value=7000)
        Sales_lag_14 = st.number_input("Vente J-14", value=7100)
        Sales_Mean_3 = st.number_input("Moyenne sur 3 jours", value=6600)
        Sales_Mean_7 = st.number_input("Moyenne sur 7 jours", value=6700)
        Sales_Mean_14 = st.number_input("Moyenne sur 14 jours", value=6800)
        Sales_DOW_Mean = st.number_input("Moyenne du jour (ex: lundi)", value=6900)
        Sales_DOW_Deviation = st.number_input("Écart à la moyenne du jour", value=50)

        submitted = st.form_submit_button("🎯 Prédire")

    if submitted:
        input_dict = {
            'DayOfWeek': DayOfWeek,
            'Promo': Promo,
            'CompetitionDistance': CompetitionDistance,
            'CompetitionOpenSinceMonth': CompetitionOpenSinceMonth,
            'Day': Day,
            'StoreType_b': StoreType_b,
            'StoreType_c': StoreType_c,
            'StoreType_d': StoreType_d,
            'WeekOfYear': WeekOfYear,
            'Promo2Since': Promo2Since,
            'CompetitionOpenSince': CompetitionOpenSince,
            'DaysSinceStart': DaysSinceStart,
            'IsPromo2Active': IsPromo2Active,
            'Sales_lag_1': Sales_lag_1,
            'Sales_lag_7': Sales_lag_7,
            'Sales_lag_14': Sales_lag_14,
            'Sales_Mean_3': Sales_Mean_3,
            'Sales_Mean_7': Sales_Mean_7,
            'Sales_Mean_14': Sales_Mean_14,
            'CompetitionIntensity': CompetitionIntensity,
            'PromoDuringHoliday': PromoDuringHoliday,
            'IsAfterPromo': IsAfterPromo,
            'IsBeforePromo': IsBeforePromo,
            'Sales_DOW_Mean': Sales_DOW_Mean,
            'Sales_DOW_Deviation': Sales_DOW_Deviation
        }

        result = predict_sales(input_dict)
        if isinstance(result, float):
            st.success(f"💸 Vente estimée : **{result} €**")
        else:
            st.error(result)