import pandas as pd
from features import load_model


model = load_model()

# Fonction de prédiction
def predict_sales(input_dict):
    try:
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        return round(prediction, 2)
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"
