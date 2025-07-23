# 📈 Prédiction des ventes dans les magasins Rossmann

Ce projet a pour objectif de prédire les ventes journalières des magasins Rossmann à l’aide d’un modèle **XGBoost** entraîné sur des données historiques enrichies (lags, moyennes mobiles, contexte promo…).

## 🔧 Technologies utilisées

- Python 3.10
- XGBoost
- Pandas, NumPy, Joblib
- Streamlit *(interface utilisateur)* ou Flask *(API REST)*
- Git + GitHub Actions *(CI/CD)*
- Pytest *(tests unitaires)*

---

## 📁 Structure du projet

```
├── app.py              # Point d'entrée (Streamlit ou Flask)
├── ui.py               # Interface Streamlit (si utilisé)
├── predict.py          # Fonction de prédiction
├── xgboost_model.pkl   # Modèle IA sauvegardé
├── requirements.txt    # Dépendances
├── tests/
│   └── test_predict.py # Test unitaire du modèle
└── .github/
    └── workflows/
        └── ci.yml      # Pipeline CI avec GitHub Actions
```

---

## 🚀 Démarrage rapide

### 🔹 Lancer l'app avec Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🔹 Lancer l’API Flask (alternative)

```bash
pip install -r requirements.txt
python app.py
```

---

## 🔍 Exemple de prédiction (API POST /predict)

```json
POST http://localhost:5000/predict
Content-Type: application/json

{
  "DayOfWeek": 3,
  "Promo": 1,
  "CompetitionDistance": 500,
  "Sales_lag_1": 6500,
  ...
}
```

---

## ✅ Tests unitaires

```bash
pytest tests/
```

Les tests valident que :
- La fonction `predict_sales()` retourne un float
- Le modèle est bien chargé et exploitable

---

## ⚙️ CI/CD avec GitHub Actions

- À chaque `push` sur la branche `main`, le workflow :
  - installe les dépendances
  - exécute les tests
  - bloque le déploiement si un test échoue ✅

---

## 📊 Résultats

- Modèle final : **XGBoost**
- RMSE : **153.81**
- MAE : **79.05**
- Données : +800 000 lignes, 25 features

---

## 🔒 Éthique et responsabilité

- Aucun usage de données personnelles
- Les biais potentiels liés à la saisonnalité ou à la localisation sont surveillés
- À intégrer dans un cadre gouverné et supervisé

---

## 📬 Contact

> Projet réalisé dans le cadre d’un cas d’usage IA – Rossmann  
> Par : *[Bintou N'DIAYE / GitHub / LinkedIn]*