# 📈 Prédiction des Ventes Rossmann avec MLflow

Ce projet implémente une solution complète de prédiction des ventes journalières pour les magasins Rossmann. Il utilise un modèle **XGBoost** entraîné sur des données historiques enrichies avec **MLflow** pour le suivi des expériences et la gestion des modèles.

## 🎯 Objectifs

- 📊 **Prédire** les ventes journalières avec précision
- 🔬 **Suivre** les expériences avec MLflow
- 🚀 **Déployer** des modèles en production
- 📈 **Optimiser** les performances avec le tuning d'hyperparamètres
- 🔄 **Automatiser** le pipeline ML de bout en bout

## 🔧 Stack Technologique

- **Python 3.10+**
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Experiment Tracking**: MLflow
- **Web Framework**: Streamlit, Flask
- **Model Persistence**: Joblib
- **Development**: Pytest, Git
- **CI/CD**: GitHub Actions

---

## 📁 Structure du projet

```text
├── 📊 data/
│   ├── train.csv                    # Données d'entraînement
│   ├── test.csv                     # Données de test
│   └── store.csv                    # Informations magasins
├── 📋 Configuration/
│   ├── requirements.txt             # Dépendances Python
│   ├── .gitignore                   # Fichiers à ignorer
│   └── MLflow_README.md             # Guide MLflow complet
├── 🧠 models/
│   ├── xgboost_model.pkl           # Modèle principal
│   └── xgboost_model1.pkl          # Modèle alternatif
├── 📚 src/
│   ├── features.py                 # Feature engineering
│   ├── predict.py                  # Prédictions
│   ├── api.py                      # API REST
│   ├── ui.py                       # Interface Streamlit
│   ├── pipelines/
│   │   ├── mlflow_utils.py         # Utilitaires MLflow
│   │   ├── mlflow_config.py        # Configuration MLflow
│   │   ├── training_with_mlflow.py # Pipeline d'entraînement
│   │   ├── demo_mlflow.py          # Démonstration MLflow
│   │   └── training.py             # Entraînement traditionnel
│   └── tests/
│       └── test_predict.py         # Tests unitaires
├── 📓 notebooks/
│   ├── 01_exploration.ipynb        # Exploration des données
│   ├── rossmann_exploration.ipynb  # Analyse détaillée
│   └── rossmann_model_comparison.ipynb  # Comparaison modèles
├── 🗂️ mlruns/                      # Expériences MLflow
├── 📤 outputs/                     # Sorties temporaires
├── 🚀 Deployment/
│   ├── app.py                      # Point d'entrée principal
│   └── test_mlflow_pipeline.py     # Test de la pipeline
└── 📖 Documentation/
    ├── MLFLOW_SETUP_COMPLETE.md    # Guide de configuration
    └── src/pipelines/MLflow_README.md  # Documentation MLflow
```

---

## 🚀 Guide de Démarrage

### 1. 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/BintouNdy/rossmann-sales-predictor.git
cd rossmann-sales-predictor

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. 🧪 Test de la Pipeline MLflow

```bash
# Tester la configuration MLflow
python test_mlflow_pipeline.py
```

### 3. 🎓 Entraîner un Modèle

```bash
# Entraînement avec MLflow (recommandé)
python src/pipelines/training_with_mlflow.py

# Démonstration complète
python src/pipelines/demo_mlflow.py

# Réglage d'hyperparamètres
python -c "
import sys; sys.path.append('src')
from pipelines.training_with_mlflow import run_hyperparameter_tuning
run_hyperparameter_tuning()
"
```

### 4. 🌐 Interface MLflow

```bash
# Démarrer l'interface web MLflow
mlflow ui --port 5000

# Ouvrir dans le navigateur: http://localhost:5000
```

### 5. 🎨 Applications de Démonstration

#### Interface Streamlit

```bash
streamlit run app.py
```

#### API Flask

```bash
python src/api.py
# API disponible sur: http://localhost:5000
```

---

## 🔍 Fonctionnalités MLflow

### 📊 Suivi des Expériences

- **Métriques automatiques**: RMSE, MAE, R² (train/test)
- **Hyperparamètres**: Tous les paramètres XGBoost
- **Artefacts**: Prédictions, importance des features
- **Tags**: Métadonnées et organisation

### 🏷️ Gestion des Modèles

```python
# Charger le meilleur modèle
from src.pipelines.mlflow_utils import load_model_from_mlflow
model = load_model_from_mlflow("rossmann_xgboost")

# Comparer les expériences
from src.pipelines.mlflow_utils import compare_runs
runs_df = compare_runs("rossmann-sales-prediction", "test_rmse")
```

### 🔧 Hyperparameter Tuning

Le projet inclut plusieurs configurations prêtes à tester :

- **Baseline**: Paramètres par défaut
- **High Learning Rate**: Apprentissage accéléré  
- **Deep Trees**: Arbres plus profonds
- **Conservative**: Approche conservatrice
- **Aggressive Regularization**: Régularisation forte

---

## 🎯 Exemple d'Utilisation API

### Prédiction via API REST

```bash
# POST /predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "DayOfWeek": 3,
    "Promo": 1,
    "Day": 15,
    "WeekOfYear": 32,
    "Sales_Mean_3": 6500.0,
    "Sales_Mean_7": 6200.0,
    "Sales_Mean_14": 6000.0,
    "CompetitionIntensity": 0.002,
    "Sales_DOW_Mean": 6800.0,
    "Sales_DOW_Deviation": -300.0
  }'
```

### Réponse Attendue

```json
{
  "prediction": 6425.67,
  "model_version": "rossmann_xgboost_v1.0",
  "features_used": 10,
  "prediction_date": "2025-07-24T14:30:00Z"
}
```

---

## ✅ Tests et Qualité

### Tests Unitaires

```bash
# Exécuter tous les tests
pytest src/tests/ -v

# Test avec couverture
pytest src/tests/ --cov=src --cov-report=html
```

### Validation du Modèle

Les tests vérifient :

- ✅ Chargement correct du modèle
- ✅ Format des prédictions (float)
- ✅ Cohérence des features
- ✅ Performance minimum (RMSE < 200)

---

## 📈 Performances du Modèle

### Métriques Actuelles

| Métrique | Entraînement | Test | Description |
|----------|-------------|------|-------------|
| **RMSE** | 118.82 | 157.76 | Erreur quadratique moyenne |
| **R²** | 0.999 | 0.997 | Coefficient de détermination |
| **MAE** | ~80 | ~100 | Erreur absolue moyenne |

### Données

- **📊 Échantillons**: 663,917 (train) + 165,980 (test)
- **🔢 Features**: 10 features sélectionnées
- **📅 Période**: Données historiques Rossmann
- **🎯 Cible**: Ventes journalières par magasin

---

## 🔄 Pipeline CI/CD

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test MLflow Pipeline
        run: python test_mlflow_pipeline.py
      - name: Run Unit Tests
        run: pytest src/tests/ -v
```

### Validation Continue

- ✅ **Tests automatiques** à chaque push
- ✅ **Validation des modèles** avant déploiement
- ✅ **Contrôle qualité** du code
- ✅ **Sécurité** des dépendances

---

## 🔒 Sécurité et Conformité

### Données

- ✅ **Aucune donnée personnelle** dans le dataset
- ✅ **Anonymisation** des identifiants magasins
- ✅ **Chiffrement** des artefacts sensibles

### Biais et Éthique

- ⚠️ **Saisonnalité**: Surveillance des biais temporels
- ⚠️ **Géographie**: Attention aux biais régionaux
- ⚠️ **Équité**: Validation sur différents segments

### Gouvernance

- 📋 **Traçabilité** complète via MLflow
- 📋 **Versioning** des modèles et données
- 📋 **Audit trail** des prédictions

---

## 📚 Documentation Complète

### Guides Disponibles

- 📖 **[Guide MLflow Complet](src/pipelines/MLflow_README.md)** - Configuration et utilisation
- 📖 **[Configuration Terminée](MLFLOW_SETUP_COMPLETE.md)** - Résumé de l'installation
- 📖 **[Notebooks d'Exploration](notebooks/)** - Analyse des données

### Ressources Externes

- [Documentation MLflow](https://mlflow.org/docs/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🚀 Roadmap et Améliorations

### Prochaines Étapes

- [ ] **Déploiement Cloud** (Azure/AWS)
- [ ] **API Authentication** et rate limiting
- [ ] **Monitoring en temps réel** des prédictions
- [ ] **A/B Testing** de modèles
- [ ] **Feature Store** centralisé
- [ ] **AutoML** avec MLflow

### Optimisations

- [ ] **Feature Engineering** automatisé
- [ ] **Hyperparameter Optimization** avec Optuna
- [ ] **Model Ensemble** et stacking
- [ ] **Real-time Serving** avec MLflow
- [ ] **Data Drift Detection**

---

## 🤝 Contribution

### Comment Contribuer

1. **Fork** le repository
2. **Créer** une branche feature (`git checkout -b feature/amazing-feature`)
3. **Committer** les changements (`git commit -m 'Add amazing feature'`)
4. **Pousser** vers la branche (`git push origin feature/amazing-feature`)
5. **Ouvrir** une Pull Request

### Standards de Code

- ✅ **PEP 8** pour le style Python
- ✅ **Type hints** recommandées
- ✅ **Docstrings** pour les fonctions
- ✅ **Tests unitaires** obligatoires

---

## 📬 Contact et Support

### Auteur

**Bintou N'DIAYE**

- 💼 LinkedIn: [linkedin.com/in/bintou-ndiaye](https://www.linkedin.com/in/bintou-n-diaye-078697107/)
- 🐙 GitHub: [github.com/BintouNdy](https://github.com/BintouNdy)

### Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/BintouNdy/rossmann-sales-predictor/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/BintouNdy/rossmann-sales-predictor/discussions)
- 📖 **Documentation**: Voir les guides dans le repository

---

**Projet réalisé dans le cadre d'une validation de compétences**

*Dernière mise à jour: Juillet 2025*
