# Guide de Configuration de la Pipeline MLflow

## 🎯 Aperçu

Ce document explique comment utiliser la pipeline MLflow pour le projet de Prédiction des Ventes Rossmann. MLflow aide à suivre les expériences, gérer les modèles et assurer la reproductibilité.

## 📁 Structure des Fichiers

```
src/pipelines/
├── mlflow_utils.py             # Fonctions MLflow principales
├── mlflow_config.py            # Paramètres de configuration
├── training_with_mlflow.py     # Pipeline d'entraînement avec MLflow
└── demo_mlflow.py              # Script de démonstration
```

## 🚀 Démarrage Rapide

### 1. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'Entraînement de Base

```python
from src.pipelines.training_with_mlflow import run_training_pipeline

# Entraîner un modèle avec logging MLflow
model = run_training_pipeline()
```

### 3. Démarrer l'Interface MLflow

```bash
mlflow ui --port 5000
```

Puis ouvrir : http://localhost:5000

### 4. Exécuter la Démonstration

```bash
cd src/pipelines
python demo_mlflow.py
```

## 📊 Fonctionnalités

### ✅ Suivi des Expériences

- **Métriques** : RMSE, MAE, R² (entraînement/test)
- **Paramètres** : Tous les hyperparamètres du modèle
- **Artefacts** : Prédictions, importance des features
- **Tags** : Métadonnées et organisation des expériences

### ✅ Gestion des Modèles

- **Logging des Modèles** : Modèles XGBoost avec signatures
- **Chargement des Modèles** : Charger des modèles depuis n'importe quel run
- **Registre des Modèles** : Contrôle de version pour les modèles de production

### ✅ Outils de Comparaison

- **Comparaison des Runs** : Comparer plusieurs expériences
- **Sélection du Meilleur Modèle** : Identification automatique du meilleur run
- **Réglage d'Hyperparamètres** : Exploration systématique des paramètres

## 🔧 Configuration

### Paramètres MLflow

La pipeline utilise ces paramètres par défaut (configurables dans `mlflow_config.py`) :

```python
EXPERIMENT_NAME = "rossmann-sales-prediction"
TRACKING_URI = "file:./mlruns"
MODEL_NAME = "rossmann_xgboost"
```

### Organisation des Expériences

| Expérience | Objectif |
|------------|----------|
| `rossmann-sales-prediction` | Expériences de production principales |
| `rossmann-sales-prediction-tuning` | Réglage d'hyperparamètres |
| `rossmann-demo` | Démonstration et tests |

## 📈 Exemples d'Utilisation

### Entraînement de Base avec Paramètres Personnalisés

```python
from src.pipelines.training_with_mlflow import run_training_pipeline

# Entraîner avec des hyperparamètres personnalisés
hyperparams = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "n_estimators": 300
}

model = run_training_pipeline(
    run_name="params_personnalises_v1",
    hyperparams=hyperparams,
    tags={"version": "v1.1", "notes": "Test d'un taux d'apprentissage plus élevé"}
)
```

### Réglage d'Hyperparamètres

```python
from src.pipelines.training_with_mlflow import run_hyperparameter_tuning

# Exécuter plusieurs expériences avec différents hyperparamètres
models = run_hyperparameter_tuning()
```

### Chargement de Modèle et Prédiction

```python
from src.pipelines.mlflow_utils import load_model_from_mlflow

# Charger le dernier modèle
model = load_model_from_mlflow("rossmann_xgboost")

# Charger une version spécifique
model = load_model_from_mlflow("rossmann_xgboost", version="1")

# Charger depuis un run spécifique
model = load_model_from_mlflow("rossmann_xgboost", run_id="abc123...")
```

### Comparer les Expériences

```python
from src.pipelines.mlflow_utils import compare_runs

# Comparer tous les runs par RMSE de test
runs_df = compare_runs(
    experiment_name="rossmann-sales-prediction",
    metric="test_rmse"
)

print("Meilleur modèle :")
print(runs_df.iloc[0])
```

## 🏷️ Registre des Modèles

### Enregistrer un Modèle

```python
from src.pipelines.mlflow_utils import register_model

# Enregistrer le meilleur modèle pour la production
model_version = register_model(
    model_name="rossmann_xgboost",
    run_id="votre_meilleur_run_id",
    registered_model_name="PredicteurVentesRossmann"
)
```

### Étapes des Modèles

Utiliser l'interface MLflow pour gérer les étapes des modèles :

- **Staging** : Pour les tests
- **Production** : Pour le déploiement
- **Archived** : Pour les anciennes versions

## 📊 Métriques Suivies

| Métrique | Description |
|----------|-------------|
| `train_rmse` | Erreur Quadratique Moyenne d'Entraînement |
| `test_rmse` | Erreur Quadratique Moyenne de Test |
| `train_mae` | Erreur Absolue Moyenne d'Entraînement |
| `test_mae` | Erreur Absolue Moyenne de Test |
| `train_r2` | Score R² d'Entraînement |
| `test_r2` | Score R² de Test |
| `overfitting_score` | Mesure du surapprentissage |

## 🏆 Bonnes Pratiques

### 1. **Nommage des Expériences**

```python
# Bon : Noms descriptifs des runs
run_name = "xgb_optimise_v2_lr_eleve"

# Mauvais : Noms génériques
run_name = "run1"
```

### 2. **Étiquetage**

```python
tags = {
    "version_donnees": "v1.2",
    "jeu_features": "engineered_v3",
    "version_modele": "xgb_optimise",
    "notes": "Correction du data leakage"
}
```

### 3. **Versioning des Modèles**

- Utiliser le versioning sémantique pour les changements majeurs
- Étiqueter les jalons importants
- Documenter les améliorations du modèle

### 4. **Gestion des Artefacts**

- Logger l'importance des features
- Sauvegarder les fichiers de prédictions
- Inclure les artefacts de préprocessing des données

## 🔍 Dépannage

### Problèmes Courants

#### 1. "Experiment not found"

```python
# Solution : Créer l'expérience d'abord
from src.pipelines.mlflow_utils import setup_mlflow
setup_mlflow("nom-de-votre-experience")
```

#### 2. "Model not found"

```python
# Solution : Vérifier le nom du modèle et l'ID du run
import mlflow
runs = mlflow.search_runs()
print(runs[['run_id', 'tags.mlflow.runName']])
```

#### 3. "Port already in use"

```bash
# Solution : Utiliser un port différent
mlflow ui --port 5001
```

## 🌟 Fonctionnalités Avancées

### Métriques Personnalisées

```python
# Ajouter des métriques personnalisées pendant l'entraînement
with mlflow.start_run():
    # ... code d'entraînement ...
    
    # Métrique métier personnalisée
    erreur_hebdomadaire = calculer_mape_hebdomadaire(y_test, y_pred)
    mlflow.log_metric("mape_hebdomadaire", erreur_hebdomadaire)
```

### Sélection Automatique de Modèle

```python
def selectionner_meilleur_modele(experiment_name, metric="test_rmse"):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_run = runs.loc[runs[f'metrics.{metric}'].idxmin()]
    return load_model_from_mlflow("rossmann_xgboost", run_id=best_run['run_id'])
```

## 📚 Ressources Supplémentaires

- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)
- [API de Suivi MLflow](https://mlflow.org/docs/latest/tracking.html)
- [Guide du Registre des Modèles](https://mlflow.org/docs/latest/model-registry.html)

---

**Bonne Expérimentation ! 🚀**