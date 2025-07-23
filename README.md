# Rossmann Sales Predictor

Ce projet vise à prédire les ventes quotidiennes des magasins Rossmann à l'aide de données historiques, en intégrant des facteurs comme les promotions, jours fériés, types de magasins et plus encore.

---

## Objectif

Aider Rossmann à :

* Mieux anticiper les ventes.
* Optimiser les niveaux de stocks.
* Ajuster les campagnes promotionnelles.
* Réduire les pertes liées à l'inventaire.

---

## Structure du projet

```
rossmann-sales-predictor/
│
├── data/                   # Données brutes (train.csv, test.csv, store.csv)
├── notebooks/              # Analyses exploratoires Jupyter
├── outputs/                # Graphiques et prédictions
├── src/                    # Code source modulaire
│   ├── data_loader.py      # Chargement des CSV
│   ├── cleaning.py         # Nettoyage des données
│   ├── features.py         # Création de variables
│   ├── visualization.py    # Graphiques exploratoires
│   └── model.py            # (future) Entraînement ML
├── main.py                 # Script d'orchestration principal
├── requirements.txt        # Dépendances Python
└── README.md
```

---

## Installation

1. Clonez le repo :

```bash
git clone https://github.com/BintouNdy/rossmann-sales-predictor.git
cd rossmann-sales-predictor
```

2. Créez un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

---

## Utilisation

Exécutez le script principal :

```bash
python main.py
```

Ou ouvrez les notebooks dans `notebooks/` pour une exploration interactive.

---

## Modèle à venir

Le modèle de prédiction utilisera :

* RandomForestRegressor
* Validation croisée
* Optimisation via GridSearchCV

---

## Visualisations incluses

* Ventes par jour de la semaine
* Impact des promotions
* Corrélation entre variables
* Analyse par type de magasin et assortiment

---

## TODO

* [x] Nettoyage des données
* [x] Exploration visuelle
* [x] Feature engineering
* [ ] Entraînement du modèle
* [ ] Soumission Kaggle
* [ ] Dashboard interactif (facultatif)

---

## Licence

Projet à but pédagogique dans le cadre de l'Épreuve 1 IA – Prédiction des ventes Rossmann.
