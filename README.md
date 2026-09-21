# fraude-drift-monitoring

Surveillance et détection de dérive (data drift, concept drift) d'un modèle de détection de fraude financière en environnement bancaire. Projet réalisé dans le cadre d'un stage / mémoire encadré par M. Mihoubi, basé sur le cas Poundbank (projet DataCamp).

## Objectif

Comparer trois outils open source de détection de drift (Evidently AI, Alibi-Detect, NannyML) sur les données Poundbank, en reprenant les critères d'évaluation d'un benchmark existant (D3Bench), puis faire évoluer le prototype vers une architecture de monitoring temps réel.

## Structure du repo

```
fraude-drift-monitoring/
├── data/                   # reference.csv et analysis.csv à placer ici (non versionnés, voir data/README.md)
├── src/
│   ├── common/              # chargement des données, définition des colonnes
│   ├── drift_evidently/     # détection de drift avec Evidently AI
│   ├── drift_alibi/         # détection de drift avec Alibi-Detect
│   └── drift_nannyml/       # détection de drift + estimation de performance (CBPE) avec NannyML
├── reports/                 # rapports générés par les scripts (HTML/JSON), non versionnés
└── docs/
    ├── etat-de-l-art.md       # plan du volet théorique (data/concept/label drift, cycle de vie ML)
    └── architecture-cible.md  # trajectoire batch → architecture temps réel (Kafka/Flink)
```

## Données

Le dataset Poundbank comprend deux fichiers de même structure : `reference.csv` (données de référence) et `analysis.csv` (données de production). Colonnes : `timestamp`, `time_since_login_min`, `transaction_amount`, `transaction_type`, `is_first_transaction`, `user_tenure_months`, `is_fraud`, `predicted_fraud_proba`, `predicted_fraud`. Voir `data/README.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Lancer les scripts

Chaque outil s'exécute indépendamment en mode batch, une fois `reference.csv` et `analysis.csv` placés dans `data/` :

```bash
python -m src.drift_evidently.run_drift_evidently
python -m src.drift_alibi.run_drift_alibi
python -m src.drift_nannyml.run_drift_nannyml
```

Les rapports générés (HTML pour Evidently, JSON pour les trois) sont écrits dans `reports/`.

## État d'avancement

Prototype batch en cours de construction (première version demandée pour le workshop suivant). Étape suivante : réflexion sur une architecture temps réel (streaming) une fois les trois scripts validés.
