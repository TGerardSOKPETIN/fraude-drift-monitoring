# fraude-drift-monitoring

Surveillance et détection de dérive (*data drift*, *concept drift*) d'un modèle de détection de fraude financière en environnement bancaire. Comparaison méthodologique de trois outils open source de monitoring de modèles en production.

## Objectif

Comparer trois outils open source de détection de drift, **Evidently AI**, **Alibi-Detect** et **NannyML**, sur les données Poundbank, en reprenant les critères d'évaluation d'un benchmark existant (D3Bench), puis proposer une architecture de monitoring adaptée à un contexte de production (batch puis temps réel).

## Résultats clés

L'analyse comparative complète est détaillée dans [`docs/analyse-comparative-outils-drift.md`](docs/analyse-comparative-outils-drift.md). Points principaux :

- Les trois outils ne s'accordent pas sur les mêmes variables, pour des raisons méthodologiques précises : Alibi-Detect teste la significativité statistique sur l'ensemble de la période (sensible à la taille d'échantillon), Evidently mesure une magnitude moyennée sur toute la période, NannyML découpe le temps en segments et révèle la dynamique que les deux premiers ne peuvent pas voir.
- Le découpage temporel de NannyML montre une dérive progressive de `time_since_login_min`, qui s'installe dès mars 2019 et s'intensifie en fin de période, invisible dans un test global.
- L'estimation de performance par confiance (CBPE) montre une dégradation simultanée de ROC AUC, Precision, Recall et F1 à partir d'avril 2019, soit environ deux mois avant que les alertes de drift univarié par seuil simple ne se déclenchent.
- Ce rapprochement temporel, présenté comme une hypothèse argumentée et non une preuve causale, structure la stratégie d'alerte et de réentraînement proposée dans [`docs/architecture-monitoring-alertes-retraining.md`](docs/architecture-monitoring-alertes-retraining.md).

## Structure du repo

```
fraude-drift-monitoring/
├── data/                       # reference.csv et analysis.csv à placer ici (non versionnés, voir data/README.md)
├── src/
│   ├── common/                 # chargement des données, définition des colonnes
│   ├── drift_evidently/        # détection de drift avec Evidently AI
│   ├── drift_alibi/            # détection de drift avec Alibi-Detect
│   └── drift_nannyml/          # détection de drift + estimation de performance (CBPE) avec NannyML
├── reports/                    # rapports générés par les scripts (HTML/JSON)
├── exploration_drift_detection.ipynb   # notebook d'exploration et de comparaison, EDA incluse
└── docs/
    ├── etat-de-l-art.md                              # plan du volet théorique (data/concept/label drift, cycle de vie ML)
    ├── analyse-comparative-outils-drift.md           # analyse comparative rigoureuse des trois outils
    ├── architecture-monitoring-alertes-retraining.md # stratégie d'alerte et de réentraînement
    ├── architecture-cible.md                         # trajectoire batch → architecture temps réel (Kafka/Flink)
    └── exploration_drift_detection.html              # export statique du notebook (lecture rapide, sans Jupyter)
```

## Données

Le dataset Poundbank comprend deux fichiers de même structure :

| Fichier | Rôle | Période |
|---|---|---|
| `reference.csv` | Données de référence (baseline) | Janvier – Octobre 2018 |
| `analysis.csv` | Données de production à surveiller | Novembre 2018 – Juin 2019 |

Colonnes communes : `timestamp`, `time_since_login_min`, `transaction_amount`, `transaction_type`, `is_first_transaction`, `user_tenure_months`, `is_fraud`, `predicted_fraud_proba`, `predicted_fraud`.

Détails complets dans [`data/README.md`](data/README.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Utilisation

Chaque outil s'exécute indépendamment en mode batch, une fois `reference.csv` et `analysis.csv` placés dans `data/` :

```bash
python -m src.drift_evidently.run_drift_evidently
python -m src.drift_alibi.run_drift_alibi
python -m src.drift_nannyml.run_drift_nannyml
```

Les rapports générés (HTML pour Evidently, JSON/CSV pour les trois) sont écrits dans `reports/`.

Le notebook `exploration_drift_detection.ipynb` regroupe l'analyse exploratoire, l'exécution des trois outils et la comparaison des résultats dans un seul document reproductible.

## Documentation

- [Notebook complet (nbviewer)](https://nbviewer.org/github/TGerardSOKPETIN/fraude-drift-monitoring/blob/main/exploration_drift_detection.ipynb) — rendu fiable pour un fichier volumineux, à privilégier sur l'aperçu GitHub natif
- [Notebook en aperçu statique (HTML)](https://htmlpreview.github.io/?https://github.com/TGerardSOKPETIN/fraude-drift-monitoring/blob/main/docs/exploration_drift_detection.html) — ouverture instantanée, sans dépendance à Jupyter
- [Analyse comparative des trois outils](docs/analyse-comparative-outils-drift.md)
- [Stratégie d'alerte et de réentraînement](docs/architecture-monitoring-alertes-retraining.md)
- [Architecture cible (batch → temps réel)](docs/architecture-cible.md)
- [Cadrage théorique](docs/etat-de-l-art.md)
