# Architecture cible — du prototype batch au temps réel (plan)

Notes de cadrage suite au Workshop 1, à affiner avec M. Mihoubi et Hichem Debbi.

## Étape 1 — Prototype batch (en cours)

- Scripts indépendants par outil (`src/drift_evidently`, `src/drift_alibi`, `src/drift_nannyml`), exécutés manuellement.
- Objectif : démontrer les résultats et comparer les trois outils sur les données Poundbank.

## Étape 2 — Benchmark structuré

- Reprendre les critères d'évaluation de D3Bench (aptitude fonctionnelle, intégrabilité, flexibilité, facilité d'utilisation, performance) et les appliquer aux résultats obtenus à l'étape 1.

## Étape 3 — Architecture événementielle temps réel (piste, non démarrée)

- Latence de détection jugée critique en fraude bancaire (approche batch insuffisante à long terme).
- Technologies évoquées en workshop : Kafka (collecte/transport des événements de transaction), Flink (traitement de flux, calcul de drift en continu).
- Composants à esquisser : ingestion des événements, calcul de drift en streaming, déclenchement d'alertes, automatisation des actions correctives (retraining).
- Question ouverte : mécanismes de retraining automatique à envisager (cf. questions ouvertes du workshop).
