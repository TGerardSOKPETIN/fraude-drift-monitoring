# État de l'art — Data Drift et monitoring de modèles ML (plan)

Squelette du volet théorique du stage/mémoire. À compléter au fil de la rédaction, en cohérence avec le cadrage détaillé avec M. Mihoubi.

## 1. Cycle de vie d'un modèle ML

- Pré-déploiement : préparation des données, entraînement, validation.
- Post-déploiement : surveillance continue, détection de dérive, diagnostic, réentraînement, maintenance.
- Schéma à produire (cycle complet, cf. consigne du workshop : "un schéma vaut mieux que mille mots").

## 2. Typologie des dérives

- Data drift / covariate drift : P(X) change, P(Y|X) stable.
- Label drift : P(Y) change (taux de fraude lui-même).
- Concept drift : P(Y|X) change (le lien features → fraude se modifie).
- Distinction data drift → retraining généralement suffisant / concept drift → remise en question plus profonde du modèle (analogie du moteur évoquée en workshop).

## 3. Méthodes statistiques de détection

- Tests univariés : Kolmogorov-Smirnov (continu), Chi-deux (catégoriel).
- Mesures de magnitude : PSI (Population Stability Index), distance de Wasserstein, divergence de Jensen-Shannon.
- Limite univariée et réponse multivariée : Maximum Mean Discrepancy (MMD).
- Référence PSI à intégrer : PSIfinal.pdf partagé en workshop (files.wmich.edu).

## 4. Outils comparés

- Evidently AI, Alibi-Detect, NannyML : fonctionnement, méthodes, forces/faiblesses respectives.
- Benchmark de référence : D3Bench (méthodologie et critères repris pour le benchmark du stage).

## 5. Estimation de performance sans labels

- CBPE (NannyML) : principe, condition de calibration, limite sous concept drift.
- Pertinence pour la fraude bancaire (label réel souvent disponible en différé).

## 6. Enjeux sectoriels

- Déséquilibre des classes en détection de fraude, métriques à privilégier (precision, recall, PR-AUC plutôt que l'accuracy seule).
- Environnement adversarial (les fraudeurs adaptent leur comportement).
