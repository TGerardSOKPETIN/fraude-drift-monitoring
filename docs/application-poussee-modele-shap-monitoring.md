---
title: Application poussée — modèles champion/challenger, explicabilité SHAP et monitoring
dernière mise à jour: 2026-10-02
statut: pipeline exécuté et validé (modèles entraînés, SHAP calculé, service testé, monitoring comparé)
---

# Application poussée : construire, expliquer et surveiller nos propres modèles

Cette partie complète le corps principal du mémoire (surveillance du modèle déjà en production, via `predicted_fraud_proba`) par une démarche différente et volontairement délimitée : construire nous-mêmes un modèle de détection de fraude, pour pouvoir répondre à une exigence restée bloquée depuis le Workshop 2 — l'explicabilité (SHAP), impossible sans accès à un vrai modèle.

Elle s'inspire du dépôt open source [sentinel-fraud-mlops](https://github.com/hugocorreia123/sentinel-fraud-mlops) (Hugo Correia), un système de détection de fraude temps réel complet (modèles champion/challenger, service d'inférence, feature store, observabilité, tests de charge, robustesse adversariale). Après évaluation, un périmètre réduit et réaliste (« version intermédiaire ») a été retenu plutôt qu'une réplique intégrale : voir la section 5 pour le détail de ce qui a été repris et de ce qui a été volontairement écarté, et pourquoi.

## 1. Méthodologie et modèles entraînés

Les cinq variables disponibles (`time_since_login_min`, `transaction_amount`, `transaction_type`, `is_first_transaction`, `user_tenure_months`) ont été transformées en matrice de features (encodage one-hot de `transaction_type`), de façon identique entre l'entraînement et le service de scoring.

Deux modèles ont été entraînés sur `reference.csv` uniquement, avec un découpage stratifié 80 % entraînement / 20 % test (« holdout »), pour reproduire une logique **champion / challenger** :

- **Champion** : LightGBM (gradient boosting), avec une petite recherche sur 3 configurations d'hyperparamètres (nombre de feuilles, taux d'apprentissage, nombre d'estimateurs), sélection sur la PR-AUC de validation.
- **Challenger** : régression logistique sur les mêmes features standardisées — modèle plus simple, servant de point de comparaison plutôt qu'un réseau de neurones tabulaire (choix délibéré, cf. section 5).

**Résultats sur le holdout** (jamais vu pendant l'entraînement) :

| Modèle | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Champion (LightGBM) | 0,788 | 0,551 | 0,649 | 0,744 | 0,793 |
| Challenger (régression logistique) | 0,737 | 0,560 | 0,636 | 0,725 | 0,778 |

Le champion devance le challenger sur la quasi-totalité des métriques, ce qui justifie son rôle de modèle servant la décision.

## 2. Un modèle indépendant du modèle d'origine — point de rigueur à ne pas perdre de vue

Le champion a ensuite scoré `analysis.csv`. La corrélation entre ses probabilités et la colonne `predicted_fraud_proba` déjà présente dans les données (celle du modèle d'origine surveillé dans le corps principal du mémoire) n'est que de **0,50**. C'est attendu : le modèle d'origine a vraisemblablement été entraîné avec davantage d'informations que les cinq variables disponibles ici. Toute conclusion tirée de ce champion (SHAP, monitoring) décrit donc le comportement de **ce modèle-ci**, pas une explication directe du modèle d'origine. Les deux analyses sont parallèles et complémentaires, pas interchangeables.

## 3. Explicabilité (SHAP)

Des valeurs SHAP ont été calculées sur le champion (TreeExplainer), séparément sur un échantillon de la période de référence et un échantillon de la période d'analyse, pour vérifier si l'importance des variables reste stable dans le temps.

| Variable | Importance SHAP (référence) | Importance SHAP (analyse) |
|---|---|---|
| `user_tenure_months` | 1,188 | 1,181 |
| `transaction_amount` | 0,189 | 0,194 |
| `time_since_login_min` | 0,080 | 0,079 |
| `transaction_type` (4 modalités + manquant) | 0,007 à 0,014 chacune | idem |
| `is_first_transaction` | 0,014 | 0,013 |

`user_tenure_months` domine très largement les autres variables, dans des proportions quasi identiques entre référence et analyse — l'ancienneté du compte est de loin le signal le plus discriminant pour ce modèle, ce qui est défendable (un compte récent est un facteur de risque classique en détection de fraude), et ce classement est stable dans le temps pour ce modèle.

Point de vigilance : c'est `time_since_login_min`, et non `user_tenure_months`, qui montrait la dérive progressive la plus nette dans l'analyse NannyML du modèle d'origine (cf. `analyse-comparative-outils-drift.md`). Comme il s'agit de deux modèles distincts (section 2), ce résultat SHAP n'explique pas directement la baisse de performance du modèle d'origine ; il documente le comportement du champion construit ici.

## 4. Service de scoring temps réel

Un service FastAPI minimal (sans Redis, sans conteneurisation Docker, cf. section 5) expose deux routes : `/health` et `/predict`, qui renvoie la probabilité de fraude du champion et du challenger, la décision (seuil à 0,5) et le temps de réponse. Testé avec succès, latence entre 5 et 16 millisecondes par requête (sans mise en charge).

## 5. Ce qui a été repris du dépôt de référence, et ce qui a été volontairement écarté

| Brique (dépôt sentinel-fraud-mlops) | Reprise ici ? | Pourquoi |
|---|---|---|
| Pipeline d'entraînement | Oui | Sur `reference.csv`, cohérent avec le reste du mémoire |
| Modèle champion | Oui (LightGBM) | — |
| Modèle challenger | Oui, simplifié (régression logistique) | Un réseau de neurones tabulaire ajoute de la complexité d'entraînement sans apport méthodologique proportionné pour un mémoire |
| Service d'inférence (FastAPI) | Oui, allégé | Sans feature store Redis : les features sont calculées directement à la volée |
| Décision coût-sensible | Non (seuil fixe à 0,5) | Piste d'amélioration identifiée, directement liée à RQ5 (seuils de décision) |
| Mode shadow / routage A/B avec journalisation | Partiel | Les deux modèles sont appelés à chaque requête, mais sans infrastructure de routage par hash ni journal dédié |
| MLflow (registre de modèles) | Non | Outil de suivi d'expériences utile en production, sans apport à l'analyse d'un mémoire ; modèles simplement sauvegardés sur disque |
| Redis (feature store en ligne) | Non | Infrastructure supplémentaire, hors périmètre retenu |
| Prometheus / Grafana / Alertmanager | Non | Le mémoire dispose déjà de sa propre couche de monitoring (Evidently, Alibi-Detect, NannyML), qu'il n'y avait pas lieu de dupliquer avec une stack d'observabilité opérationnelle |
| Tests de charge (Locust) | Non | Démonstration d'ingénierie de production, non centrale aux questions de recherche du mémoire |
| Évaluation de robustesse adversariale (FGSM) | Non | Hors périmètre retenu, mentionnée comme piste d'ouverture possible |
| Explicabilité (SHAP) | Ajoutée (absente du dépôt source) | Répond directement à la demande de M. Mihoubi (Workshop 2, section 5) |

## 6. Monitoring du champion — comparaison avec le modèle d'origine

Le même protocole CBPE que celui déjà utilisé sur le modèle d'origine a été appliqué aux prédictions du champion (calibré sur le holdout de référence, pour éviter toute fuite des données d'entraînement), et confronté à la performance **réelle** mensuelle, calculable ici car `is_fraud` est disponible dans `analysis.csv`.

**Convergence avec le modèle d'origine.** La performance réelle du champion reste stable de novembre 2018 à mars 2019, puis chute nettement à partir d'avril 2019 et le reste jusqu'en juin (ROC-AUC ~0,75 → ~0,71 ; Recall ~0,56 → ~0,49 ; F1 ~0,65 → ~0,60), exactement la même fenêtre temporelle que la dégradation observée sur le modèle d'origine. Ces deux modèles étant entraînés indépendamment (corrélation de prédictions de seulement 0,50, cf. section 2), cette convergence est un indice fort qu'un changement structurel des données s'est produit à partir d'avril 2019, plutôt qu'un artefact propre au modèle d'origine seul.

**CBPE suit la tendance mais sous-estime l'ampleur.** L'estimation CBPE (sans utiliser les vraies étiquettes de la période d'analyse) suit la même direction que la performance réelle, mais la sous-estime, en particulier pour Recall et F1 à partir d'avril (Recall estimé ~0,52 contre ~0,48-0,49 réel). La Precision ne montre de tendance nette ni dans l'estimation ni dans le réel, cohérent avec l'observation faite sur le modèle d'origine.

**Aucune alerte déclenchée — constat méthodologique important.** Sur les 4 métriques et les 8 mois d'analyse, NannyML n'a levé aucune alerte pour ce champion, malgré une dégradation réelle et visible. Le seuil d'alerte de NannyML est calibré sur la variabilité observée pendant la période de référence ; cette variabilité était ici assez large pour que la baisse d'avril-juin reste dans la bande considérée comme normale. Ce résultat renforce directement la recommandation déjà posée dans `architecture-monitoring-alertes-retraining.md` : un seuil fondé uniquement sur la variabilité de la référence peut manquer une dégradation réelle, d'où l'intérêt d'une règle complémentaire fondée sur la tendance (plusieurs mois consécutifs de baisse) plutôt qu'un seuil statique seul.

## 7. Limites à assumer explicitement

Le champion et le challenger construits ici sont des modèles indépendants du modèle d'origine (section 2) : les résultats SHAP et de monitoring de cette partie documentent leur propre comportement, et ne peuvent pas être présentés comme une explication directe de la dégradation du modèle d'origine. La convergence temporelle observée (section 6) est un indice de cohérence, pas une preuve que les deux modèles dégradent pour la même raison exacte. Le seuil de décision du champion (0,5) n'est pas optimisé selon un coût métier asymétrique (faux négatif plus coûteux qu'un faux positif), contrairement à ce que propose le dépôt de référence — amélioration possible, directement rattachée à RQ5.
