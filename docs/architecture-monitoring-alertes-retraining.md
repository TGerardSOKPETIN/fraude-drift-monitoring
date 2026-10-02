---
title: Vision MLOps — alerting, seuils et stratégie de réentraînement
dernière mise à jour: 2026-10-02
statut: s'appuie directement sur les résultats de analyse-comparative-outils-drift.md
---

# Vision MLOps : du monitoring batch à une stratégie d'alerte et de réentraînement

Ce document répond au point prioritaire 3 du Workshop 2 (sections 9 et 10 du compte-rendu) : décrire le cycle de vie complet du modèle, proposer des seuils d'alerte concrets, et poser une stratégie de réentraînement. Il s'appuie sur le constat établi dans `analyse-comparative-outils-drift.md` plutôt que sur des recommandations génériques, conformément à la même exigence de rigueur.

## 1. Cycle de vie ML visé

```
Entraînement ──► Validation ──► Déploiement ──► Monitoring continu ──┬──► Détection de drift (univarié, chunké)
                                                                      └──► Estimation de performance (CBPE)
                                                                              │
                                                                              ▼
                                                                   Règles d'alerte (seuils + tendance)
                                                                              │
                                                              ┌───────────────┴───────────────┐
                                                              ▼                                ▼
                                                     Investigation / diagnostic        Déclenchement réentraînement
                                                     (quelle variable, depuis quand)    (pipeline + validation + bascule)
                                                              │                                │
                                                              └──────────────► Retour au monitoring continu
```

Ce schéma reprend la structure demandée en réunion (Training → Validation → Déploiement → Monitoring → Détection de drift → Alerte → Réentraînement), en y ajoutant explicitement l'estimation de performance (CBPE) comme deuxième bras du monitoring, à côté de la détection de drift univarié — c'est ce double canal qui s'avère déterminant dans le constat du point 2.

## 2. Ce que le constat empirique impose à la conception des alertes

Le point 2 a montré deux choses qu'une stratégie d'alerte ne peut pas ignorer :

1. **Un test global sur toute la période de production est un mauvais détecteur précoce.** Dans notre cas, Alibi-Detect et Evidently, appliqués une seule fois sur l'ensemble de la fenêtre d'analyse, ne permettent ni l'un ni l'autre de dater le début de la dérive. Seul le découpage temporel (NannyML) l'a révélé, et seulement a posteriori.
2. **Le signal de performance (CBPE) a précédé de deux mois le signal de drift univarié basé sur un seuil.** La dégradation commence en avril 2019 ; l'alerte de drift univarié sur `time_since_login_min` ne se déclenche qu'en juin. Un dispositif qui n'aurait surveillé que le franchissement d'un seuil de drift aurait perdu deux mois.

Conséquence directe pour la conception : **l'alerte ne doit pas reposer sur un verdict binaire unique (drift / pas drift) recalculé périodiquement sur toute la fenêtre de production**, mais sur un monitoring chunké dans le temps, combinant surveillance de tendance et surveillance de performance.

## 3. Proposition de dispositif d'alerte à deux niveaux

### Niveau 1 — Surveillance de distribution, chunkée, avec détection de tendance

- Recalcul mensuel (ou à fréquence de chunk fixe) des statistiques de drift univarié (KS pour les variables numériques, Chi² pour les catégorielles), sur le modèle NannyML déjà en place.
- **Règle de tendance, en plus du seuil ponctuel** : déclencher une alerte « précoce » dès que la statistique augmente de façon monotone sur 3 segments consécutifs, même si le seuil d'alerte n'est pas encore franchi. Appliquée rétrospectivement à nos données, cette règle aurait signalé `time_since_login_min` dès le segment de mars 2019 — soit environ trois mois avant l'alerte par seuil simple en juin 2019.
- Priorité de surveillance sur les deux variables qui se sont révélées porteuses de signal réel dans le point 2 (`time_since_login_min`, `transaction_amount`), sans pour autant cesser de surveiller les autres : `user_tenure_months` a montré qu'une dérive peut apparaître brutalement sur un seul segment après une longue période stable.
- Les alertes isolées et non reproduites (cas de `is_first_transaction`, décembre 2018) ne doivent pas déclencher d'action : exiger une confirmation sur au moins deux segments consécutifs avant de considérer le signal comme actionnable, afin de limiter les faux positifs inhérents aux tests répétés à chaque segment.

### Niveau 2 — Surveillance de performance estimée (CBPE), levier décisionnel principal

- Suivi mensuel de ROC AUC, Precision, Recall et F1 estimés par CBPE, avec une attention particulière portée au Recall : c'est la métrique la plus directement liée au risque métier (fraudes non détectées), et c'est elle qui a affiché le recul relatif le plus marqué dans notre analyse (≈ 0,929 → ≈ 0,885).
- Seuil d'alerte proposé : écart relatif supérieur à un seuil fixé avec les équipes métier (par exemple 2–3 points de Recall en dessous de la performance de référence), confirmé sur deux mois consécutifs plutôt que déclenché sur un seul point — ce qui correspond exactement au schéma observé (avril puis mai puis juin 2019).
- Ce niveau constitue le signal le plus directement actionnable, car il traduit un impact métier et non une simple différence de distribution : une dérive de distribution sans impact de performance ne justifie pas nécessairement un réentraînement.

### Croisement des deux niveaux pour le diagnostic

Le rôle du niveau 1 n'est pas seulement de déclencher une alerte, mais de permettre le diagnostic une fois qu'une alerte de performance (niveau 2) est levée : identifier *quelle* variable dérive et *depuis quand*, pour orienter la décision de réentraînement (simple réajustement vs collecte de nouvelles données vs révision des features). C'est cette articulation, plus que chaque outil pris isolément, qui constitue la valeur du dispositif.

## 4. Stratégie de réentraînement

### Déclencheurs proposés (approche hybride, pas uniquement calendaire ni uniquement événementielle)

- **Déclencheur événementiel prioritaire** : alerte de performance (niveau 2) confirmée sur deux mois consécutifs → lancement d'un pipeline de réentraînement.
- **Déclencheur événementiel secondaire** : alerte de tendance de drift (niveau 1) confirmée sur une variable déjà identifiée comme influente, même sans alerte de performance encore visible → investigation renforcée et préparation anticipée des données de réentraînement (sans forcément déclencher la bascule en production).
- **Déclencheur calendaire de sécurité** : réentraînement planifié à fréquence fixe (par exemple trimestrielle) indépendamment des alertes, pour couvrir les dérives lentes qui ne franchiraient aucun seuil mais s'accumuleraient sur une période plus longue que la fenêtre d'analyse disponible.

### Processus de bascule

1. Réentraînement sur une fenêtre de données incluant la période récente identifiée comme dérivante.
2. Validation hors-ligne sur un jeu de test récent, en comparant explicitement le nouveau modèle à l'ancien (champion/challenger) sur les mêmes métriques que celles suivies en production (ROC AUC, Precision, Recall, F1).
3. Déploiement en *shadow mode* (le nouveau modèle scoring en parallèle, sans impact opérationnel) avant bascule complète, le temps de confirmer sur données réelles que la dérive observée est bien corrigée.
4. Bascule effective, avec conservation de l'ancien modèle disponible pour rollback immédiat en cas de dégradation inattendue.
5. Reprise du cycle de monitoring sur le nouveau modèle, avec une nouvelle période de référence redéfinie à partir des données les plus récentes.

### Point de vigilance spécifique au contexte fraude

L'estimation CBPE fonctionne sans étiquettes réelles, ce qui est nécessaire ici puisque les vraies confirmations de fraude arrivent typiquement avec un délai (contestations, enquêtes). Il est recommandé de **revalider périodiquement les estimations CBPE a posteriori**, dès que les étiquettes réelles deviennent disponibles avec leur délai habituel, afin de s'assurer que l'estimation de confiance n'a pas elle-même dérivé — CBPE reste valide sous data drift mais pas sous concept drift avéré, ce qui est précisément l'hypothèse non tranchée soulevée au point 2.

## 5. Lien avec l'architecture cible (batch → temps réel)

Ce dispositif d'alerte à deux niveaux est pensé pour s'intégrer à la trajectoire déjà esquissée dans `docs/architecture-cible.md` du dépôt (passage d'un prototype batch à une architecture de streaming). Dans une architecture temps réel, les niveaux 1 et 2 deviennent des jobs de calcul périodique (par exemple horaires ou journaliers plutôt que mensuels) consommant le flux de scoring, avec les mêmes règles de tendance et de confirmation multi-fenêtres décrites ci-dessus — le changement de fréquence ne change pas la logique de décision, seulement la granularité des chunks.

## 6. Limites à assumer explicitement dans le mémoire

- Le lien entre la dérive de `time_since_login_min` et la baisse de performance reste une hypothèse argumentée par concordance temporelle (cf. point 2), pas une causalité démontrée : l'analyse d'attribution (SHAP) nécessaire pour la confirmer est hors de portée sans accès au modèle réel.
- Les seuils proposés ici (2–3 points de Recall, 3 segments consécutifs) sont des propositions méthodologiques à calibrer avec les équipes métier et risque, pas des valeurs validées sur un historique de production réel.
- La fenêtre de données disponible (environ 8 mois d'analyse) ne permet pas de distinguer une dérive cyclique/saisonnière d'une dérive durable — point à mentionner explicitement comme limite du dispositif proposé.
