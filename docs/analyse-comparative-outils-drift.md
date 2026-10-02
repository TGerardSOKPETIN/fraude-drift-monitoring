---
title: Analyse comparative rigoureuse — Evidently / Alibi-Detect / NannyML
dernière mise à jour: 2026-10-02
statut: résultats extraits directement du notebook exécuté (exploration_drift_detection.ipynb), aucune valeur recalculée ou estimée
---

# Analyse comparative rigoureuse des trois outils de détection de drift

Ce document répond au point prioritaire 2 identifié lors du Workshop 2 : « une analyse plus poussée des résultats déjà obtenus, avec une lecture fine des graphiques NannyML ». Conformément à la décision méthodologique actée le 2026-10-02 (cf. `retours-workshop2-mihoubi.md`), toutes les valeurs ci-dessous sont extraites telles quelles des sorties réellement produites par le notebook exécuté sur la machine de Gérard (environnement validé : Python 3.11.16, evidently 0.7.23, alibi-detect 0.13.0, nannyml 0.13.1). Aucune n'est reconstruite ou approximée.

## 1. Rappel du dispositif

- **Référence** : `reference.csv`, janvier–octobre 2018 (période supposée stable, sans drift).
- **Analyse** : `analysis.csv`, novembre 2018–juin 2019 (période de production à surveiller).
- **Variables surveillées** : 3 numériques (`time_since_login_min`, `transaction_amount`, `user_tenure_months`) et 2 catégorielles (`transaction_type`, `is_first_transaction`).
- Les trois outils n'utilisent ni le même test statistique, ni la même unité de mesure, ni le même découpage temporel. C'est précisément ce qui rend leur comparaison instructive plutôt que redondante.

## 2. Résultats bruts, outil par outil

### 2.1 Evidently (`DataDriftPreset`, test global sur l'ensemble de la période)

| Variable | Test utilisé | Score (magnitude) | Verdict Evidently |
|---|---|---|---|
| `time_since_login_min` | Wasserstein distance (normée) | 0,0475 | Not Detected |
| `transaction_amount` | Wasserstein distance (normée) | 0,0146 | Not Detected |
| `user_tenure_months` | Wasserstein distance (normée) | 0,0055 | Not Detected |
| `transaction_type` | Jensen-Shannon distance | 0,0020 | Not Detected |
| `is_first_transaction` | Jensen-Shannon distance | 0,0012 | Not Detected |

Aucune variable n'est signalée. Les scores sont tous très faibles en valeur absolue (le plus élevé, 0,0475, reste loin des seuils d'alerte usuels d'Evidently, de l'ordre de 0,1–0,2 selon le test).

### 2.2 Alibi-Detect (tests KS / Chi², correction FDR, test global sur l'ensemble de la période)

| Variable | Type | Drift détecté | p-value |
|---|---|---|---|
| `time_since_login_min` | numérique | **True** | 0,0000 |
| `transaction_amount` | numérique | **True** | 0,0085 |
| `is_first_transaction` | catégorielle | False | 0,6197 |
| `user_tenure_months` | numérique | False | 0,6303 |
| `transaction_type` | catégorielle | False | 0,9411 |

Verdict global imprimé par le notebook : *« Dataset drift (au moins une variable en drift, seuil p < 0,05 après correction FDR) : True »*.

### 2.3 NannyML (`UnivariateDriftCalculator`, découpage temporel en ~11 segments d'environ 24 jours chacun, de novembre 2018 à fin juin 2019)

| Variable | Test | Alerte(s) déclenchée(s) | Valeur au moment de l'alerte |
|---|---|---|---|
| `time_since_login_min` | KS | 1 — dernier segment (se terminant le 30/06/2019) | 0,2842 |
| `transaction_amount` | KS | 0 | dernière valeur observée : 0,2725 (sous le seuil propre à cette variable) |
| `user_tenure_months` | KS | 1 — dernier segment (se terminant le 30/06/2019) | 0,3235 |
| `is_first_transaction` | Chi² | 1 — segment se terminant le 19/12/2018 | 6,528 |
| `transaction_type` | Chi² | 0 | — |

Point méthodologique important : NannyML calcule, pour chaque variable, un seuil d'alerte propre déduit de la variabilité observée *pendant la période de référence elle-même* (et non un seuil universel). Les seuils ne sont donc pas comparables entre variables.

## 3. Pourquoi les trois outils ne racontent pas la même histoire

C'est ici que réside l'essentiel de la valeur du mémoire : comprendre *pourquoi* trois outils appliqués aux mêmes données produisent des verdicts différents, plutôt que se contenter de les juxtaposer.

### 3.1 `time_since_login_min` : le cas le plus parlant

- **Alibi-Detect** considère la variable en drift de façon très nette (p ≈ 0).
- **Evidently** ne détecte rien, avec un score de magnitude très faible (0,0475).
- **NannyML**, en découpant le temps, révèle la raison de ce désaccord apparent : la valeur de la statistique KS **progresse graduellement** tout au long de la période d'analyse (environ 0,01–0,02 de novembre 2018 à février 2019, puis une hausse continue à partir de mars 2019 — 0,012 → 0,040 → 0,049 → 0,052 → 0,051 — avant un saut net à 0,284 sur le tout dernier segment, fin juin 2019).

Autrement dit : la dérive n'est pas constante sur toute la période, elle **s'installe progressivement à partir de mars 2019 et s'accentue fortement en fin de période**. C'est exactement ce que les deux tests globaux (Evidently et Alibi-Detect) ne peuvent pas voir, puisqu'ils résument toute la période de novembre à juin en un seul chiffre :

- Alibi-Detect, avec un échantillon d'environ 40 000 observations, atteint la significativité statistique même pour un écart réel modeste — c'est précisément le phénomène que M. Mihoubi avait anticipé lors du cadrage (« avec de grands échantillons, des écarts triviaux deviennent significatifs »). Le test global reste cependant cohérent avec le signal réel : la seconde moitié de la période (où l'écart est le plus marqué) pèse suffisamment dans l'échantillon global pour faire basculer le test.
- Evidently, en calculant une distance de Wasserstein sur l'ensemble de la période d'un coup, **moyenne** la faible dérive des 7–8 premiers segments avec la forte dérive du dernier segment — et le résultat moyen reste sous le seuil d'alerte. C'est un résultat cohérent avec sa méthode, mais qui **sous-estime** la dynamique réelle si on ne regarde que ce chiffre unique.
- NannyML est le seul des trois à exposer la trajectoire temporelle, et donc le seul à permettre de dater le début de la dérive (environ mars 2019) plutôt que de se contenter d'un verdict binaire sur toute la période.

### 3.2 `transaction_amount` : un signal plus tardif et plus modéré

Même lecture, à une échelle plus modérée : la statistique KS par segment reste stable et faible jusqu'à la mi-avril 2019 (0,011–0,016), puis augmente à partir de mai (0,034 puis 0,035) et atteint 0,2725 sur le dernier segment — sans toutefois franchir le seuil propre à cette variable (NannyML ne déclenche aucune alerte). Alibi-Detect, lui, détecte un drift significatif sur le test global (p = 0,0085) : là encore, la seconde moitié de la période pèse suffisamment pour faire basculer un test portant sur l'ensemble des 40 000 observations, alors que le score de magnitude d'Evidently (0,0146) reste très faible une fois moyenné sur toute la période.

### 3.3 `user_tenure_months` : l'effet de dilution temporelle est maximal ici

C'est le cas le plus net de divergence par dilution : NannyML déclenche une alerte sur le seul dernier segment (0,3235, une valeur élevée), alors que ce même segment ne représente qu'environ 1/11 de la période d'analyse. Une fois noyée dans un test global portant sur les huit mois complets, cette dérive localisée disparaît statistiquement : Alibi-Detect ne détecte rien (p = 0,63) et Evidently non plus (score de 0,0055). Les trois outils ont, chacun à sa façon, raison compte tenu de leur méthode — mais seul le découpage temporel de NannyML permet de voir qu'un épisode de dérive bien réel s'est produit, uniquement en toute fin de période.

### 3.4 `is_first_transaction` et `transaction_type` : convergence vers l'absence de dérive structurelle

Pour ces deux variables catégorielles, les trois outils s'accordent globalement : pas de drift détecté par Evidently, pas de drift détecté par Alibi-Detect (p = 0,62 et p = 0,94). La seule alerte isolée vient de NannyML sur `is_first_transaction`, au segment se terminant le 19 décembre 2018 (Chi² = 6,53) — un pic unique, non reproduit dans les segments suivants (qui reviennent à des valeurs de l'ordre de 0,5 à 3), et non corroboré par le test global d'Alibi-Detect. En toute rigueur, ce point isolé s'interprète comme du bruit statistique plutôt que comme un signal de dérive réelle : avec onze tests indépendants effectués sur la même variable (un par segment), l'apparition d'un faux positif isolé est statistiquement attendue, et aucune correction multi-tests n'est appliquée dans le calcul NannyML par défaut contrairement à Alibi-Detect (qui applique une correction FDR sur son test global).

## 4. Le lien avec la dégradation de performance (CBPE)

L'estimation de performance par confiance (CBPE, NannyML), calculée mensuellement sans disposer des vraies étiquettes sur la période d'analyse, montre un schéma net sur les quatre métriques suivies :

| Métrique | Nov 2018 – Mars 2019 | Avril – Juin 2019 |
|---|---|---|
| ROC AUC | ≈ 0,971 | ≈ 0,961 |
| Precision | ≈ 0,957–0,958 | ≈ 0,940–0,941 |
| Recall | ≈ 0,929 | ≈ 0,885 |
| F1 | ≈ 0,943 | ≈ 0,910–0,912 |

Les quatre métriques sont stables pendant cinq mois, puis chutent **simultanément à partir d'avril 2019** et restent à ce niveau dégradé jusqu'en juin — avec des alertes NannyML déclenchées sur les trois derniers segments (avril, mai, juin 2019) pour chacune des quatre métriques. Le recul relatif le plus marqué concerne le rappel (Recall), ce qui est le signal le plus sensible pour un modèle de détection de fraude : il traduit une proportion croissante de fraudes non détectées.

**Lecture rigoureuse de ce rapprochement temporel :**

- La baisse de performance **précède d'environ deux mois** le moment où le drift univarié franchit les seuils d'alerte (juin 2019 pour `time_since_login_min` et `user_tenure_months`).
- Elle coïncide en revanche avec le moment où la statistique KS de `time_since_login_min` entame sa hausse continue (mars 2019 → avril 2019), observable uniquement grâce au découpage temporel de NannyML.
- Le taux de fraude mensuel (cf. section EDA du notebook) ne présente pas de rupture notable à ce même moment : il reste dans la fourchette observée sur l'ensemble de la période analysée, ce qui tend à écarter un simple décalage du taux de base (*label drift*) comme explication de la baisse de performance.
- Ce faisceau d'éléments est **cohérent avec** une explication par dérive progressive de la distribution d'une covariable (`time_since_login_min`, et dans une moindre mesure `transaction_amount`) affectant la capacité discriminante du modèle — ce qui s'apparente à un effet de *concept drift* au sens où la relation apprise par le modèle entre les variables et la fraude devient progressivement moins fiable à mesure que la distribution des entrées évolue.
- **Cette explication reste une hypothèse étayée par la concordance temporelle, et non une preuve causale.** Établir un lien de causalité formel nécessiterait une analyse d'attribution (par exemple via SHAP), explicitement non réalisable à ce stade faute d'accès au modèle réel (cf. point 5 du compte-rendu Workshop 2). Le mémoire doit présenter ce rapprochement comme une piste d'interprétation argumentée, pas comme une conclusion établie.

## 5. Tableau de synthèse

| Variable | Evidently (global, magnitude) | Alibi-Detect (global, significativité) | NannyML (chunké, temporel) | Lecture rigoureuse |
|---|---|---|---|---|
| `time_since_login_min` | Pas de drift (0,0475) | Drift (p≈0) | Alerte en fin de période seulement ; hausse progressive dès mars 2019 | Dérive réelle, progressive, qui s'intensifie en fin de période — seul NannyML en montre la dynamique |
| `transaction_amount` | Pas de drift (0,0146) | Drift (p=0,0085) | Pas d'alerte ; hausse dès mai 2019, sous le seuil | Dérive plus tardive et plus modérée que la précédente |
| `user_tenure_months` | Pas de drift (0,0055) | Pas de drift (p=0,63) | Alerte, mais uniquement sur le tout dernier segment | Épisode localisé en toute fin de période, dilué dans les tests globaux |
| `is_first_transaction` | Pas de drift (0,0012) | Pas de drift (p=0,62) | 1 alerte isolée (déc. 2018), non reproduite | Convergence des trois outils : pas de dérive structurelle, signal isolé interprété comme du bruit |
| `transaction_type` | Pas de drift (0,0020) | Pas de drift (p=0,94) | Pas d'alerte | Convergence totale des trois outils : aucune dérive |

## 6. Ce que cette comparaison apporte au mémoire

1. **Elle illustre concrètement** la distinction significativité/magnitude demandée par M. Mihoubi dès le cadrage : un même écart peut être « significatif » (Alibi-Detect, grand échantillon) et « négligeable en magnitude » (Evidently), sans que l'un des deux outils ait tort — ils répondent à des questions différentes.
2. **Elle démontre l'intérêt du découpage temporel** (NannyML) par rapport à un test unique sur toute la période : c'est le seul des trois outils qui permette de dater l'apparition d'une dérive et d'en observer la progression, plutôt que de se limiter à un verdict binaire global.
3. **Elle établit un rapprochement argumenté, mais prudent,** entre la dynamique de drift observée et la dégradation de performance estimée par CBPE — avec, pour conséquence directe, une implication pour la stratégie d'alerte développée au point 3 : un dispositif fondé uniquement sur un test de drift global (comme celui d'Alibi-Detect ou d'Evidently appliqué en une fois) aurait manqué deux mois d'alerte précoce par rapport à un suivi chunké des features ou à un suivi direct de la performance estimée.
