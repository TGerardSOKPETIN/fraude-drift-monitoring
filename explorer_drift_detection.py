"""
Script d'exploration : détection de data drift sur le dataset Poundbank
avec les trois outils vus au workshop (Evidently AI, Alibi-Detect, NannyML).

Objectif : un seul fichier, à lire de haut en bas, pour comprendre comment
chaque outil fonctionne avant de les séparer proprement dans le repo GitHub.

Pré-requis :
    pip install pandas numpy scipy evidently alibi-detect nannyml

Données attendues, dans le même dossier que ce script (ou adapte DATA_DIR) :
    reference.csv
    analysis.csv

Colonnes du dataset Poundbank :
    timestamp, time_since_login_min, transaction_amount, transaction_type,
    is_first_transaction, user_tenure_months, is_fraud,
    predicted_fraud_proba, predicted_fraud
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuration : chemins et colonnes
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent  # adapte si tes CSV sont ailleurs

TIMESTAMP_COL = "timestamp"
TARGET_COL = "is_fraud"
PREDICTION_COL = "predicted_fraud"
PREDICTED_PROBA_COL = "predicted_fraud_proba"

# Variables numériques continues (tests KS / Wasserstein)
NUMERICAL_FEATURES = ["time_since_login_min", "transaction_amount", "user_tenure_months"]

# Variables catégorielles / binaires (test du Chi-deux)
CATEGORICAL_FEATURES = ["transaction_type", "is_first_transaction"]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = DATA_DIR / "reference.csv"
    analysis_path = DATA_DIR / "analysis.csv"
    for path in (reference_path, analysis_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}. "
                "Place reference.csv et analysis.csv à côté de ce script, "
                "ou modifie DATA_DIR en haut du fichier."
            )
    reference = pd.read_csv(reference_path, parse_dates=[TIMESTAMP_COL])
    analysis = pd.read_csv(analysis_path, parse_dates=[TIMESTAMP_COL])
    return reference, analysis


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Evidently AI — Data Drift preset
#    Sélectionne automatiquement KS / Chi-deux / PSI selon le type de colonne
#    et compare reference vs analysis colonne par colonne.
# ---------------------------------------------------------------------------

def run_evidently(reference: pd.DataFrame, analysis: pd.DataFrame) -> None:
    print_header("1. Evidently AI — Data Drift preset")
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=analysis[FEATURE_COLUMNS],
    )

    # Rapport HTML interactif, à ouvrir dans un navigateur
    result.save_html("evidently_data_drift.html")
    print("Rapport HTML généré : evidently_data_drift.html")

    # Aperçu synthétique en console
    result_dict = result.dict()
    print("Résumé brut disponible via result.dict() -- ouvre le HTML pour le détail par colonne.")
    print(f"Nombre de métriques calculées : {len(result_dict.get('metrics', []))}")


# ---------------------------------------------------------------------------
# 2. Alibi-Detect — TabularDrift (KS pour le numérique, Chi-deux pour le
#    catégoriel), avec correction FDR car on teste plusieurs variables à la fois.
# ---------------------------------------------------------------------------

def run_alibi_detect(reference: pd.DataFrame, analysis: pd.DataFrame) -> None:
    print_header("2. Alibi-Detect — TabularDrift (KS + Chi-deux, correction FDR)")
    from alibi_detect.cd import TabularDrift

    ref = reference[FEATURE_COLUMNS].copy()
    cur = analysis[FEATURE_COLUMNS].copy()

    # Alibi-Detect attend des catégorielles encodées en entiers, avec un
    # mapping {index_colonne: nombre_de_modalités}.
    categories_per_feature: dict[int, int] = {}
    for col in CATEGORICAL_FEATURES:
        ref[col] = ref[col].astype("category")
        categories = ref[col].cat.categories
        cur[col] = cur[col].astype(pd.CategoricalDtype(categories=categories))
        ref[col] = ref[col].cat.codes
        cur[col] = cur[col].cat.codes
        categories_per_feature[FEATURE_COLUMNS.index(col)] = len(categories)

    x_ref = ref.to_numpy(dtype=np.float64)
    x_cur = cur.to_numpy(dtype=np.float64)

    detector = TabularDrift(
        x_ref=x_ref,
        p_val=0.05,
        categories_per_feature=categories_per_feature,
        correction="fdr",
    )
    prediction = detector.predict(x_cur, drift_type="feature")

    print(f"{'Feature':<24}{'Type':<14}{'Drift':<8}{'p-value':<10}")
    for i, col in enumerate(FEATURE_COLUMNS):
        col_type = "categorical" if col in CATEGORICAL_FEATURES else "numerical"
        is_drift = bool(prediction["data"]["is_drift"][i])
        p_val = float(prediction["data"]["p_val"][i])
        flag = "DRIFT" if is_drift else "stable"
        print(f"{col:<24}{col_type:<14}{flag:<8}{p_val:<10.4f}")

    dataset_drift = bool(prediction["data"]["is_drift"].any())
    print(f"\nDataset drift (au moins une variable en drift) : {dataset_drift}")


# ---------------------------------------------------------------------------
# 3. NannyML — drift univarié + estimation de performance sans labels (CBPE)
#    C'est le seul des trois outils qui estime la performance du modèle
#    (ROC-AUC, precision, recall) sans avoir besoin des vrais labels de fraude.
# ---------------------------------------------------------------------------

def run_nannyml(reference: pd.DataFrame, analysis: pd.DataFrame) -> None:
    print_header("3. NannyML — drift univarié + CBPE (estimation de performance)")
    import nannyml as nml

    # --- 3a. Drift univarié ---
    univariate_calc = nml.UnivariateDriftCalculator(
        column_names=FEATURE_COLUMNS,
        continuous_methods=["kolmogorov_smirnov"],
        categorical_methods=["chi2"],
        timestamp_column_name=TIMESTAMP_COL,
    )
    univariate_calc.fit(reference)
    drift_results = univariate_calc.calculate(analysis)
    drift_df = drift_results.to_df()
    print("Drift univarié calculé. Aperçu des colonnes du résultat :")
    print(drift_df.columns.tolist()[:10], "...")
    drift_df.to_csv("nannyml_univariate_drift.csv")
    print("Détail complet : nannyml_univariate_drift.csv")

    # --- 3b. CBPE : estimation de performance sans les vrais labels ---
    # Attention : ne fonctionne bien que sous data drift, pas sous concept
    # drift, et nécessite des probabilités bien calibrées.
    estimator = nml.CBPE(
        y_pred_proba=PREDICTED_PROBA_COL,
        y_pred=PREDICTION_COL,
        y_true=TARGET_COL,
        timestamp_column_name=TIMESTAMP_COL,
        metrics=["roc_auc", "precision", "recall", "f1"],
        chunk_period="M",
    )
    estimator.fit(reference)
    performance_results = estimator.estimate(analysis)
    performance_df = performance_results.to_df()
    performance_df.to_csv("nannyml_cbpe_performance.csv")
    print("Performance estimée (sans labels) : nannyml_cbpe_performance.csv")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reference_df, analysis_df = load_data()
    print(f"reference.csv : {len(reference_df)} lignes")
    print(f"analysis.csv  : {len(analysis_df)} lignes")

    run_evidently(reference_df, analysis_df)
    run_alibi_detect(reference_df, analysis_df)
    run_nannyml(reference_df, analysis_df)

    print_header("Terminé")
    print("Fichiers générés dans ce dossier : evidently_data_drift.html, "
          "nannyml_univariate_drift.csv, nannyml_cbpe_performance.csv")
