"""Détection de drift univarié + estimation de performance (CBPE) avec NannyML.

Deux volets, propres à NannyML par rapport aux deux autres outils :
1. Drift univarié (KS pour le numérique, Chi-deux pour le catégoriel).
2. CBPE : estimation de la performance du modèle (ROC-AUC, precision, recall)
   sur les données d'analyse SANS utiliser les vrais labels, à partir des
   probabilités prédites. Ne fonctionne correctement que sous data drift
   (pas sous concept drift) : à garder en tête dans l'interprétation.

Usage :
    python -m src.drift_nannyml.run_drift_nannyml
"""
from __future__ import annotations

import json

import nannyml as nml

from src.common.data_loading import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTED_PROBA_COL,
    PREDICTION_COL,
    TARGET_COL,
    TIMESTAMP_COL,
    ensure_reports_dir,
    load_reference_and_analysis,
)


def main() -> None:
    reference, analysis = load_reference_and_analysis()
    reports_dir = ensure_reports_dir()

    # --- 1. Drift univarié ---
    univariate_calc = nml.UnivariateDriftCalculator(
        column_names=NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        continuous_methods=["kolmogorov_smirnov"],
        categorical_methods=["chi2"],
        timestamp_column_name=TIMESTAMP_COL,
    )
    univariate_calc.fit(reference)
    drift_results = univariate_calc.calculate(analysis)
    drift_df = drift_results.to_df()
    drift_csv_path = reports_dir / "nannyml_univariate_drift.csv"
    drift_df.to_csv(drift_csv_path)
    print(f"Rapport de drift univarié NannyML : {drift_csv_path}")

    # --- 2. Estimation de performance sans labels (CBPE) ---
    # Nécessite des probabilités bien calibrées : à vérifier avant de
    # prendre les résultats pour argent comptant (voir docs/etat-de-l-art.md).
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
    perf_csv_path = reports_dir / "nannyml_cbpe_performance.csv"
    performance_df.to_csv(perf_csv_path)
    print(f"Estimation de performance (CBPE) NannyML : {perf_csv_path}")

    summary = {
        "n_features_analysed": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
        "drift_report": str(drift_csv_path),
        "performance_report": str(perf_csv_path),
    }
    summary_path = reports_dir / "nannyml_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
