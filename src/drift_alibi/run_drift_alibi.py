"""Détection de data drift avec Alibi-Detect (TabularDrift : KS + Chi-deux par colonne).

Applique un test KS sur chaque variable numérique et un test du Chi-deux sur
chaque variable catégorielle, avec correction pour tests multiples (FDR)
puisque plusieurs variables sont testées en parallèle sur le même dataset.

Usage :
    python -m src.drift_alibi.run_drift_alibi
"""
from __future__ import annotations

import json

import numpy as np
from alibi_detect.cd import TabularDrift

from src.common.data_loading import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    ensure_reports_dir,
    load_reference_and_analysis,
)


def main() -> None:
    reference, analysis = load_reference_and_analysis()
    reports_dir = ensure_reports_dir()

    ref = reference[FEATURE_COLUMNS].copy()
    cur = analysis[FEATURE_COLUMNS].copy()

    # Alibi-Detect attend des colonnes catégorielles encodées en entiers,
    # avec un mapping {index_colonne: nombre_de_modalités}.
    categories_per_feature: dict[int, int] = {}
    for col in CATEGORICAL_FEATURES:
        ref[col] = ref[col].astype("category")
        categories = ref[col].cat.categories
        cur[col] = cur[col].astype(
            __import__("pandas").CategoricalDtype(categories=categories)
        )
        ref[col] = ref[col].cat.codes
        cur[col] = cur[col].cat.codes
        col_index = FEATURE_COLUMNS.index(col)
        categories_per_feature[col_index] = len(categories)

    x_ref = ref.to_numpy(dtype=np.float64)
    x_cur = cur.to_numpy(dtype=np.float64)

    detector = TabularDrift(
        x_ref=x_ref,
        p_val=0.05,
        categories_per_feature=categories_per_feature,
        correction="fdr",
    )
    prediction = detector.predict(x_cur, drift_type="feature")

    results = []
    for i, col in enumerate(FEATURE_COLUMNS):
        results.append(
            {
                "feature": col,
                "type": "categorical" if col in CATEGORICAL_FEATURES else "numerical",
                "is_drift": bool(prediction["data"]["is_drift"][i]),
                "p_value": float(prediction["data"]["p_val"][i]),
                "threshold": float(prediction["data"]["threshold"][i]),
            }
        )

    output = {
        "dataset_drift": bool(prediction["data"]["is_drift"].any()),
        "correction": "fdr",
        "per_feature": results,
    }

    json_path = reports_dir / "alibi_detect_drift.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"Rapport Alibi-Detect généré : {json_path}")
    for r in results:
        flag = "DRIFT" if r["is_drift"] else "stable"
        print(f"  {r['feature']:<24} {flag:<6} p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
