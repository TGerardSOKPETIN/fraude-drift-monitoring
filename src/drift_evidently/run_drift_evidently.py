"""Détection de data drift avec Evidently AI (Data Drift preset).

La méthode statistique par colonne est choisie automatiquement par Evidently
selon le type de variable et la taille d'échantillon (KS, Chi-deux, PSI,
Jensen-Shannon, Wasserstein...). Voir README du repo racine pour le contexte.

Usage :
    python -m src.drift_evidently.run_drift_evidently
"""
from __future__ import annotations

from evidently import Report
from evidently.presets import DataDriftPreset

from src.common.data_loading import (
    FEATURE_COLUMNS,
    ensure_reports_dir,
    load_reference_and_analysis,
)


def main() -> None:
    reference, analysis = load_reference_and_analysis()
    reports_dir = ensure_reports_dir()

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=analysis[FEATURE_COLUMNS],
    )

    html_path = reports_dir / "evidently_data_drift.html"
    json_path = reports_dir / "evidently_data_drift.json"
    result.save_html(str(html_path))
    result.save_json(str(json_path))

    print(f"Rapport Evidently généré : {html_path}")
    print(f"Rapport Evidently (JSON) : {json_path}")


if __name__ == "__main__":
    main()
