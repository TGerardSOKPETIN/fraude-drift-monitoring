"""Chargement et définition des colonnes du dataset Poundbank.

Partagé par les trois scripts de détection de drift (Evidently, Alibi-Detect,
NannyML) pour garder une seule source de vérité sur les noms de colonnes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

TIMESTAMP_COL = "timestamp"
TARGET_COL = "is_fraud"
PREDICTION_COL = "predicted_fraud"
PREDICTED_PROBA_COL = "predicted_fraud_proba"

# Variables numériques continues, candidates aux tests KS / Wasserstein
NUMERICAL_FEATURES = [
    "time_since_login_min",
    "transaction_amount",
    "user_tenure_months",
]

# Variables catégorielles (ou binaires traitées comme telles), candidates au Chi-deux
CATEGORICAL_FEATURES = [
    "transaction_type",
    "is_first_transaction",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_reference_and_analysis(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge reference.csv et analysis.csv depuis data_dir.

    Lève une erreur explicite si les fichiers ne sont pas encore présents
    (voir data/README.md pour où les placer).
    """
    reference_path = data_dir / "reference.csv"
    analysis_path = data_dir / "analysis.csv"

    for path in (reference_path, analysis_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}. "
                "Place reference.csv et analysis.csv dans data/ avant de lancer ce script "
                "(voir data/README.md)."
            )

    reference = pd.read_csv(reference_path, parse_dates=[TIMESTAMP_COL])
    analysis = pd.read_csv(analysis_path, parse_dates=[TIMESTAMP_COL])
    return reference, analysis


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR
