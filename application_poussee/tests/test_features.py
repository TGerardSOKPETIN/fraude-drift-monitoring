"""Tests de la feature engineering partagee champion/challenger/service
(models/features.py). Rapides et deterministes -- ne necessitent ni modele
entraine ni service demarre, contrairement aux autres fichiers de test."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_feature_matrix, feature_columns, TRANSACTION_TYPES  # noqa: E402


def _sample_df():
    return pd.DataFrame([
        {
            "time_since_login_min": 12.0,
            "transaction_amount": 50.0,
            "transaction_type": "PAYMENT",
            "is_first_transaction": True,
            "user_tenure_months": 6.0,
        },
        {
            "time_since_login_min": 3.0,
            "transaction_amount": 999.0,
            "transaction_type": None,  # valeur manquante, cf. section EDA (~6% de NA)
            "is_first_transaction": False,
            "user_tenure_months": 24.0,
        },
    ])


def test_feature_columns_matches_build_feature_matrix_columns():
    """Les colonnes retournees par feature_columns() doivent correspondre
    exactement a celles produites par build_feature_matrix (meme ordre),
    car le service d'inference indexe sur ces colonnes (cf. inference_app.py,
    X = build_feature_matrix(row)[COLS])."""
    df = _sample_df()
    matrix = build_feature_matrix(df)
    assert list(matrix.columns) == feature_columns()


def test_one_hot_encoding_known_category():
    df = _sample_df()
    matrix = build_feature_matrix(df)
    assert matrix.loc[0, "transaction_type_PAYMENT"] == 1
    for cat in TRANSACTION_TYPES:
        if cat != "PAYMENT":
            assert matrix.loc[0, f"transaction_type_{cat}"] == 0
    assert matrix.loc[0, "transaction_type_MISSING"] == 0


def test_missing_transaction_type_falls_back_to_missing_category():
    """cf. notes EDA : transaction_type a ~6% de valeurs manquantes,
    imputees par la modalite dediee 'MISSING' plutot que supprimees."""
    df = _sample_df()
    matrix = build_feature_matrix(df)
    assert matrix.loc[1, "transaction_type_MISSING"] == 1
    for cat in TRANSACTION_TYPES:
        assert matrix.loc[1, f"transaction_type_{cat}"] == 0


def test_is_first_transaction_cast_to_int():
    df = _sample_df()
    matrix = build_feature_matrix(df)
    assert matrix["is_first_transaction"].tolist() == [1, 0]
    assert matrix["is_first_transaction"].dtype.kind in ("i", "u")


def test_feature_columns_length():
    # 3 numeriques + is_first_transaction + 4 types + MISSING = 9
    assert len(feature_columns()) == 9
