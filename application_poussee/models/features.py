"""Feature engineering partagée champion/challenger, cohérente avec le
notebook d'exploration (mêmes colonnes, mêmes types) afin que les
prédictions produites ici restent comparables aux résultats déjà établis
dans exploration_drift_detection.ipynb."""
import pandas as pd

NUMERICAL_FEATURES = ["time_since_login_min", "transaction_amount", "user_tenure_months"]
CATEGORICAL_FEATURES = ["transaction_type"]
BOOLEAN_FEATURES = ["is_first_transaction"]
TARGET_COL = "is_fraud"

TRANSACTION_TYPES = ["CASH-IN", "CASH-OUT", "PAYMENT", "TRANSFER"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode transaction_type, cast is_first_transaction en int,
    impute les valeurs manquantes de transaction_type par une modalité
    'MISSING' dédiée (cohérent avec le constat EDA : ~6% de NA sur cette
    seule colonne, concentrés de façon comparable entre référence et
    analyse)."""
    out = pd.DataFrame(index=df.index)
    out[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES]
    out["is_first_transaction"] = df["is_first_transaction"].astype(int)

    tx_type = df["transaction_type"].fillna("MISSING")
    for cat in TRANSACTION_TYPES:
        out[f"transaction_type_{cat}"] = (tx_type == cat).astype(int)
    out["transaction_type_MISSING"] = (tx_type == "MISSING").astype(int)

    return out


def feature_columns() -> list[str]:
    cols = list(NUMERICAL_FEATURES) + ["is_first_transaction"]
    cols += [f"transaction_type_{c}" for c in TRANSACTION_TYPES]
    cols += ["transaction_type_MISSING"]
    return cols
