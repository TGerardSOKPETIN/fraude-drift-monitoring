"""Service de scoring temps reel, leger (pas de Redis ni de Docker,
perimetre "intermediaire" convenu). Charge le champion (LightGBM) et le
challenger (regression logistique) au demarrage, expose /predict (scoring
avec le champion, challenger renvoye a titre de comparaison, logique
champion/challenger sans routage shadow complet) et /health."""
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_feature_matrix, feature_columns  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
COLS = feature_columns()

app = FastAPI(
    title="Sentinel-lite — scoring fraude (Poundbank)",
    description="Service de scoring minimal, champion LightGBM + challenger "
    "regression logistique, construit pour le memoire (perimetre reduit : "
    "pas de feature store Redis, pas de conteneurisation Docker).",
    version="0.1.0",
)

_state = {}


@app.on_event("startup")
def load_models():
    _state["champion"] = joblib.load(ARTIFACTS / "champion_lightgbm.joblib")
    challenger_bundle = joblib.load(ARTIFACTS / "challenger_logreg.joblib")
    _state["challenger"] = challenger_bundle["model"]
    _state["scaler"] = challenger_bundle["scaler"]
    _state["model_version"] = "champion_lightgbm_v1"


class Transaction(BaseModel):
    time_since_login_min: float = Field(..., ge=0, description="Minutes depuis la derniere connexion")
    transaction_amount: float = Field(..., ge=0, description="Montant de la transaction (£)")
    transaction_type: str = Field(..., description="CASH-IN, CASH-OUT, PAYMENT ou TRANSFER")
    is_first_transaction: bool = Field(..., description="Premiere transaction de l'utilisateur")
    user_tenure_months: float = Field(..., ge=0, description="Anciennete du compte (mois)")


class PredictionResponse(BaseModel):
    fraud_probability_champion: float
    fraud_probability_challenger: float
    decision: str
    threshold: float
    model_version: str
    latency_ms: float


THRESHOLD = 0.5


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_state.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    t0 = time.perf_counter()
    row = pd.DataFrame([tx.model_dump()])
    X = build_feature_matrix(row)[COLS]

    champion_proba = float(_state["champion"].predict_proba(X)[:, 1][0])
    X_scaled = _state["scaler"].transform(X)
    challenger_proba = float(_state["challenger"].predict_proba(X_scaled)[:, 1][0])

    decision = "FRAUDE_SUSPECTEE" if champion_proba >= THRESHOLD else "LEGITIME"
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        fraud_probability_champion=round(champion_proba, 4),
        fraud_probability_challenger=round(challenger_proba, 4),
        decision=decision,
        threshold=THRESHOLD,
        model_version=_state["model_version"],
        latency_ms=round(latency_ms, 3),
    )
