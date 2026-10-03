"""Service de scoring temps reel, leger (pas de Redis ni de Docker complexe,
perimetre "intermediaire" convenu). Charge le champion (LightGBM) et le
challenger (regression logistique) au demarrage, expose /predict (scoring
avec le champion, challenger renvoye a titre de comparaison, logique
champion/challenger sans routage shadow complet), /health et /model-info.

Chaque appel a /predict est journalise (entrees + sorties + version de
modele + latence) dans une base SQLite, afin que le monitoring (drift,
performance) puisse un jour tourner sur du vrai trafic plutot que sur un
fichier statique (voir monitoring/run_monitoring.py). Minimisation des
donnees : on ne journalise que les features effectivement utilisees par le
modele, rien de plus (pas d'identifiant client, pas de donnees brutes
superflues) -- cf. architecture-monitoring-alertes-retraining.md et
docs/architecture Workshop 3 section 9.
"""
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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

# Chemin de la base de logs : surchargeable par variable d'environnement
# (utile pour les tests), par defaut un dossier "logs/" a la racine de
# l'app -- monte en volume par docker-compose.yml pour que les logs
# survivent au redemarrage du conteneur et restent lisibles depuis
# l'hote (par les scripts de monitoring, executes hors conteneur).
LOG_DB_PATH = Path(os.environ.get("LOG_DB_PATH", Path(__file__).resolve().parent.parent / "logs" / "predictions.db"))

app = FastAPI(
    title="Sentinel-lite — scoring fraude (Poundbank)",
    description="Service de scoring minimal, champion LightGBM + challenger "
    "regression logistique, construit pour le memoire (perimetre reduit : "
    "pas de feature store Redis, pas de conteneurisation Docker avancee).",
    version="0.1.0",
)

_state = {}


def _init_log_db(path: Path) -> None:
    """Note d'implementation : `with sqlite3.connect(path) as conn:` ne
    ferme PAS la connexion a la sortie du bloc (seulement commit/rollback de
    la transaction, specificite du module sqlite3) -- sans fermeture
    explicite, chaque (re)demarrage du service laisse une connexion ouverte.
    Sans impact notable en production (un seul appel par cycle de vie du
    processus), mais bloquant pour les tests, qui redemarrent le service a
    chaque test (cf. tests/test_inference_app.py) : le fichier de logs reste
    verrouille sous Windows meme apres la fin de la suite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                model_version TEXT NOT NULL,
                time_since_login_min REAL,
                transaction_amount REAL,
                transaction_type TEXT,
                is_first_transaction INTEGER,
                user_tenure_months REAL,
                fraud_probability_champion REAL,
                fraud_probability_challenger REAL,
                decision TEXT,
                threshold REAL,
                latency_ms REAL,
                is_fraud_confirmed INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def _log_connection():
    conn = sqlite3.connect(LOG_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
def load_models():
    _state["champion"] = joblib.load(ARTIFACTS / "champion_lightgbm.joblib")
    challenger_bundle = joblib.load(ARTIFACTS / "challenger_logreg.joblib")
    _state["challenger"] = challenger_bundle["model"]
    _state["scaler"] = challenger_bundle["scaler"]
    _state["model_version"] = "champion_lightgbm_v1"
    _state["loaded_at"] = datetime.now(timezone.utc).isoformat()
    _init_log_db(LOG_DB_PATH)


class Transaction(BaseModel):
    time_since_login_min: float = Field(..., ge=0, description="Minutes depuis la derniere connexion")
    transaction_amount: float = Field(..., ge=0, description="Montant de la transaction (£)")
    transaction_type: str = Field(..., description="CASH-IN, CASH-OUT, PAYMENT ou TRANSFER")
    is_first_transaction: bool = Field(..., description="Premiere transaction de l'utilisateur")
    user_tenure_months: float = Field(..., ge=0, description="Anciennete du compte (mois)")


class PredictionResponse(BaseModel):
    request_id: str
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


@app.get("/model-info")
def model_info():
    return {
        "model_version": _state.get("model_version"),
        "champion_algorithm": "LightGBM",
        "challenger_algorithm": "LogisticRegression",
        "threshold": THRESHOLD,
        "feature_columns": COLS,
        "loaded_at": _state.get("loaded_at"),
        "log_db_path": str(LOG_DB_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    row = pd.DataFrame([tx.model_dump()])
    X = build_feature_matrix(row)[COLS]

    champion_proba = float(_state["champion"].predict_proba(X)[:, 1][0])
    X_scaled = _state["scaler"].transform(X)
    challenger_proba = float(_state["challenger"].predict_proba(X_scaled)[:, 1][0])

    decision = "FRAUDE_SUSPECTEE" if champion_proba >= THRESHOLD else "LEGITIME"
    latency_ms = (time.perf_counter() - t0) * 1000

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    with _log_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                request_id, timestamp_utc, model_version,
                time_since_login_min, transaction_amount, transaction_type,
                is_first_transaction, user_tenure_months,
                fraud_probability_champion, fraud_probability_challenger,
                decision, threshold, latency_ms, is_fraud_confirmed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                request_id, timestamp_utc, _state["model_version"],
                tx.time_since_login_min, tx.transaction_amount, tx.transaction_type,
                int(tx.is_first_transaction), tx.user_tenure_months,
                round(champion_proba, 6), round(challenger_proba, 6),
                decision, THRESHOLD, round(latency_ms, 3),
            ),
        )
        conn.commit()

    return PredictionResponse(
        request_id=request_id,
        fraud_probability_champion=round(champion_proba, 4),
        fraud_probability_challenger=round(challenger_proba, 4),
        decision=decision,
        threshold=THRESHOLD,
        model_version=_state["model_version"],
        latency_ms=round(latency_ms, 3),
    )
