"""Tests du service de scoring (apps/inference_app.py). Necessitent que les
artefacts du champion/challenger existent deja sur disque
(models/artifacts/*.joblib) -- produits par `dvc repro` (cf. dvc.yaml).
Dans la CI, le job "test" doit donc s'executer apres un job "train" (ou un
`dvc pull`/`dvc repro`), jamais avant -- cf. etat-avancement-memoire.md."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apps.inference_app import app  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "champion_lightgbm.joblib").exists(),
    reason="Artefacts du modele absents -- lancer `dvc repro` avant les tests (cf. dvc.yaml)",
)

VALID_PAYLOAD = {
    "time_since_login_min": 15.0,
    "transaction_amount": 120.5,
    "transaction_type": "PAYMENT",
    "is_first_transaction": False,
    "user_tenure_months": 24.0,
}


@pytest.fixture()
def client():
    # Le context manager declenche le lifespan (startup/shutdown), donc
    # load_models() tourne avant le premier test -- sans ca, _state est
    # vide et /predict echoue avec une KeyError.
    with TestClient(app) as c:
        yield c


def test_health_reports_models_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "champion" in body["models_loaded"]
    assert "challenger" in body["models_loaded"]


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion_algorithm"] == "LightGBM"
    assert body["challenger_algorithm"] == "LogisticRegression"
    assert len(body["feature_columns"]) == 9


def test_predict_valid_payload_returns_coherent_response(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability_champion"] <= 1.0
    assert 0.0 <= body["fraud_probability_challenger"] <= 1.0
    assert body["decision"] in ("FRAUDE_SUSPECTEE", "LEGITIME")
    assert body["threshold"] == 0.5
    assert body["model_version"] == "champion_lightgbm_v1"
    assert body["latency_ms"] >= 0


def test_predict_decision_consistent_with_threshold(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    body = resp.json()
    expected = "FRAUDE_SUSPECTEE" if body["fraud_probability_champion"] >= body["threshold"] else "LEGITIME"
    assert body["decision"] == expected


def test_predict_missing_field_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    del payload["transaction_amount"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    payload["transaction_amount"] = "pas-un-nombre"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_amount_rejected(client):
    """transaction_amount a une contrainte ge=0 (cf. Transaction, apps/inference_app.py)."""
    payload = dict(VALID_PAYLOAD)
    payload["transaction_amount"] = -10.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
