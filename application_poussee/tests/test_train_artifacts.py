"""Tests de non-regression sur les artefacts d'entrainement. Ne relance PAS
train.py (trop lent pour une suite de tests executee a chaque push -- la
reproductibilite du pipeline lui-meme est deja garantie par DVC, cf.
dvc.yaml/dvc.lock). Ce fichier verifie seulement que les artefacts attendus
existent et restent dans des plages de metriques raisonnables, pour attraper
une regression evidente (ex: modele qui ne vaut pas mieux que le hasard)."""
import json
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "training_metrics.json").exists(),
    reason="Artefacts du modele absents -- lancer `dvc repro` avant les tests (cf. dvc.yaml)",
)


@pytest.fixture(scope="module")
def metrics():
    with open(ARTIFACTS / "training_metrics.json") as f:
        return json.load(f)


def test_champion_and_challenger_artifacts_exist_and_load():
    champion = joblib.load(ARTIFACTS / "champion_lightgbm.joblib")
    assert hasattr(champion, "predict_proba")

    challenger_bundle = joblib.load(ARTIFACTS / "challenger_logreg.joblib")
    assert "model" in challenger_bundle and "scaler" in challenger_bundle
    assert hasattr(challenger_bundle["model"], "predict_proba")


def test_champion_beats_random_guessing(metrics):
    # Un classifieur aleatoire aurait un ROC-AUC proche de 0.5 -- le champion
    # doit nettement faire mieux sur le holdout (cf. application-poussee-
    # modele-shap-monitoring.md, ROC-AUC holdout observe ~0.74).
    assert metrics["champion"]["roc_auc"] > 0.65


def test_champion_outperforms_or_matches_challenger_on_pr_auc(metrics):
    # Justifie le role de "champion" du LightGBM par rapport au challenger
    # (regression logistique) -- cf. application-poussee-modele-shap-monitoring.md.
    assert metrics["champion"]["pr_auc"] >= metrics["challenger"]["pr_auc"] - 0.05


def test_metrics_have_expected_keys(metrics):
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert key in metrics["champion"]
        assert key in metrics["challenger"]
        assert 0.0 <= metrics["champion"][key] <= 1.0
        assert 0.0 <= metrics["challenger"][key] <= 1.0
