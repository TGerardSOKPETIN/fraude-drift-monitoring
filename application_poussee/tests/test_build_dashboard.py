"""Tests du tableau de bord (monitoring/build_dashboard.py). Le script n'utilise
que la bibliotheque standard, ces tests n'ont donc besoin ni du modele ni
d'Evidently : ils fabriquent de petits fichiers de resultats factices."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.build_dashboard import build, compute_status, parse_drift  # noqa: E402

META = {"commit": "abc1234", "branch": "main", "generated_at": "2026-01-01 00:00 UTC", "run_url": ""}


def _drift_json(values):
    metrics = []
    for column, distance in values.items():
        metrics.append({
            "metric_name": f"ValueDrift(column={column})",
            "config": {"type": "evidently:metric_v2:ValueDrift", "column": column,
                       "method": "Wasserstein distance (normed)", "threshold": 0.1},
            "value": distance,
        })
    # Metrique de synthese d'Evidently : doit etre ignoree par parse_drift
    metrics.append({"metric_name": "DriftedColumnsCount",
                    "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                    "value": {"count": 1.0, "share": 0.2}})
    return {"metrics": metrics, "tests": []}


def _metrics_json():
    block = {"precision": 0.8, "recall": 0.5, "f1": 0.6, "roc_auc": 0.7, "pr_auc": 0.75}
    return {"champion": {"params": {"n_estimators": 400}, **block}, "challenger": dict(block),
            "n_train": 100, "n_holdout": 25}


def test_parse_drift_ignores_summary_metric_and_flags_columns():
    columns = parse_drift(_drift_json({"a": 0.14, "b": 0.02}))
    assert [c["column"] for c in columns] == ["a", "b"]
    assert [c["drifted"] for c in columns] == [True, False]


def test_alert_triggers_at_threshold_share():
    columns = parse_drift(_drift_json({"a": 0.14, "b": 0.02, "c": 0.01, "d": 0.03, "e": 0.0}))
    assert compute_status(columns, 0.2)["alert"] is True   # 1/5 = 20 %
    assert compute_status(columns, 0.3)["alert"] is False


def test_no_columns_gives_no_alert():
    assert compute_status([], 0.2)["alert"] is False


def _run(tmp_path, values):
    m, d, out = tmp_path / "m.json", tmp_path / "d.json", tmp_path / "out" / "dashboard.html"
    m.write_text(json.dumps(_metrics_json()))
    d.write_text(json.dumps(_drift_json(values)))
    status = build(m, d, out, 0.2, META)
    return status, out.read_text(encoding="utf-8")


def test_dashboard_shows_alert_and_version(tmp_path):
    status, page = _run(tmp_path, {"time_since_login_min": 0.14, "transaction_amount": 0.04})
    assert status["alert"] is True
    assert "ALERTE DRIFT" in page
    assert "time_since_login_min" in page
    assert "abc1234" in page


def test_dashboard_calm_when_no_drift(tmp_path):
    status, page = _run(tmp_path, {"time_since_login_min": 0.02, "transaction_amount": 0.04})
    assert status["alert"] is False
    assert "Pas d'alerte" in page
    assert "ALERTE DRIFT" not in page
