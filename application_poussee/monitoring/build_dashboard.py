"""Genere le tableau de bord de surveillance du modele (une page HTML
autonome, sans JavaScript ni dependance externe) a partir de deux fichiers :

  - models/artifacts/training_metrics.json : metriques du champion et du
    challenger sur le jeu de test (produit par train.py via `dvc repro`) ;
  - reports/monitoring/drift_production_log.json : mesures de drift par
    variable sur les logs de production (produit par run_drift_on_logs.py).

Regle d'alerte identique a run_drift_on_logs.py : une variable est "en
drift" quand sa distance atteint le seuil Evidently de la colonne, et une
alerte est declenchee quand la part de variables en drift atteint
--alert-share (0.2 par defaut).

Le script n'utilise que la bibliotheque standard : il tourne tel quel dans
la CI (job "monitor") et dans les tests, sans installer Evidently.

Usage :
    python application_poussee/monitoring/build_dashboard.py
"""
import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "models" / "artifacts" / "training_metrics.json"
DEFAULT_DRIFT = ROOT / "reports" / "monitoring" / "drift_production_log.json"
DEFAULT_OUT = ROOT / "reports" / "monitoring" / "dashboard.html"
DEFAULT_ALERT_SHARE = 0.2

METRIC_LABELS = [
    ("precision", "Précision"),
    ("recall", "Rappel"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]


def parse_drift(drift_json: dict) -> list:
    """Extrait, pour chaque variable, la distance mesuree et son seuil."""
    columns = []
    for metric in drift_json.get("metrics", []):
        config = metric.get("config", {})
        if not str(config.get("type", "")).endswith("ValueDrift"):
            continue
        distance = float(metric["value"])
        threshold = float(config["threshold"])
        columns.append({
            "column": config["column"],
            "method": config.get("method", ""),
            "distance": distance,
            "threshold": threshold,
            "drifted": distance >= threshold,
        })
    return columns


def compute_status(columns: list, alert_share: float) -> dict:
    n_total = len(columns)
    n_drift = sum(1 for c in columns if c["drifted"])
    share = n_drift / n_total if n_total else 0.0
    return {
        "n_total": n_total,
        "n_drift": n_drift,
        "share": share,
        "alert_share": alert_share,
        "alert": share >= alert_share,
    }


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def _metrics_rows(metrics: dict) -> str:
    champion, challenger = metrics["champion"], metrics["challenger"]
    rows = []
    for key, label in METRIC_LABELS:
        c, ch = champion[key], challenger[key]
        c_cls = ' class="best"' if c >= ch else ""
        ch_cls = ' class="best"' if ch > c else ""
        rows.append(
            f"<tr><th scope='row'>{label}</th>"
            f"<td{c_cls}>{_num(c)}</td><td{ch_cls}>{_num(ch)}</td></tr>"
        )
    return "\n".join(rows)


def _drift_rows(columns: list) -> str:
    if not columns:
        return "<tr><td colspan='5'>Aucune mesure de drift disponible.</td></tr>"
    scale = max(max(c["threshold"] * 2, c["distance"] * 1.15) for c in columns)
    rows = []
    for c in sorted(columns, key=lambda c: c["distance"] / c["threshold"], reverse=True):
        fill = min(c["distance"] / scale, 1.0) * 100
        mark = min(c["threshold"] / scale, 1.0) * 100
        state = "En drift" if c["drifted"] else "Stable"
        cls = "drift" if c["drifted"] else "ok"
        rows.append(
            "<tr>"
            f"<th scope='row'>{html.escape(c['column'])}</th>"
            f"<td class='method'>{html.escape(c['method'])}</td>"
            f"<td class='num'>{_num(c['distance'])} / {_num(c['threshold'], 2)}</td>"
            "<td class='barcell'><div class='track'>"
            f"<div class='fill {cls}' style='width:{fill:.1f}%'></div>"
            f"<div class='mark' style='left:{mark:.1f}%' title='seuil'></div>"
            "</div></td>"
            f"<td><span class='badge {cls}'>{state}</span></td>"
            "</tr>"
        )
    return "\n".join(rows)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--text:#1c1c1a;--muted:#6b6b66;--line:#e3e3de;
--ok:#1f7a4d;--ok-bg:#e6f3ec;--bad:#b3261e;--bad-bg:#fbe9e7;--track:#ececE7;--mark:#1c1c1a}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#1f1f1c;--text:#ecece8;
--muted:#9a9a93;--line:#34342f;--ok:#6fcf9b;--ok-bg:#183326;--bad:#ff8a80;--bad-bg:#3a1f1c;
--track:#2c2c28;--mark:#ecece8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:0 0 12px}
.meta{color:var(--muted);font-size:13px;margin:0 0 20px}
.meta a{color:inherit}
.banner{border-radius:10px;padding:14px 16px;margin-bottom:20px;border:1px solid var(--line)}
.banner.alert{background:var(--bad-bg);border-color:var(--bad)}
.banner.calm{background:var(--ok-bg);border-color:var(--ok)}
.banner strong{display:block;font-size:17px}
.banner.alert strong{color:var(--bad)}.banner.calm strong{color:var(--ok)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin-bottom:20px;overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
thead th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
td.method{color:var(--muted);font-size:13px}
td.best{font-weight:700}
.note{color:var(--muted);font-size:13px;margin:10px 0 0}
.barcell{min-width:140px;width:30%}
@media (max-width:640px){.method{display:none}}
.track{position:relative;height:10px;background:var(--track);border-radius:5px}
.fill{height:100%;border-radius:5px}
.fill.ok{background:var(--ok)}.fill.drift{background:var(--bad)}
.mark{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--mark)}
.badge{display:inline-block;white-space:nowrap;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}
.badge.ok{background:var(--ok-bg);color:var(--ok)}.badge.drift{background:var(--bad-bg);color:var(--bad)}
"""


def render_html(metrics: dict, columns: list, status: dict, meta: dict) -> str:
    if status["alert"]:
        banner_cls = "alert"
        banner_title = "ALERTE DRIFT"
    else:
        banner_cls = "calm"
        banner_title = "Pas d'alerte"
    banner_text = (
        f"{status['n_drift']} variable(s) sur {status['n_total']} en drift "
        f"({_pct(status['share'])}) ; l'alerte se déclenche à partir de "
        f"{_pct(status['alert_share'])}."
    )

    params = metrics["champion"].get("params", {})
    params_txt = " · ".join(f"{k} = {v}" for k, v in params.items())

    meta_parts = [f"Version <code>{html.escape(meta['commit'])}</code>"]
    if meta.get("branch"):
        meta_parts.append(f"branche {html.escape(meta['branch'])}")
    meta_parts.append(f"généré le {html.escape(meta['generated_at'])}")
    if meta.get("run_url"):
        meta_parts.append(f"<a href='{html.escape(meta['run_url'])}'>exécution CI</a>")
    meta_line = " · ".join(meta_parts)

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tableau de bord, surveillance du modèle</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Surveillance du modèle de détection de fraude</h1>
<p class="meta">{meta_line}</p>

<div class="banner {banner_cls}" role="status">
<strong>{banner_title}</strong>
{html.escape(banner_text)}
</div>

<section>
<h2>Performance du modèle (jeu de test)</h2>
<table>
<thead><tr><th>Métrique</th><th>Champion (LightGBM)</th><th>Challenger (régression logistique)</th></tr></thead>
<tbody>
{_metrics_rows(metrics)}
</tbody>
</table>
<p class="note">En gras : meilleure valeur de la ligne. Hyperparamètres du champion : {html.escape(params_txt)}.
Entraînement sur {metrics.get('n_train', '?')} lignes, test sur {metrics.get('n_holdout', '?')} lignes.</p>
</section>

<section>
<h2>Drift des variables en production</h2>
<table>
<thead><tr><th>Variable</th><th class='method'>Méthode</th><th>Distance / seuil</th><th>Écart au seuil</th><th>État</th></tr></thead>
<tbody>
{_drift_rows(columns)}
</tbody>
</table>
<p class="note">La barre montre la distance mesurée ; le trait noir marque le seuil de la variable.
Mesure : jeu de référence comparé aux transactions journalisées par le service.</p>
</section>
</main>
</body>
</html>
"""


def build(metrics_path: Path, drift_path: Path, out_path: Path,
          alert_share: float, meta: dict) -> dict:
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(drift_path, encoding="utf-8") as f:
        drift = json.load(f)

    columns = parse_drift(drift)
    status = compute_status(columns, alert_share)
    page = render_html(metrics, columns, status, meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return status


def _meta_from_env() -> dict:
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else ""
    return {
        "commit": sha,
        "branch": os.environ.get("GITHUB_REF_NAME", ""),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_url": run_url,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--drift", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alert-share", type=float, default=DEFAULT_ALERT_SHARE)
    args = parser.parse_args()

    status = build(args.metrics, args.drift, args.out, args.alert_share, _meta_from_env())
    label = "ALERTE DRIFT" if status["alert"] else "Pas d'alerte"
    print(f"Tableau de bord ecrit dans : {args.out}")
    print(f"{label} : {status['n_drift']}/{status['n_total']} variables en drift ({_pct(status['share'])})")


if __name__ == "__main__":
    main()
