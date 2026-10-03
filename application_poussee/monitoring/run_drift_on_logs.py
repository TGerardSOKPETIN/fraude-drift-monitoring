"""Compare les features journalisees en production (base SQLite alimentee
par l'API a chaque /predict, voir apps/inference_app.py) au jeu de
reference (reference.csv), avec Evidently -- meme outil et meme logique
que src/drift_evidently/run_drift_evidently.py, mais applique ici a du
vrai trafic journalise plutot qu'a un fichier statique (analysis.csv).

Regle d'alerte (cf. docs/architecture-monitoring-alertes-retraining.md) :
alerte si la part de colonnes en drift depasse le seuil --threshold
(0.3 par defaut, a ajuster avec les equipes metier comme mentionne dans
le document Workshop 3, section 15 "a finaliser").

Prerequis : le service doit avoir recu du trafic (voir replay_traffic.py
si la base de logs est trop petite).

Usage :
    python application_poussee/monitoring/run_drift_on_logs.py
    python application_poussee/monitoring/run_drift_on_logs.py --min-rows 50 --threshold 0.3
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
LOG_DB = ROOT / "logs" / "predictions.db"
OUT_DIR = ROOT / "reports" / "monitoring"

FEATURE_COLUMNS = [
    "time_since_login_min", "transaction_amount", "transaction_type",
    "is_first_transaction", "user_tenure_months",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=30,
                         help="Nombre minimal de lignes journalisees pour lancer l'analyse")
    parser.add_argument("--threshold", type=float, default=0.2,
                         help="Part de colonnes en drift au-dela de laquelle une alerte est declenchee. "
                              "Avec seulement 5 features, 0.2 revient a declencher des qu'une seule "
                              "variable derive significativement -- plus realiste ici qu'un seuil a 0.3 "
                              "(2 variables sur 5), qui ratereait un cas a une seule variable dominante "
                              "comme celui identifie sur ce jeu de donnees (time_since_login_min).")
    args = parser.parse_args()

    if not LOG_DB.exists():
        print(f"Base de logs introuvable : {LOG_DB}")
        print("Demarre le service (docker compose up -d) et envoie des requetes avant de relancer ce script.")
        sys.exit(1)

    with sqlite3.connect(LOG_DB) as conn:
        production = pd.read_sql("SELECT * FROM predictions", conn)

    if len(production) < args.min_rows:
        print(f"Seulement {len(production)} ligne(s) journalisee(s) (minimum requis : {args.min_rows}).")
        print("Genere du trafic avec : python application_poussee/monitoring/replay_traffic.py --n 300")
        sys.exit(1)

    production["is_first_transaction"] = production["is_first_transaction"].astype(bool)

    reference = pd.read_csv(DATA_DIR / "reference.csv", parse_dates=["timestamp"])

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=production[FEATURE_COLUMNS],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / "drift_production_log.html"
    json_path = OUT_DIR / "drift_production_log.json"
    result.save_html(str(html_path))
    result.save_json(str(json_path))

    result_dict = result.dict()
    drifted_count, share = None, None
    for metric in result_dict.get("metrics", []):
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            drifted_count = metric["value"]["count"]
            share = metric["value"]["share"]
            break

    print(f"\nLignes de production analysees : {len(production)}")
    print(f"Rapport detaille ecrit dans : {html_path}")

    if drifted_count is not None:
        print(f"Colonnes en drift : {int(drifted_count)}/{len(FEATURE_COLUMNS)} ({share:.1%})")
        if share >= args.threshold:
            print(f"\n*** ALERTE DRIFT *** (seuil {args.threshold:.0%} depasse)")
        else:
            print(f"\nPas d'alerte (seuil {args.threshold:.0%} non atteint)")
    else:
        print("Impossible d'extraire le nombre de colonnes en drift du resume -- ouvrir le rapport HTML pour le detail.")


if __name__ == "__main__":
    main()
