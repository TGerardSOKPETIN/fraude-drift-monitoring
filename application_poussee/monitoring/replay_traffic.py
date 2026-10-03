"""Rejoue un echantillon de analysis.csv (donnees de production simulees)
contre le service /predict, pour peupler la base de logs (voir
apps/inference_app.py) avec un volume de trafic realiste avant de lancer
la detection de drift sur ces logs (run_drift_on_logs.py). Sans ce script,
la base de logs ne contient que les quelques appels manuels de test.

Le service doit etre demarre au prealable (docker compose up -d).

Usage :
    python application_poussee/monitoring/replay_traffic.py --n 300
    python application_poussee/monitoring/replay_traffic.py --n 300 --month 2019-06
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
API_URL = "http://localhost:8000/predict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Nombre de transactions a rejouer")
    parser.add_argument("--month", type=str, default=None,
                         help="Mois a rejouer, format AAAA-MM (ex: 2019-06). "
                              "Sans cette option, echantillon sur toute la periode d'analyse.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", type=str, default=API_URL)
    args = parser.parse_args()

    analysis = pd.read_csv(DATA_DIR / "analysis.csv", parse_dates=["timestamp"])
    if args.month:
        analysis = analysis[analysis["timestamp"].dt.strftime("%Y-%m") == args.month]
        if analysis.empty:
            print(f"Aucune ligne pour le mois {args.month} dans analysis.csv")
            sys.exit(1)

    sample = analysis.sample(n=min(args.n, len(analysis)), random_state=args.seed)
    print(f"Rejeu de {len(sample)} transactions vers {args.api_url} ...")

    n_ok, n_err = 0, 0
    for _, row in sample.iterrows():
        payload = {
            "time_since_login_min": float(row["time_since_login_min"]),
            "transaction_amount": float(row["transaction_amount"]),
            "transaction_type": str(row["transaction_type"]) if pd.notna(row["transaction_type"]) else "PAYMENT",
            "is_first_transaction": bool(row["is_first_transaction"]),
            "user_tenure_months": float(row["user_tenure_months"]),
        }
        try:
            resp = requests.post(args.api_url, json=payload, timeout=5)
            resp.raise_for_status()
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  Erreur sur une requete : {e}")

    print(f"\n{n_ok} requetes envoyees avec succes, {n_err} erreurs.")
    print("Lance maintenant : python application_poussee/monitoring/run_drift_on_logs.py")


if __name__ == "__main__":
    main()
