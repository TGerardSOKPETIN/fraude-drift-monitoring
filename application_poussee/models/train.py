"""Entraine le modele champion (LightGBM) et le modele challenger
(regression logistique), sur reference.csv uniquement (periode de
developpement), avec un holdout stratifie. Evalue les deux modeles sur
le jeu de metriques demande par M. Mihoubi (Precision, Recall, F1,
ROC-AUC, PR-AUC), puis scores analysis.csv et sauvegarde tout sur
disque (modeles + predictions) pour les etapes suivantes (SHAP,
service de scoring, comparaison avec predicted_fraud_proba d'origine).
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_feature_matrix, feature_columns, TARGET_COL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
OUT_DIR = ROOT / "models" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Graine aleatoire lue depuis params.yaml (a la racine du depot) plutot que
# codee en dur, pour que DVC la detecte comme parametre versionne du pipeline
# (cf. dvc.yaml, stage "train") et qu'un changement declenche un re-entrainement.
PARAMS_PATH = ROOT.parent / "params.yaml"
with open(PARAMS_PATH) as f:
    _params = yaml.safe_load(f)
RANDOM_STATE = _params["train"]["random_state"]


def evaluate(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def main():
    reference = pd.read_csv(DATA_DIR / "reference.csv", parse_dates=["timestamp"])
    analysis = pd.read_csv(DATA_DIR / "analysis.csv", parse_dates=["timestamp"])

    X = build_feature_matrix(reference)
    y = reference[TARGET_COL].astype(int)
    cols = feature_columns()

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X[cols], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape[0]} lignes | Holdout: {X_holdout.shape[0]} lignes")
    print(f"Taux de fraude train: {y_train.mean():.4f} | holdout: {y_holdout.mean():.4f}")

    # --- Champion : LightGBM, petite recherche d'hyperparametres (pas d'Optuna,
    # perimetre "intermediaire") sur un split train/validation interne ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    param_grid = [
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400, "min_child_samples": 20},
        {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 600, "min_child_samples": 30},
        {"num_leaves": 15, "learning_rate": 0.08, "n_estimators": 300, "min_child_samples": 15},
    ]
    best_model, best_pr_auc, best_params = None, -1.0, None
    for params in param_grid:
        model = lgb.LGBMClassifier(
            objective="binary", random_state=RANDOM_STATE, verbosity=-1, **params
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_proba)
        print(f"  config {params} -> PR-AUC val = {val_pr_auc:.4f}")
        if val_pr_auc > best_pr_auc:
            best_model, best_pr_auc, best_params = model, val_pr_auc, params

    # Reentrainement du champion sur tout X_train avec les meilleurs hyperparametres
    champion = lgb.LGBMClassifier(
        objective="binary", random_state=RANDOM_STATE, verbosity=-1, **best_params
    )
    champion.fit(X_train, y_train)
    champion_holdout_proba = champion.predict_proba(X_holdout[cols])[:, 1]
    champion_metrics = evaluate(y_holdout, champion_holdout_proba)
    print("Champion (LightGBM) - holdout:", champion_metrics, "| best_params:", best_params)

    # --- Challenger : regression logistique, modele simple et interpretable,
    # entraine sur les memes features standardisees ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout[cols])

    challenger = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    challenger.fit(X_train_scaled, y_train)
    challenger_holdout_proba = challenger.predict_proba(X_holdout_scaled)[:, 1]
    challenger_metrics = evaluate(y_holdout, challenger_holdout_proba)
    print("Challenger (LogReg) - holdout:", challenger_metrics)

    # --- Sauvegarde des artefacts ---
    joblib.dump(champion, OUT_DIR / "champion_lightgbm.joblib")
    joblib.dump({"model": challenger, "scaler": scaler}, OUT_DIR / "challenger_logreg.joblib")

    metrics = {
        "champion": {"params": best_params, **champion_metrics},
        "challenger": challenger_metrics,
        "n_train": int(X_train.shape[0]),
        "n_holdout": int(X_holdout.shape[0]),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_holdout": float(y_holdout.mean()),
    }
    with open(OUT_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- Score analysis.csv avec le champion (production simulee) ---
    X_analysis = build_feature_matrix(analysis)[cols]
    analysis_scored = analysis.copy()
    analysis_scored["champion_proba"] = champion.predict_proba(X_analysis)[:, 1]
    analysis_scored["challenger_proba"] = challenger.predict_proba(
        scaler.transform(X_analysis)
    )[:, 1]
    analysis_scored.to_csv(OUT_DIR / "analysis_scored.csv", index=False)

    # Comparaison avec predicted_fraud_proba d'origine (deja presente dans
    # analysis.csv) -- sert de verification de coherence pour le memoire
    corr_champion = analysis_scored["champion_proba"].corr(analysis_scored["predicted_fraud_proba"])
    corr_challenger = analysis_scored["challenger_proba"].corr(analysis_scored["predicted_fraud_proba"])
    print(f"Correlation champion vs predicted_fraud_proba d'origine : {corr_champion:.4f}")
    print(f"Correlation challenger vs predicted_fraud_proba d'origine : {corr_challenger:.4f}")

    with open(OUT_DIR / "comparison_with_original.json", "w") as f:
        json.dump(
            {
                "correlation_champion_vs_original": float(corr_champion),
                "correlation_challenger_vs_original": float(corr_challenger),
            },
            f, indent=2,
        )

    print("\nArtefacts ecrits dans", OUT_DIR)


if __name__ == "__main__":
    main()
