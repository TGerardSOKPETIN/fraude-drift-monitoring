"""Applique le meme protocole de monitoring (CBPE) que celui utilise sur
le modele d'origine, mais sur les predictions de notre propre champion,
et le confronte a la performance REELLE mensuelle (calculable ici car
is_fraud est disponible dans analysis.csv, ce qui permet de valider la
qualite de l'estimation CBPE elle-meme)."""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nannyml as nml
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_feature_matrix, feature_columns, TARGET_COL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/home/claude")
ARTIFACTS = ROOT / "models" / "artifacts"
OUT_DIR = ROOT / "reports" / "monitoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
COLOR_REF = "#2a78d6"
COLOR_ANA = "#eb6834"


def main():
    champion = joblib.load(ARTIFACTS / "champion_lightgbm.joblib")
    cols = feature_columns()

    reference = pd.read_csv(DATA_DIR / "reference.csv", parse_dates=["timestamp"])
    analysis_scored = pd.read_csv(ARTIFACTS / "analysis_scored.csv", parse_dates=["timestamp"])

    X = build_feature_matrix(reference)
    y = reference[TARGET_COL].astype(int)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X[cols], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    holdout_ts = reference.loc[X_holdout.index, "timestamp"]

    ref_df = pd.DataFrame({
        "timestamp": holdout_ts.values,
        "y_pred_proba": champion.predict_proba(X_holdout)[:, 1],
        "is_fraud": y_holdout.values,
    })
    ref_df["y_pred"] = (ref_df["y_pred_proba"] >= 0.5).astype(int)
    ref_df = ref_df.sort_values("timestamp").reset_index(drop=True)

    ana_df = analysis_scored[["timestamp", "champion_proba", "is_fraud"]].rename(
        columns={"champion_proba": "y_pred_proba"}
    )
    ana_df["y_pred"] = (ana_df["y_pred_proba"] >= 0.5).astype(int)
    ana_df = ana_df.sort_values("timestamp").reset_index(drop=True)

    estimator = nml.CBPE(
        problem_type="classification_binary",
        y_pred_proba="y_pred_proba", y_pred="y_pred", y_true="is_fraud",
        metrics=["roc_auc", "precision", "recall", "f1"],
        chunk_period="M", timestamp_column_name="timestamp",
    )
    estimator.fit(ref_df)
    est_df = estimator.estimate(ana_df).to_df()
    est_df = est_df[est_df[("chunk", "period")] == "analysis"].copy()
    est_df["month"] = pd.to_datetime(est_df[("chunk", "start_date")]).dt.strftime("%Y-%m")

    # Performance reelle mensuelle
    ana_df["month"] = pd.to_datetime(ana_df["timestamp"]).dt.to_period("M").astype(str)
    real_rows = []
    for month, grp in ana_df.groupby("month"):
        real_rows.append({
            "month": month,
            "precision": precision_score(grp["is_fraud"], grp["y_pred"]),
            "recall": recall_score(grp["is_fraud"], grp["y_pred"]),
            "f1": f1_score(grp["is_fraud"], grp["y_pred"]),
            "roc_auc": roc_auc_score(grp["is_fraud"], grp["y_pred_proba"]),
        })
    real_df = pd.DataFrame(real_rows)

    comparison = real_df.copy()
    for metric in ["roc_auc", "precision", "recall", "f1"]:
        comparison[f"{metric}_estime_cbpe"] = est_df[(metric, "value")].values
        comparison[f"{metric}_alerte_cbpe"] = est_df[(metric, "alert")].values
        comparison = comparison.rename(columns={metric: f"{metric}_reel"})
    comparison.to_csv(OUT_DIR / "comparaison_cbpe_vs_reel_champion.csv", index=False)
    print(comparison.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    axes = axes.ravel()
    x = pd.to_datetime(comparison["month"])
    for ax, metric in zip(axes, ["roc_auc", "precision", "recall", "f1"]):
        ax.plot(x, comparison[f"{metric}_estime_cbpe"], marker="o", color=COLOR_REF, label="CBPE (estimé)")
        ax.plot(x, comparison[f"{metric}_reel"], marker="s", color=COLOR_ANA, label="Réel")
        ax.set_title(metric.upper())
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Champion (modèle entraîné pour cette partie) — performance estimée (CBPE) vs réelle, par mois")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cbpe_vs_reel_champion.png")
    plt.close(fig)

    n_alerts = sum(comparison[f"{m}_alerte_cbpe"].sum() for m in ["roc_auc", "precision", "recall", "f1"])
    print(f"\nNombre total d'alertes CBPE declenchees (4 metriques x 8 mois) : {n_alerts}")
    print("\nRapports ecrits dans", OUT_DIR)


if __name__ == "__main__":
    main()
