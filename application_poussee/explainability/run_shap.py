"""Analyse SHAP sur le modele champion (LightGBM), pour repondre au
besoin Explainable AI identifie par M. Mihoubi (section 5 du compte
rendu Workshop 2) : quelles variables influencent le plus les
predictions, et comment le modele reagit aux nouvelles donnees
(comparaison SHAP reference vs analyse)."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.features import build_feature_matrix, feature_columns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/home/claude")
ARTIFACTS = ROOT / "models" / "artifacts"
OUT_DIR = ROOT / "reports" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_REF = "#2a78d6"
COLOR_ANA = "#eb6834"


def main():
    champion = joblib.load(ARTIFACTS / "champion_lightgbm.joblib")
    cols = feature_columns()

    reference = pd.read_csv(DATA_DIR / "reference.csv", parse_dates=["timestamp"])
    analysis = pd.read_csv(DATA_DIR / "analysis.csv", parse_dates=["timestamp"])

    X_ref = build_feature_matrix(reference)[cols]
    X_ana = build_feature_matrix(analysis)[cols]

    rng = np.random.RandomState(42)
    ref_sample = X_ref.sample(n=5000, random_state=rng)
    ana_sample = X_ana.sample(n=5000, random_state=rng)

    explainer = shap.TreeExplainer(champion)
    sv_ref = explainer.shap_values(ref_sample)
    sv_ana = explainer.shap_values(ana_sample)
    # LightGBM binary classifier via TreeExplainer renvoie parfois une liste [classe0, classe1]
    if isinstance(sv_ref, list):
        sv_ref = sv_ref[1]
        sv_ana = sv_ana[1]

    # --- Importance globale (moyenne des |SHAP|), reference vs analyse ---
    imp_ref = pd.Series(np.abs(sv_ref).mean(axis=0), index=cols).sort_values(ascending=False)
    imp_ana = pd.Series(np.abs(sv_ana).mean(axis=0), index=cols).sort_values(ascending=False)

    imp_df = pd.DataFrame({"reference": imp_ref, "analyse": imp_ana.reindex(imp_ref.index)})
    imp_df = imp_df.sort_values("reference", ascending=False)
    imp_df.to_csv(OUT_DIR / "shap_importance_reference_vs_analyse.csv")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    y_pos = np.arange(len(imp_df))
    ax.barh(y_pos - 0.2, imp_df["reference"], height=0.4, color=COLOR_REF, label="Référence")
    ax.barh(y_pos + 0.2, imp_df["analyse"], height=0.4, color=COLOR_ANA, label="Analyse")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df.index)
    ax.invert_yaxis()
    ax.set_xlabel("Importance SHAP moyenne (|valeur SHAP|)")
    ax.set_title("Importance des variables (SHAP) — référence vs analyse")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_importance_reference_vs_analyse.png")
    plt.close(fig)

    # --- Summary plot standard SHAP (sur reference) ---
    plt.figure(figsize=(8, 5), dpi=150)
    shap.summary_plot(sv_ref, ref_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_reference.png")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    shap.summary_plot(sv_ana, ana_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_analyse.png")
    plt.close()

    # --- Dependence plot sur time_since_login_min (feature au coeur du
    # diagnostic drift etabli precedemment) ---
    plt.figure(figsize=(7, 5), dpi=150)
    shap.dependence_plot(
        "time_since_login_min", sv_ref, ref_sample, show=False,
        interaction_index=None, dot_size=12,
    )
    plt.title("Effet SHAP de time_since_login_min (référence)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_dependence_time_since_login_min_reference.png")
    plt.close()

    plt.figure(figsize=(7, 5), dpi=150)
    shap.dependence_plot(
        "time_since_login_min", sv_ana, ana_sample, show=False,
        interaction_index=None, dot_size=12,
    )
    plt.title("Effet SHAP de time_since_login_min (analyse)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_dependence_time_since_login_min_analyse.png")
    plt.close()

    print("Importance SHAP (reference), classee :")
    print(imp_df)

    summary = {
        "top_feature_reference": str(imp_ref.index[0]),
        "top_feature_analyse": str(imp_ana.index[0]),
        "importance_reference": imp_ref.to_dict(),
        "importance_analyse": imp_ana.to_dict(),
    }
    with open(OUT_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nRapports SHAP ecrits dans", OUT_DIR)


if __name__ == "__main__":
    main()
