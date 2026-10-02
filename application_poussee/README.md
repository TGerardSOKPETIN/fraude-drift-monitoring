# Application poussée — modèles champion/challenger, SHAP, monitoring

Code et rapports de la partie "application poussée" du mémoire. Détail complet
et interprétation dans `docs/application-poussee-modele-shap-monitoring.md`.

```
application_poussee/
├── models/
│   ├── features.py             # feature engineering partagee (entrainement + service)
│   ├── train.py                # entrainement champion (LightGBM) + challenger (regression logistique)
│   ├── training_metrics.json   # metriques holdout des deux modeles
│   └── comparison_with_original.json  # correlation avec predicted_fraud_proba d'origine
├── explainability/
│   └── run_shap.py             # analyse SHAP (importance, summary plot, dependence plot)
├── apps/
│   └── inference_app.py        # service de scoring FastAPI (/health, /predict)
├── monitoring/
│   └── run_monitoring.py       # CBPE sur le champion + comparaison performance reelle
└── reports/
    ├── shap/                   # graphiques et donnees SHAP
    └── monitoring/             # graphique et donnees CBPE vs reel
```

Les modèles entraînés (.joblib) et les CSV de prédictions volumineux ne sont pas
versionnés ici (régénérables en relançant `models/train.py`), cohérent avec le
reste du dépôt (`reports/` non versionné par défaut).

Pour exécuter : `pip install lightgbm shap fastapi "uvicorn[standard]" nannyml scikit-learn joblib`,
puis dans l'ordre `models/train.py`, `explainability/run_shap.py`, `monitoring/run_monitoring.py`.
