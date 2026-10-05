# Application poussée : service de scoring, modèles et monitoring

Code et rapports du service de détection de fraude et de sa surveillance. Vue d'ensemble, architecture et démarrage rapide dans le [README principal](../README.md) ; interprétation détaillée des résultats dans [`docs/application-poussee-modele-shap-monitoring.md`](../docs/application-poussee-modele-shap-monitoring.md).

```
application_poussee/
├── apps/
│   ├── inference_app.py        # service FastAPI : /health, /model-info, /predict (journalisation SQLite)
│   └── requirements.txt        # dépendances du service uniquement
├── models/
│   ├── features.py             # variables partagées entre entraînement et service
│   ├── train.py                # champion LightGBM et challenger régression logistique (étape DVC "train")
│   └── artifacts/              # modèles et métriques, suivis par DVC
├── explainability/
│   └── run_shap.py             # analyse SHAP (importance, summary plot, dependence plot)
├── monitoring/
│   ├── replay_traffic.py       # rejoue des transactions contre le service pour alimenter les logs
│   ├── run_drift_on_logs.py    # dérive Evidently : logs de production contre jeu de référence
│   ├── build_dashboard.py      # tableau de bord HTML (métriques, dérive par variable, alerte)
│   └── run_monitoring.py       # estimation de performance CBPE et comparaison avec la performance réelle
├── tests/                      # tests pytest : variables, service, métriques, tableau de bord
├── reports/
│   ├── shap/                   # graphiques et données SHAP
│   └── monitoring/             # rapports de dérive et de performance
└── Dockerfile                  # image du service (modèles inclus, sans données brutes)
```

## Commandes courantes

Depuis la racine du dépôt :

Dépendances supplémentaires pour l'explicabilité et l'estimation de performance : `pip install shap nannyml`.

```bash
dvc repro                                                  # entraîner (si données, code ou paramètres ont changé)
python -m pytest application_poussee/tests -q              # tests
docker compose up --build -d                               # service sur http://localhost:8000/docs
python application_poussee/monitoring/replay_traffic.py --n 300
python application_poussee/monitoring/run_drift_on_logs.py
python application_poussee/monitoring/build_dashboard.py
```

Les modèles entraînés (`.joblib`) et les fichiers de prédictions volumineux sont suivis par DVC et non par Git : `dvc pull` les récupère, `dvc repro` les régénère. La base de logs (`logs/predictions.db`) est un résultat d'exécution local, jamais commité.
