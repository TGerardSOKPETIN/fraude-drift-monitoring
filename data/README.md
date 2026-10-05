# Données

Ce dossier contient les deux fichiers du jeu Poundbank. Ils sont suivis par [DVC](https://dvc.org) (fichiers `.dvc` versionnés dans Git, contenu stocké sur le remote DVC) et non directement par Git.

- `reference.csv` : données de référence
- `analysis.csv` : données d'analyse (production)

## Récupérer les données

```bash
pip install "dvc[gdrive]" "cryptography<44"
dvc pull data/reference.csv data/analysis.csv
```

Le remote est un dossier Google Drive : l'accès doit être accordé à chaque contributeur. La CI y accède avec un compte de service en lecture seule.

## Colonnes

| Colonne | Description |
|---|---|
| `timestamp` | Date de la transaction |
| `time_since_login_min` | Temps écoulé depuis la connexion de l'utilisateur à l'application |
| `transaction_amount` | Montant, en livres sterling (£), envoyé vers un autre compte |
| `transaction_type` | Type de transaction : CASH-OUT, PAYMENT, CASH-IN, TRANSFER |
| `is_first_transaction` | 1 si première transaction de l'utilisateur, 0 sinon |
| `user_tenure_months` | Ancienneté en mois depuis l'inscription de l'utilisateur |
| `is_fraud` | Étiquette réelle : 1 si transaction frauduleuse, 0 sinon |
| `predicted_fraud_proba` | Probabilité de fraude estimée par le modèle d'origine |
| `predicted_fraud` | Étiquette prédite (seuillage de `predicted_fraud_proba`) |
