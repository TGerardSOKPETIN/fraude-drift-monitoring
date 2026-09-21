# Données

Ce dossier accueille les deux fichiers du dataset Poundbank, non versionnés dans le repo (voir `.gitignore`) :

- `reference.csv` — données de référence (baseline)
- `analysis.csv` — données d'analyse (production)

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
| `predicted_fraud_proba` | Probabilité de fraude estimée par le modèle |
| `predicted_fraud` | Étiquette prédite (seuillage de `predicted_fraud_proba`) |

Place les deux fichiers ici avant de lancer les scripts dans `src/`.
