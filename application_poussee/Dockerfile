FROM python:3.11-slim

WORKDIR /app

# libgomp1 : librairie système requise par LightGBM pour charger son
# binaire natif (absente de l'image "slim" par défaut — sans elle,
# joblib.load() plante au démarrage avec "libgomp.so.1: cannot open
# shared object file")
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances du service uniquement (pas celles de l'analyse de drift,
# inutiles pour servir l'API — voir apps/requirements.txt)
COPY apps/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Code et artefacts nécessaires au service de scoring, copiés explicitement
# (pas de COPY . . : on évite d'embarquer data/, reports/, les notebooks)
COPY apps/ ./apps/
COPY models/features.py ./models/features.py
COPY models/artifacts/champion_lightgbm.joblib ./models/artifacts/champion_lightgbm.joblib
COPY models/artifacts/challenger_logreg.joblib ./models/artifacts/challenger_logreg.joblib

EXPOSE 8000

CMD ["uvicorn", "apps.inference_app:app", "--host", "0.0.0.0", "--port", "8000"]
