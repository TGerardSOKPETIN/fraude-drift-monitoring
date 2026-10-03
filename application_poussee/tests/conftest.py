"""Configuration partagee des tests. Redirige la base de logs SQLite vers un
fichier de test, AVANT que inference_app.py ne soit importe par les autres
modules de test (LOG_DB_PATH est lu une seule fois, au chargement du module
-- cf. apps/inference_app.py), afin de ne jamais polluer la vraie base de
logs de production (application_poussee/logs/predictions.db) en lancant la
suite de tests.
"""
import os
from pathlib import Path

TEST_LOG_DB = Path(__file__).resolve().parent / "test_predictions.db"
os.environ.setdefault("LOG_DB_PATH", str(TEST_LOG_DB))


def pytest_sessionfinish(session, exitstatus):
    """Nettoie la base de logs de test a la fin de la suite, pour ne pas
    laisser de fichier genere trainer dans le depot."""
    if TEST_LOG_DB.exists():
        TEST_LOG_DB.unlink()
