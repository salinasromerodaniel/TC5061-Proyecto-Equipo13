"""Configuración compartida del proyecto; septiembre de 2026 es una prueba de ingesta."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SOURCE_PAGE = "https://www.mibici.net/es/datos-abiertos/"
DEFAULT_START_MONTH = "2026-09"
DEFAULT_END_MONTH = "2026-09"
STATIONS_FILENAME = "nomenclatura_2026_09.csv"
DEFAULT_MANIFEST = PROJ_ROOT / "references" / "data-manifest-2026-09-v2.json"
RANDOM_SEED = 13
CHUNK_SIZE = 100_000
TIMEOUT_SECONDS = 60
HTTP_RETRIES = 3
TRANSFORM_VERSION = "2"
MANIFEST_VERSION = 1

# P1: experimento retrospectivo de una semana para Zapopan.
P1_START_MONTH = "2024-10"
P1_END_MONTH = "2026-09"
P1_DATA_MANIFEST = PROJ_ROOT / "references" / "data-manifest-2024-10_2026-09-v2.json"
P1_BOUNDARY_MANIFEST = PROJ_ROOT / "references" / "zapopan-boundary-manifest.json"
P1_BOUNDARY_URL = "https://gaia.inegi.org.mx/wscatgeo/v2/geo/mgem/14120"
P1_MUNICIPALITY_CODE = "14120"
P1_MUNICIPALITY_NAME = "Zapopan"
P1_HISTORY_WEEKS = 4
P1_VALIDATION_WEEKS = 8
P1_TEST_WEEKS = 8
P1_EXPERIMENT = "mibici-equipo13"
P1_REPORT_DIR = REPORTS_DIR / "p1"
P1_TRACKING_DB = PROJ_ROOT / "mlflow_tracking.db"
P1_ARTIFACT_DIR = PROJ_ROOT / "mlartifacts"

P1_COHORT_START_MONTH = "2025-10"
P1_SEASONAL_WEEKS = 52

# P2: clasificación de riesgo indirecto
P2_EXPERIMENT = "mibici-equipo13"
P2_REPORT_DIR = REPORTS_DIR / "p2"
P2_TRACKING_DB = P1_TRACKING_DB
P2_ARTIFACT_DIR = P1_ARTIFACT_DIR
P2_DATA_MANIFEST = PROJ_ROOT / "references" / "data-manifest-2024-10_2026-09-v2.json"
P2_VALIDATION_WEEKS = 8
P2_TEST_WEEKS = 8

# Entrega 2: valores resueltos en Python; MLproject usa el marcador "config".
PROJECT_EXPERIMENT = P1_EXPERIMENT
PROJECT_TRACKING_HOST = "127.0.0.1"
PROJECT_TRACKING_PORT = 5013
PROJECT_TRACKING_URI = f"http://{PROJECT_TRACKING_HOST}:{PROJECT_TRACKING_PORT}"
PROJECT_DATA_MANIFEST = P1_DATA_MANIFEST
PROJECT_DEFAULT_QUESTION = "p1"
PROJECT_DEFAULT_MODELS = {"p1": "hgb-search", "p2": "baseline", "p3": "kmeans"}
PROJECT_DEFAULT_INGEST_MODE = "reproduce"
PROJECT_REPORT_DIR = REPORTS_DIR / "delivery2"
P1_CANDIDATE_FEATURES = ["lag_1", "mean_previous_4", "lag_52", "week_sin", "week_cos"]
P1_HGB_DEFAULTS = {"loss": "poisson", "learning_rate": 0.05, "max_iter": 150,
                   "max_leaf_nodes": 15, "min_samples_leaf": 20,
                   "l2_regularization": 1.0, "early_stopping": False,
                   "random_state": RANDOM_SEED}
P1_HGB_SEARCH = [{"max_leaf_nodes": 7}, {"max_leaf_nodes": 15}, {"max_leaf_nodes": 31}]
P2_MODEL_DEFAULTS = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5,
                     "class_weight": "balanced", "random_state": RANDOM_SEED, "n_jobs": -1}
P3_MODEL_DEFAULTS = {"n_clusters": 4, "random_state": RANDOM_SEED, "n_init": 10}

# P2: candidatas adicionales al Dummy de la primera entrega.
P2_LOGISTIC_DEFAULTS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced",
                        "random_state": RANDOM_SEED}
P2_SEARCH_CONFIGS = [
    {"name": "rf_200_depth12", "family": "random_forest", "parameters": {}},
    {"name": "rf_300_depth6", "family": "random_forest",
     "parameters": {"n_estimators": 300, "max_depth": 6}},
    {"name": "logistic", "family": "logistic", "parameters": {}},
]
