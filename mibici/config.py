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
DEFAULT_MANIFEST = PROJ_ROOT / "references" / "data-manifest-2026-09.json"
RANDOM_SEED = 13
CHUNK_SIZE = 100_000
TIMEOUT_SECONDS = 60
HTTP_RETRIES = 3
TRANSFORM_VERSION = "1"
MANIFEST_VERSION = 1

# P1: experimento retrospectivo de una semana para Zapopan.
P1_START_MONTH = "2024-10"
P1_END_MONTH = "2026-09"
P1_DATA_MANIFEST = PROJ_ROOT / "references" / "data-manifest-2024-10_2026-09.json"
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
