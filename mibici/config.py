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
