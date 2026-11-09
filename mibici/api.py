"""Sirve el pronóstico P1 con un paquete de modelo descargado del Registry."""

from contextlib import asynccontextmanager
import logging
from math import isclose
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici import config
from mibici.serving import load_model_bundle

logger = logging.getLogger(__name__)

MAX_INSTANCES = 1000
WEEK_PHASE_TOLERANCE = 1e-3


class P1Features(BaseModel):
    """Variables ya calculadas para la semana objetivo, en el orden del entrenamiento."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    lag_1: float = Field(ge=0)
    mean_previous_4: float = Field(ge=0)
    # Una estación puede no tener la semana equivalente del año anterior.
    lag_52: float | None = Field(default=None, ge=0)
    week_sin: float = Field(ge=-1.0, le=1.0)
    week_cos: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_week_phase(self) -> "P1Features":
        # El par debe estar en el círculo unitario, con margen para el redondeo.
        norm_squared = self.week_sin**2 + self.week_cos**2
        if not isclose(norm_squared, 1, rel_tol=0, abs_tol=WEEK_PHASE_TOLERANCE):
            raise ValueError("week_sin y week_cos deben representar la misma fase semanal.")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[P1Features] = Field(min_length=1, max_length=MAX_INSTANCES)


class ModelIdentity(BaseModel):
    name: str
    version: str
    run_id: str
    git_commit: str
    data_version: str


class PredictionResponse(BaseModel):
    predictions: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        min_length=1,
        max_length=MAX_INSTANCES,
    )
    model: ModelIdentity


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    """Configura la API y permite que las pruebas indiquen su propio paquete."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        package_path = (
            model_dir if model_dir is not None
            else os.getenv("MIBICI_MODEL_DIR", str(config.API_MODEL_DIR))
        )
        # Verifica y carga el paquete al arrancar. Las peticiones comparten esta instancia.
        app.state.predictor = load_model_bundle(Path(package_path))
        yield
        del app.state.predictor

    application = FastAPI(
        title="MiBici P1 — Demanda semanal",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
        # Conserva el diagnóstico del 422 aunque el valor recibido sea NaN o infinito.
        detail = [
            {field: item[field] for field in ("loc", "msg", "type")}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.get("/health")
    def health():
        predictor = application.state.predictor
        return {"status": "ok", "model": predictor.identity}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest):
        try:
            predictor = application.state.predictor
            rows = [row.model_dump() for row in payload.instances]
            predictions = predictor.predict(rows)
        except Exception as error:
            logger.exception("Falló la inferencia P1")
            raise HTTPException(status_code=500, detail="No se pudo generar la predicción.") from error
        return PredictionResponse(predictions=predictions, model=predictor.identity)

    return application


app = create_app()
