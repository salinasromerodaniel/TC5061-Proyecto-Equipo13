"""API de inferencia P1; carga una versión del Registry preparada localmente."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mibici import config
from mibici.serving import load_model_bundle

logger = logging.getLogger(__name__)


class P1Features(BaseModel):
    """Variables ya calculadas para la semana objetivo, en el orden del entrenamiento."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lag_1: float
    mean_previous_4: float
    lag_52: float | None = None
    week_sin: float
    week_cos: float


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[P1Features] = Field(min_length=1, max_length=1000)


class ModelIdentity(BaseModel):
    name: str
    version: str
    run_id: str
    git_commit: str
    data_version: str


class PredictionResponse(BaseModel):
    predictions: list[float]
    model: ModelIdentity


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        directory = Path(model_dir) if model_dir is not None else Path(
            os.getenv("MIBICI_MODEL_DIR", str(config.API_MODEL_DIR)))
        # Fallar al arrancar si falta el artefacto; nunca devolver un modelo simulado.
        app.state.predictor = load_model_bundle(directory)
        yield
        del app.state.predictor

    application = FastAPI(title="MiBici P1 — Demanda semanal", version="1.0.0",
                          lifespan=lifespan)

    @application.get("/health")
    def health():
        predictor = application.state.predictor
        return {"status": "ok", "model": predictor.identity}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest):
        try:
            values = application.state.predictor.predict(
                [row.model_dump() for row in payload.instances])
        except Exception as error:
            logger.exception("Falló la inferencia P1")
            raise HTTPException(status_code=500, detail="No se pudo generar la predicción.") from error
        return PredictionResponse(predictions=values,
                                  model=application.state.predictor.identity)

    return application


app = create_app()
