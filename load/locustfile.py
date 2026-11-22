"""Carga cerrada de POST /predict con el lote público de dos estaciones de P1."""
import json
import math
import os
from pathlib import Path

from locust import HttpUser, constant, task

PAYLOAD = json.loads((Path(__file__).resolve().parents[1] / "examples/p1-predict.json")
                     .read_text(encoding="utf-8"))
EXPECTED_MODEL = json.loads(os.environ["MIBICI_LOAD_MODEL"])


class P1User(HttpUser):
    # Cada usuario espera la respuesta antes de volver a consultar. Sin pausa
    # deliberada, el experimento mide concurrencia con presión continua.
    wait_time = constant(0)

    @task
    def predict(self):
        with self.client.post("/predict", json=PAYLOAD, name="POST /predict (2 instances)",
                              timeout=10, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
                values = body["predictions"]
                valid = (body["model"] == EXPECTED_MODEL and isinstance(values, list)
                         and len(values) == len(PAYLOAD["instances"])
                         and all(type(value) in (int, float) and math.isfinite(value)
                                 and value >= 0 for value in values))
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                response.failure("Respuesta o identidad del modelo incorrecta")
