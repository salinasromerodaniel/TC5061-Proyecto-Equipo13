"""Modelo de persistencia P1 empaquetado como MLflow Models From Code."""

import mlflow
import numpy as np


class PreviousWeekModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return np.asarray(model_input["lag_1"], dtype=float)


mlflow.models.set_model(PreviousWeekModel())
