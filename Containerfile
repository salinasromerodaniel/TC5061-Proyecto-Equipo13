FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MIBICI_MODEL_DIR=/app/models/serving/p1 \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mibici/ ./mibici/
COPY examples/p1-predict.json ./examples/p1-predict.json
ARG VCS_REF=working-tree
LABEL org.opencontainers.image.revision=$VCS_REF
EXPOSE 8000
USER 10001:10001
CMD ["python", "-m", "uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Compose aprovisiona el modelo en un volumen después del build.
FROM runtime AS services

# Etapa predeterminada: conserva la imagen autónoma de la entrega anterior.
FROM runtime AS bundled
COPY models/serving/p1/ ./models/serving/p1/
RUN test -f models/serving/p1/registry.json && test -f models/serving/p1/model/MLmodel
