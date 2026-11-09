FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MIBICI_MODEL_DIR=/app/models/serving/p1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mibici/ ./mibici/
# El host descarga una versión del Registry antes de construir la imagen.
COPY models/serving/p1/ ./models/serving/p1/
RUN test -f models/serving/p1/registry.json && test -f models/serving/p1/model/MLmodel

# El proceso de inferencia solo necesita leer el código y el modelo.
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000"]
