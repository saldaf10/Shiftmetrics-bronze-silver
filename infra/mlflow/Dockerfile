FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow + drivers de base de datos + GCS support
RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.10 \
    google-cloud-storage==2.19.0 \
    gunicorn==23.0.0

# Puerto estándar de Cloud Run
ENV PORT=8080

EXPOSE 8080

# Startup: MLflow server con backend PostgreSQL y artifacts en GCS
# Las variables BACKEND_STORE_URI y ARTIFACT_ROOT se inyectan desde Cloud Run
CMD ["sh", "-c", \
     "mlflow server \
       --host 0.0.0.0 \
       --port $PORT \
       --backend-store-uri $BACKEND_STORE_URI \
       --default-artifact-root $ARTIFACT_ROOT \
       --serve-artifacts \
       --workers 2"]
