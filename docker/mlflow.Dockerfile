# The MLflow server: experiment tracking and the model registry.
# Runs and registered models are kept in a SQLite file, and the model files in a folder, both in the "mlflow-data" volume.
# Clients (Airflow, the API) talk to it over HTTP, so they don't need access to those files.
# Open http://localhost:5050 (the port docker-compose publishes) to see the experiments and the registered models.
FROM python:3.12-slim

# SQLAlchemy 2.1 is too new for this MLflow version
RUN pip install --no-cache-dir mlflow==2.22.0 "sqlalchemy<2.1"

ARG UID=1000
RUN useradd --create-home --uid ${UID} mlflow && mkdir /mlflow && chown mlflow /mlflow
USER mlflow
WORKDIR /mlflow

EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts"]
