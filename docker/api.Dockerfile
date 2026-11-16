# The prediction service (service/api.py, with service/monitoring.py). The model is not baked in: the service asks the MLflow registry
# for the "champion" version and loads a new one by itself when it changes, without a rebuild or restart.
FROM python:3.12-slim

ARG UID=1000
RUN useradd --create-home --uid ${UID} api

ENV EVIDENTLY_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements.txt .
# the notebook libraries, without JupyterLab and Optuna, plus the web server, the MLflow client and Evidently (drift reports)
RUN grep -vE '^(#|$)|^jupyterlab|^optuna' requirements.txt > api-requirements.txt \
    && pip install --no-cache-dir -r api-requirements.txt fastapi==0.115.6 uvicorn==0.34.0 mlflow-skinny==2.22.0 evidently==0.7.23

# Optional: also use the DistilBERT part of the ensemble (adds about 1 GB, PyTorch for CPU, and the language model
# is downloaded the first time the service starts). Without it the service answers with the style model only.
ARG INSTALL_EMBEDDINGS=false
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir "sentence-transformers>=3"; \
    fi

COPY service ./service
RUN mkdir monitoring && chown api monitoring     # the request log; docker-compose mounts ./monitoring over it
USER api
EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
