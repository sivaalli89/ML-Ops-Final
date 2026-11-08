# Slim API image: TF-IDF style model only (no PyTorch). The model file is mounted in, not baked in.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY api ./api

ENV MODEL_PATH=/models/model.joblib \
    USE_EMBEDDINGS=false

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
