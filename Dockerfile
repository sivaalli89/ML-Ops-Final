# JupyterLab with the libraries the notebook needs.
# The notebook and the data are not baked in: docker-compose mounts them from your computer,
# so edits and the notebook's saved results stay in your project folder.
FROM python:3.12-slim

# run as a normal user instead of root (change UID if your files belong to a different user on Linux)
ARG UID=1000
RUN useradd --create-home --uid ${UID} jupyter

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: compute the DistilBERT numbers inside the container too (adds about 1 GB, PyTorch for CPU).
# Not needed if data/embeddings/distilbert.npy already exists.
ARG INSTALL_EMBEDDINGS=false
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir "sentence-transformers>=3"; \
    fi

USER jupyter
# the notebook reads ../data/train.csv, so it must run from the notebooks folder
WORKDIR /app/notebooks
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
