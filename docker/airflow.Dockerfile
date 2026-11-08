# Airflow with our package installed, so the DAG's tasks can run the training code.
# (No fastapi/uvicorn here: those are for the API image only.)
FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(#|fastapi|uvicorn)' /tmp/requirements.txt > /tmp/train-requirements.txt \
    && pip install --no-cache-dir -r /tmp/train-requirements.txt

COPY pyproject.toml /opt/airflow/project/pyproject.toml
COPY src /opt/airflow/project/src
RUN pip install --no-cache-dir --no-deps /opt/airflow/project

# data/ and artifacts/ are mounted here by docker-compose
ENV NAME_THAT_LLM_HOME=/opt/airflow
