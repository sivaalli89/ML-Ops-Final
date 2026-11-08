# Airflow plus the libraries the notebook needs, so Airflow can run the notebook.
# The notebook, data, models and the DAG are mounted from your computer by docker-compose.
FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^(#|$)' /tmp/requirements.txt > /tmp/notebook-requirements.txt \
    && pip install --no-cache-dir -r /tmp/notebook-requirements.txt
