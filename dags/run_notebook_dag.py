"""Run the Name That LLM notebook with Airflow, on demand or on a schedule.

This file has no model code. It only runs the notebook, so the notebook stays the single place where the work is done.

Each run saves its own executed copy of the notebook in ./runs (the original is never changed) and
writes the trained model and its info to ./models.

To run it on a schedule, change schedule=None to "@weekly" (or "@daily"). To run it now: click the play button
in the Airflow page, or  docker compose exec airflow airflow dags trigger run_notebook
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

HOME = "/opt/airflow"

with DAG(
    dag_id="run_notebook",
    description="run notebooks/name_that_llm.ipynb and save the executed copy and the trained model",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["name-that-llm"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command=f'test -f {HOME}/data/train.csv || {{ echo "train.csv is missing: put it in ./data on your computer"; exit 1; }}',
    )

    run_notebook = BashOperator(
        task_id="run_notebook",
        bash_command=(
            f"cd {HOME}/notebooks && "
            "jupyter nbconvert --to notebook --execute name_that_llm.ipynb "
            "--output-dir ../runs --output 'run_{{ ts_nodash }}' --ExecutePreprocessor.timeout=3600"
        ),
        execution_timeout=None,
    )

    show_model_info = BashOperator(
        task_id="show_model_info",
        bash_command=f"cat {HOME}/models/model_info.json",
    )

    check_data >> run_notebook >> show_model_info
