"""Train the model with Airflow, on demand or on a schedule.

This file has no model code. It runs the notebook (which tunes with Optuna, trains and saves the model), then registers
the result in MLflow. The notebook stays the single place where the work is done.

  check_data -> run_notebook -> register_model

* run_notebook saves its own executed copy of the notebook in ./runs (the original is never changed) and writes the
  model, its info and the held-out passages to ./models.
* register_model (pipeline/register_model.py) logs the run to MLflow, registers a new model version and moves the
  "champion" alias to it if its held-out score is at least as good as the current champion's. The API loads the champion.

To run it on a schedule, change schedule=None to "@weekly" (or "@daily"). To run it now: click the play button
in the Airflow page, or  docker compose exec airflow airflow dags trigger train_pipeline
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

HOME = "/opt/airflow"

with DAG(
    dag_id="train_pipeline",
    description="run notebooks/name_that_llm.ipynb, then register the trained model in MLflow",
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
            "--output-dir ../runs --output 'run_{{ ts_nodash }}' --ExecutePreprocessor.timeout=7200"
        ),
        execution_timeout=None,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {HOME}/pipeline && "
            "python register_model.py --models ../models --notebook '../runs/run_{{ ts_nodash }}.ipynb'"
        ),
    )

    check_data >> run_notebook >> register_model
