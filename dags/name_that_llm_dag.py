"""Retrain Name That LLM: group the prompts, tune with Optuna, train, and check the score.

Trigger it from the Airflow page, or change schedule=None to "@weekly" to retrain every week.
Every task runs inside the Airflow container, where the project is installed and ./data and
./artifacts are mounted. The API reads the finished model from ./artifacts.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="name_that_llm_training",
    description="group prompts -> tune (Optuna) -> train -> evaluate",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={"trials": 15},   # how many Optuna trials to run; change it when you trigger the DAG
    tags=["name-that-llm"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command=(
            'test -f "$NAME_THAT_LLM_HOME/data/train.csv" '
            '|| { echo "train.csv is missing: put it in ./data on your machine"; exit 1; }'
        ),
    )

    find_prompt_groups = BashOperator(
        task_id="find_prompt_groups",
        bash_command="python -m name_that_llm.grouping",
    )

    tune_with_optuna = BashOperator(
        task_id="tune_with_optuna",
        bash_command="python -m name_that_llm.tune --trials {{ params.trials }}",
        execution_timeout=timedelta(hours=2),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command='python -m name_that_llm.train --fit --params "$NAME_THAT_LLM_HOME/artifacts/best_params.json"',
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            'python -m name_that_llm.train --cv --params "$NAME_THAT_LLM_HOME/artifacts/best_params.json" '
            '--metrics-out "$NAME_THAT_LLM_HOME/artifacts/metrics.json"'
        ),
    )

    check_data >> find_prompt_groups >> tune_with_optuna >> train_model >> evaluate_model
