"""Checks the Airflow DAG. Skipped when Airflow isn't installed (it only runs inside the Airflow container)."""
from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


def test_dag_loads_without_errors():
    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert bag.import_errors == {}
    assert "name_that_llm_training" in bag.dags


def test_tasks_run_in_the_expected_order():
    dag = DagBag(dag_folder=str(DAGS), include_examples=False).dags["name_that_llm_training"]
    order = ["check_data", "find_prompt_groups", "tune_with_optuna", "train_model", "evaluate_model"]
    assert [t.task_id for t in dag.topological_sort()] == order
