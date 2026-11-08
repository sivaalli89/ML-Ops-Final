"""The notebook must stay valid and keep using names that exist in the package.

Running the whole notebook takes minutes and needs the real data, so this only checks that every code cell parses and
that everything imported from name_that_llm exists.
"""
import ast
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("optuna")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "name_that_llm.ipynb"


def _code_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def test_every_code_cell_parses():
    for source in _code_cells():
        ast.parse(source)


def test_everything_imported_from_the_package_exists():
    for source in _code_cells():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("name_that_llm"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"


def test_notebook_has_one_section_per_airflow_task():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    notebook_text = "\n".join("".join(c["source"]) for c in cells)
    dag_text = (NOTEBOOK.parents[1] / "dags" / "name_that_llm_dag.py").read_text(encoding="utf-8")
    for task in ["check_data", "find_prompt_groups", "tune_with_optuna", "train_model", "evaluate_model", "register_model"]:
        assert f'task_id="{task}"' in dag_text, f"the DAG has no {task} task"
        assert f"`{task}`" in notebook_text, f"the notebook does not mention the {task} task"
