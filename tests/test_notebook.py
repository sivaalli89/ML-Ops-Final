"""The explanation notebook has its own simple code. Just make sure it stays valid Python."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "name_that_llm.ipynb"


def test_every_code_cell_parses():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_is_not_wired_to_the_pipeline():
    text = NOTEBOOK.read_text(encoding="utf-8")
    code = "".join("".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "code")
    for name in ("name_that_llm", "mlflow", "optuna", "airflow"):
        assert name not in code, f"the notebook code should not use {name}"
