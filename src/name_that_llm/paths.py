"""Where the data and saved models live.

By default that is the project folder. Inside Docker the package is installed elsewhere, so set
NAME_THAT_LLM_HOME to the folder that contains data/ and artifacts/.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NAME_THAT_LLM_HOME") or Path(__file__).resolve().parents[2])
