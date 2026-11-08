"""Name That LLM: guess which of 18 LLMs wrote a passage."""
import os
from pathlib import Path

__version__ = "0.1.0"

# Where data/ and artifacts/ live. By default the project folder. Inside Docker the package is installed
# elsewhere, so set NAME_THAT_LLM_HOME to the folder that contains data/ and artifacts/.
ROOT = Path(os.environ.get("NAME_THAT_LLM_HOME") or Path(__file__).resolve().parents[2])
