"""Small made-up data shared by the tests (no real competition data needed)."""
import numpy as np
import pandas as pd
import pytest

N_MODELS = 4
N_PROMPTS = 20


def _word(rng, length=7):
    return "".join(rng.choice(list("bcdfghklmnprstvz"), length))


@pytest.fixture(scope="session")
def toy_data():
    """20 prompts x 4 models. Each prompt has its own rare topic words; each model has its own habit."""
    rng = np.random.default_rng(0)
    habits = ["moreover , it is clear that", "honestly ; i think that", "in short - the point is", "well , one could say"]
    rows = []
    for prompt in range(N_PROMPTS):
        topic = " ".join(_word(rng) for _ in range(4))
        for model in range(N_MODELS):
            filler = " ".join(_word(rng, 5) for _ in range(6))
            rows.append({
                "prompt_id": prompt,
                "model": f"model-{model}",
                "text": f"{habits[model]} {topic} and also {filler} . {habits[model]} {topic} .",
            })
    return pd.DataFrame(rows)
