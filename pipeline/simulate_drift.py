"""Drift test for the running service: a clean baseline check, then deliberately corrupted data.

    docker compose --profile api --profile drift run --rm drift-test

It uses the held-out passages the notebook saved (models/holdout.csv). They are split by prompt into two halves:
the service built its monitoring baseline from the even-numbered prompts, and this script sends the odd-numbered
ones, which the baseline has never seen. Steps:

  1. Baseline validation: send the clean half, one prompt per request. The accuracy should match what the model scored on
     the held-out prompts, and the monitor should say OK.
  2. Drift simulation: corrupt the same passages in several ways, send them, and ask the monitor what it sees:
       truncated        keep only the first fifth of every passage
       random_chars     replace 30% of the letters with random ones
       shuffled_words   shuffle the words inside every passage
       mixed_prompts    put passages from different prompts into the same request (breaks the grouping the model relies on)
       broken_schema    send requests with the wrong field names, wrong types, or too few passages
  3. Print a table, and save it to monitoring/drift_test.json.

The accuracy column is only possible here because this script knows the true answers. A live service doesn't, which is
why the monitor watches the inputs and the model's confidence instead.
"""
import argparse
import json
import random
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def call(api, path, body=None):
    request = urllib.request.Request(api + path, json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or "{}")


# ---- the corruptions: each takes the list of requests (a list of (text, true_llm) lists) and returns new ones ----

def truncated(requests, rng):
    return [[(text[: max(len(text) // 5, 20)], llm) for text, llm in request] for request in requests]


def random_chars(requests, rng):
    def corrupt(text):
        return "".join(rng.choice(string.ascii_lowercase) if ch.isalpha() and rng.random() < 0.3 else ch for ch in text)
    return [[(corrupt(text), llm) for text, llm in request] for request in requests]


def shuffled_words(requests, rng):
    def corrupt(text):
        words = text.split(" ")
        rng.shuffle(words)
        return " ".join(words)
    return [[(corrupt(text), llm) for text, llm in request] for request in requests]


def mixed_prompts(requests, rng):
    everything = [item for request in requests for item in request]
    rng.shuffle(everything)
    size = max(len(request) for request in requests)
    chunks = [everything[i:i + size] for i in range(0, len(everything), size)]
    return [chunk for chunk in chunks if len(chunk) >= 2]          # the service needs at least two passages


SCENARIOS = {"truncated": truncated, "random_chars": random_chars, "shuffled_words": shuffled_words, "mixed_prompts": mixed_prompts}


def send(api, requests):
    """Send each request; return the accuracy against the true LLMs."""
    right = total = 0
    for request in requests:
        status, answer = call(api, "/predict", {"passages": [text for text, _ in request]})
        assert status == 200, (status, answer)
        right += sum(guess == llm for guess, (_, llm) in zip(answer["llms"], request))
        total += len(request)
    return right / total, total


def monitor(api, passages, requests):
    status, report = call(api, f"/monitor?last={passages}&last_requests={requests}")
    assert status == 200, (status, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://api:8000")
    parser.add_argument("--holdout", default="/app/models/holdout.csv")
    parser.add_argument("--out", default="/app/monitoring/drift_test.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    holdout = pd.read_csv(args.holdout)
    clean_half = holdout[holdout["group"] % 2 == 1]       # the other half is the service's baseline (monitoring.split_holdout)
    requests = [list(zip(group["text"], group["model"])) for _, group in clean_half.groupby("group") if len(group) >= 2]
    n_passages = sum(len(request) for request in requests)

    _, info = call(args.api, "/model")
    print(f"model {info['version']} ({info['source']}); held-out score when trained: {info['balanced_accuracy']}")
    print(f"sending {len(requests)} clean prompts, {n_passages} passages, one prompt per request\n")

    results = []
    accuracy, _ = send(args.api, requests)
    report = monitor(args.api, n_passages, len(requests))
    results.append({"scenario": "clean", "accuracy": round(accuracy, 3), "status": report["status"], "drifted": report["drift"]["drifted"],
                    "rejected": report["invalid_requests"]})

    for name, corrupt in SCENARIOS.items():
        corrupted = corrupt(requests, rng)
        accuracy, sent = send(args.api, corrupted)
        report = monitor(args.api, sent, len(corrupted))
        results.append({"scenario": name, "accuracy": round(accuracy, 3), "status": report["status"], "drifted": report["drift"]["drifted"],
                        "rejected": report["invalid_requests"]})

    bad_bodies = [{"texts": ["a", "b", "c"]}, {"passages": "one long string"}, {"passages": [1, 2, 3]},
                  {"passages": ["only one"]}, {"passages": []}, {"passages": [{"text": "x"}, {"text": "y"}]}]
    codes = [call(args.api, "/predict", body)[0] for body in bad_bodies]
    report = monitor(args.api, 1, len(bad_bodies))
    results.append({"scenario": "broken_schema", "accuracy": None, "status": report["status"], "drifted": [],
                    "rejected": report["invalid_requests"], "http_codes": codes})

    print(f"{'scenario':<16}{'accuracy':>9}  {'monitor':<8} what the monitor saw")
    for row in results:
        saw = ", ".join(row["drifted"]) if row["drifted"] else (f"{row['rejected']} rejected requests (HTTP {set(row['http_codes'])})" if row["scenario"] == "broken_schema" else "nothing unusual")
        print(f"{row['scenario']:<16}{'' if row['accuracy'] is None else row['accuracy']:>9}  {row['status']:<8} {saw}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": info, "results": results}, indent=2))
    print("\nsaved", args.out)
    sys.exit(0 if results[0]["status"] == "OK" else 1)       # the clean data must not raise an alert


if __name__ == "__main__":
    main()
