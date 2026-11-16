"""Watching the live service for drift, with Evidently.

The service has no labels in production, so it watches what it can see: the passages it is sent and how confident the
model is about them. For every passage it records eight numbers (FEATURES below). The same numbers are computed for half
of the held-out prompts when a model is loaded (see split_holdout); that is the baseline. Evidently's data drift report
(evidently 0.7.23, DataDriftPreset) compares the recent passages with the baseline, one column at a time, with the
normalised Wasserstein distance: how far the recent values have moved, in units of the baseline's own spread. A column
has drifted when that distance is above DRIFT_THRESHOLD.

Why a distance and not a p-value test: the passages of one prompt resemble each other, so a few prompts do not behave like
that many independent rows, and p-value tests (Evidently's default is K-S at 0.05) raised false alarms on clean data in
my first version. A distance measures how big the move is whatever the sample size. DRIFT_THRESHOLD and MIN_SAMPLES were
chosen from clean and corrupted samples of the held-out prompts (see the README), so they fit this data and model.

    status "OK"      nothing has drifted
    status "ALERT"   at least one column drifted, or several requests were rejected as malformed
    status "WAITING" fewer than MIN_SAMPLES recent passages, too few to tell

Used by service/api.py (live) and compared in pipeline/simulate_drift.py (the drift test).
"""
import html
import json
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURES = {
    "n_chars": "passage length (characters)",
    "n_words": "passage length (words)",
    "avg_word_len": "average word length",
    "punct_share": "share of punctuation and symbols",
    "unknown_word_share": "share of words not seen in training",
    "known_pair_share": "share of neighbouring word pairs seen in training (word order)",
    "group_similarity": "similarity to the other passages in the request (same prompt = high)",
    "confidence": "the model's top probability for the passage",
}

DRIFT_THRESHOLD = 1.0   # normalised Wasserstein distance above which a column has drifted
MIN_SAMPLES = 80        # fewer recent passages than this (about 5 prompts): no verdict
MAX_INVALID = 3         # this many rejected requests among the recent ones is an alert

WORD = re.compile(r"\b\w\w+\b")      # the same rule the word TF-IDF in the model uses


def split_holdout(holdout):
    """The held-out passages split in two by prompt: the baseline half, and a clean half kept for testing against it.
    If the baseline and the test were the same passages, the test could not fail."""
    even = holdout["group"] % 2 == 0
    return holdout[even], holdout[~even]


def text_stats(texts, vocabulary):
    rows = []
    for text in texts:
        words = WORD.findall(text.lower())
        pairs = [f"{a} {b}" for a, b in zip(words, words[1:])]
        rows.append({
            "n_chars": len(text),
            "n_words": len(words),
            "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
            "punct_share": sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(len(text), 1),
            "unknown_word_share": float(np.mean([w not in vocabulary for w in words])) if words else 1.0,
            "known_pair_share": float(np.mean([pair in vocabulary for pair in pairs])) if pairs else 0.0,
        })
    return pd.DataFrame(rows)


def request_stats(saved, texts, probs, vocabulary):
    """The eight numbers for each passage of one request. probs are the model's probabilities, before the assignment step."""
    stats = text_stats(texts, vocabulary)
    x = saved["features"]["vectorizers"]["words"].transform(texts)
    similarity = (x @ x.T).toarray()
    stats["group_similarity"] = (similarity.sum(axis=1) - np.diag(similarity)) / max(len(texts) - 1, 1)
    stats["confidence"] = probs.max(axis=1)
    return stats[list(FEATURES)]


def evidently_snapshot(baseline, recent):
    """Evidently's data drift report: the recent passages (current) against the baseline (reference)."""
    report = Report([DataDriftPreset(method="wasserstein", threshold=DRIFT_THRESHOLD)])
    return report.run(recent[list(FEATURES)], baseline[list(FEATURES)])


def drift_report(baseline, recent):
    """Which columns have drifted, from Evidently's report."""
    if len(recent) < MIN_SAMPLES:
        return {"status": "WAITING", "samples": len(recent), "features": [], "drifted": []}
    distances = {metric["config"]["column"]: metric["value"]
                 for metric in evidently_snapshot(baseline, recent).dict()["metrics"]
                 if metric["config"]["type"].endswith("ValueDrift")}
    rows = [{
        "feature": name,
        "description": description,
        "baseline_mean": round(float(baseline[name].mean()), 4),
        "recent_mean": round(float(recent[name].mean()), 4),
        "distance": round(float(distances[name]), 3),
        "drifted": bool(distances[name] > DRIFT_THRESHOLD),
    } for name, description in FEATURES.items()]
    drifted = [row["feature"] for row in rows if row["drifted"]]
    return {"status": "ALERT" if drifted else "OK", "samples": len(recent), "features": rows, "drifted": drifted}


class Monitor:
    """Keeps the recent requests in memory, and in a file so they survive a restart."""

    def __init__(self, folder=None, keep=5000):
        self.passages = deque(maxlen=keep)     # one dict per passage: time, model version, the eight numbers
        self.requests = deque(maxlen=keep)     # one bool per request: was it accepted?
        self.lock = threading.Lock()
        self.file = Path(folder) / "requests.jsonl" if folder else None
        if self.file is not None:
            try:
                self.file.parent.mkdir(parents=True, exist_ok=True)
                self._load()
            except OSError as error:
                print("monitoring log not available:", error, flush=True)
                self.file = None

    def _load(self):
        if not self.file.exists():
            return
        with self.file.open() as lines:
            for line in deque(lines, maxlen=20000):      # the file only grows, so read just the end
                entry = json.loads(line)
                (self.passages if entry["kind"] == "passage" else self.requests).append(
                    entry if entry["kind"] == "passage" else entry["ok"])

    def _write(self, entries):
        if self.file is None:
            return
        try:
            with self.file.open("a") as out:
                out.writelines(json.dumps(entry) + "\n" for entry in entries)
        except OSError:
            pass                                          # logging must never break a prediction

    def record(self, version, stats):
        now = time.time()
        entries = [{"kind": "passage", "time": now, "version": version, **row} for row in stats.to_dict("records")]
        entries.append({"kind": "request", "time": now, "ok": True})
        with self.lock:
            self.passages.extend(entry for entry in entries if entry["kind"] == "passage")
            self.requests.append(True)
            self._write(entries)

    def record_invalid(self):
        with self.lock:
            self.requests.append(False)
            self._write([{"kind": "request", "time": time.time(), "ok": False}])

    def recent(self, version, last=500):
        """The last `last` passages that this model version answered."""
        with self.lock:
            rows = [entry for entry in self.passages if entry["version"] == version][-last:]
        return pd.DataFrame(rows, columns=["time", "version", *FEATURES])

    def report(self, baseline, version, last=500, last_requests=100):
        recent = self.recent(version, last)
        with self.lock:
            invalid = sum(1 for ok in list(self.requests)[-last_requests:] if not ok)
        drift = drift_report(baseline, recent) if baseline is not None else {"status": "NO BASELINE", "samples": len(recent), "features": [], "drifted": []}
        status = drift["status"]
        if invalid >= MAX_INVALID:
            status = "ALERT"
        return {"status": status, "model_version": version, "invalid_requests": invalid, "requests_checked": last_requests,
                "drift": drift, "thresholds": {"distance": DRIFT_THRESHOLD, "min_samples": MIN_SAMPLES, "max_invalid": MAX_INVALID}}

    def evidently_html(self, baseline, version, last=500):
        """Evidently's own full report page (distributions per column), or a short message if there is nothing to show yet."""
        recent = self.recent(version, last)
        if baseline is None or len(recent) < MIN_SAMPLES:
            return f"<p>Not enough data yet: {len(recent)} recent passages, need {MIN_SAMPLES}.</p>"
        return evidently_snapshot(baseline, recent).get_html_str(as_iframe=False)


def render_dashboard(report, model):
    """A small self-contained HTML page for a person to look at."""
    colour = {"OK": "ok", "ALERT": "alert"}.get(report["status"], "wait")
    drift = report["drift"]
    rows = "".join(
        f"<tr class=\"{'alert' if row['drifted'] else ''}\"><td>{html.escape(row['description'])}</td>"
        f"<td>{row['baseline_mean']}</td><td>{row['recent_mean']}</td><td>{row['distance']}</td>"
        f"<td>{'DRIFT' if row['drifted'] else 'ok'}</td></tr>"
        for row in drift["features"]) or f"<tr><td colspan=5>Not enough data yet: {drift['samples']} recent passages, need {MIN_SAMPLES}.</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5"><title>Name That LLM monitor</title><style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --ok:#1b7f3b; --alert:#b3261e; --wait:#8a6d00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181d; --fg:#e8e8e8; --muted:#9aa0a6; --line:#33363d; --ok:#5dd28a; --alert:#ff7b72; --wait:#e3b341; }} }}
body {{ font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); margin:0 auto; max-width:900px; padding:16px; }}
.badge {{ display:inline-block; padding:4px 14px; border-radius:6px; font-weight:700; color:#fff; }}
.badge.ok {{ background:var(--ok); }} .badge.alert {{ background:var(--alert); }} .badge.wait {{ background:var(--wait); }}
table {{ border-collapse:collapse; width:100%; }} td,th {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }}
tr.alert td {{ color:var(--alert); font-weight:600; }} .muted {{ color:var(--muted); }} .scroll {{ overflow-x:auto; }}
</style></head><body>
<h1>Name That LLM: monitor</h1>
<p><span class="badge {colour}">{html.escape(report['status'])}</span></p>
<p class="muted">Model {html.escape(str(model['version']))} ({html.escape(model['source'])}). Held-out accuracy {model['baseline_accuracy']}.
Recent passages checked: {drift['samples']}. Rejected requests among the last {report['requests_checked']}: {report['invalid_requests']}
(alert from {report['thresholds']['max_invalid']}). Refreshes every 5 seconds.</p>
<div class="scroll"><table><tr><th>Feature</th><th>Baseline mean</th><th>Recent mean</th><th>Distance</th><th></th></tr>{rows}</table></div>
<p class="muted">Distance is Evidently's normalised Wasserstein distance: how far the recent values moved, in units of the baseline's
spread. A feature drifts above {report['thresholds']['distance']}. The baseline is half of the held-out prompts, sent as whole prompts;
requests with only a few passages look different on purpose. No verdict below {report['thresholds']['min_samples']} recent passages.
<a href="/dashboard/evidently">Evidently's full report</a>.</p>
</body></html>"""
