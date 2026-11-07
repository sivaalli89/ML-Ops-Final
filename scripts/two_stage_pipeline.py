"""Two-stage (family -> model) classifier with richer style features, scored with the notebook's GroupKFold(5).

Compared against the notebook's best (char+word TF-IDF, prompt-centering, Hungarian assignment = 0.422).

Features (all fitted on the training fold only):
  char  : char_wb 2-5-grams                       (spelling / punctuation habits)
  word  : word 1-2-grams, stopwords kept          (phrasing)
  skel  : "function-word skeleton" 1-4-grams      (NEW) every non-function word -> W, punctuation kept,
          e.g. "The cat sat; it was warm." -> "the W W ; it was W ."  = syntax rhythm, almost no topic
Everything is centred per prompt group (removes topic), then
  stage 1: family classifier (claude / gemini / gpt / glm / kimi / qwen)
  stage 2: one classifier per family, trained only on that family's passages
  P(model) = P(family) * P(model | family), then Hungarian assignment inside each prompt group.
"""
from pathlib import Path
import re, sys, time
import numpy as np, pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
ROOT = Path(__file__).resolve().parent.parent   # project folder

df = pd.read_csv(str(ROOT / "data" / "train_with_groups.csv")).dropna(subset=["text", "model"]).reset_index(drop=True)
X = df["text"]
le = LabelEncoder().fit(df["model"]); Y = le.transform(df["model"]); G = df["group"].to_numpy()
N_CLS = len(le.classes_)
fam_of = np.array([m.split("-")[0] for m in le.classes_])        # claude / gemini / glm / gpt / kimi / qwen3.8
fam_names = sorted(set(fam_of)); F = np.array([fam_names.index(f) for f in fam_of])[Y]
TOK = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")

def skeleton(texts, keep):
    out = []
    for t in texts:
        out.append(" ".join(w.lower() if w.lower() in keep else ("W" if w[0].isalpha() or w[0] == "'" else w) for w in TOK.findall(t)))
    return out

def featurize(tr_text, te_text, with_skel):
    blocks = [
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=20_000, dtype=np.float32),
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=15_000, dtype=np.float32),
    ]
    A = [b.fit_transform(tr_text).toarray() for b in blocks]
    B = [b.transform(te_text).toarray() for b in blocks]
    if with_skel:
        keep = {w for w, _ in Counter(w.lower() for t in tr_text for w in re.findall(r"[A-Za-z']+", t)).most_common(150)}
        sv = TfidfVectorizer(ngram_range=(1, 4), token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=3, max_features=20_000, dtype=np.float32)
        A.append(sv.fit_transform(skeleton(tr_text, keep)).toarray())
        B.append(sv.transform(skeleton(te_text, keep)).toarray())
    return np.hstack(A), np.hstack(B)

def center(M, groups):
    out = M.copy()
    for g in np.unique(groups):
        i = groups == g
        out[i] -= M[i].mean(axis=0)
    return out

def assign(P, groups):
    pred = P.argmax(1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        cost = np.tile(-np.log(P[idx] + 1e-9), int(np.ceil(len(idx) / N_CLS)))
        r, c = linear_sum_assignment(cost)
        pred[idx[r]] = c % N_CLS
    return pred

def lr(C=5):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=600)

def flat_proba(a, ytr, b, C=5):
    m = lr(C).fit(a, ytr); P = np.zeros((len(b), N_CLS)); P[:, m.classes_] = m.predict_proba(b); return P

def two_stage_proba(a, ytr, ftr, b, C=5, C2=5):
    s1 = lr(C).fit(a, ftr); Pf = s1.predict_proba(b)                 # (n, n_families)
    P = np.zeros((len(b), N_CLS))
    for fi, f in enumerate(fam_names):
        members = np.where(fam_of == f)[0]
        col = list(s1.classes_).index(fi)
        if len(members) == 1:
            P[:, members[0]] = Pf[:, col]; continue
        k = np.isin(ytr, members)
        s2 = lr(C2).fit(a[k], ytr[k])
        P[:, s2.classes_] = Pf[:, [col]] * s2.predict_proba(b)
    return P

def run_fold(tr, te):
    res = {}
    for feat_name, with_skel in [("old feats", False), ("new feats (+skeleton)", True)]:
        A, B = featurize(X.iloc[tr], X.iloc[te], with_skel)
        Ac, Bc = center(A, G[tr]), center(B, G[te])
        for stage, fn in [("flat", lambda: flat_proba(Ac, Y[tr], Bc)),
                          ("two-stage", lambda: two_stage_proba(Ac, Y[tr], F[tr], Bc))]:
            P = fn()
            res[(feat_name, stage, "centred")] = balanced_accuracy_score(Y[te], P.argmax(1))
            res[(feat_name, stage, "centred + assignment")] = balanced_accuracy_score(Y[te], assign(P, G[te]))
            if stage == "two-stage":   # family-level accuracy: how well do we get the company right
                fam_pred = F[np.argmax(P, 1)] if False else np.array([fam_names.index(fam_of[j]) for j in P.argmax(1)])
                res[(feat_name, stage, "family acc (argmax)")] = float((fam_pred == F[te]).mean())
        print(f"  fold done ({feat_name})", flush=True)
    return res

if __name__ == "__main__":
    t0 = time.time()
    folds = list(GroupKFold(5).split(X, Y, G))
    out = Parallel(n_jobs=5)(delayed(run_fold)(tr, te) for tr, te in folds)
    tab = pd.DataFrame(out).T
    tab.index = pd.MultiIndex.from_tuples(tab.index, names=["features", "classifier", "post-processing"])
    summary = pd.DataFrame({"mean": tab.mean(1), "std": tab.std(1)}).round(3)
    print(summary.to_string())
    summary.to_csv(ROOT / "results" / "two_stage_results.csv")
    print(f"\ntotal {time.time()-t0:.0f}s")
