"""2D map of the passages, one colour per model.

Left/overview: all 18 models in one scatter.  Grid: one panel per model (that model in colour, the rest grey).
Two views are drawn so you can see the point of the notebook:
  TOPIC view  = LSA of content words   -> models are mixed together (topic says nothing about author)
  STYLE view  = char n-grams, centred per prompt group -> models start to separate
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
ROOT = Path(__file__).resolve().parent.parent   # project folder

df = pd.read_csv(str(ROOT / "data" / "train_with_groups.csv")).dropna(subset=["text", "model"]).reset_index(drop=True)
models = sorted(df["model"].unique())
G = df["group"].to_numpy()

# --- topic view (same recipe as notebook section 2) ---
vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", sublinear_tf=True, min_df=2, max_df=0.05)
Zt = normalize(TruncatedSVD(200, random_state=42).fit_transform(vec.fit_transform(df["text"])))
topic2d = PCA(2, random_state=42).fit_transform(Zt)

# --- style view: char n-grams, subtract each prompt group's mean (prompt-centering), then 2D ---
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=20_000, dtype=np.float32)
S = char.fit_transform(df["text"]).toarray()
for g in np.unique(G):
    i = G == g
    S[i] -= S[i].mean(axis=0)
style2d = PCA(2, random_state=42).fit_transform(S)

cmap = plt.get_cmap("tab20")
col = {m: cmap(i) for i, m in enumerate(models)}

for name, xy in [("topic", topic2d), ("style", style2d)]:
    title = {"topic": "TOPIC space (LSA of content words)", "style": "STYLE space (char n-grams, centred per prompt)"}[name]
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(4, 8)
    ax = fig.add_subplot(gs[:, :3])
    for m in models:
        k = (df["model"] == m).to_numpy()
        ax.scatter(*xy[k].T, s=9, alpha=.7, color=col[m], label=m)
    ax.set(title=f"{title}: all models", xticks=[], yticks=[])
    ax.legend(fontsize=7, markerscale=2, loc="best")
    for j, m in enumerate(models):
        a = fig.add_subplot(gs[j // 5, 3 + j % 5])
        k = (df["model"] == m).to_numpy()
        a.scatter(*xy[~k].T, s=3, color="#dddddd")
        a.scatter(*xy[k].T, s=8, color=col[m])
        a.scatter(*xy[k].mean(0), s=120, marker="X", color="black")   # black X = that model's average position
        a.set_title(m, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}: each dot is one passage, black X = the model's centre", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(ROOT / "figures" / f"models_2d_{name}.png"), dpi=110)
    print("saved", str(ROOT / "figures" / f"models_2d_{name}.png"))
