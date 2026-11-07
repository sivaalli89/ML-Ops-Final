"""Non-PCA 2D maps of the passages, one colour per model.

t-SNE : unsupervised, keeps close neighbours close. Input = prompt-centred char n-gram style features.
LDA   : supervised, axes chosen to separate the models. Fitted on all data, so it is OPTIMISTIC (in-sample).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
ROOT = Path(__file__).resolve().parent.parent   # project folder

df = pd.read_csv(str(ROOT / "data" / "train_with_groups.csv")).dropna(subset=["text", "model"]).reset_index(drop=True)
models = sorted(df["model"].unique())
G = df["group"].to_numpy()

char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=20_000, dtype=np.float32)
S = char.fit_transform(df["text"]).toarray()
for g in np.unique(G):          # prompt-centering: remove the topic
    i = G == g
    S[i] -= S[i].mean(axis=0)
R = TruncatedSVD(100, random_state=42).fit_transform(S)   # compress before t-SNE / LDA

views = {
    "tsne": ("t-SNE (unsupervised, style space)", TSNE(2, perplexity=30, init="pca", random_state=42).fit_transform(R)),
    "lda": ("LDA (supervised, in-sample = optimistic)", LinearDiscriminantAnalysis(n_components=2).fit(R, df["model"]).transform(R)),
}

cmap = plt.get_cmap("tab20")
col = {m: cmap(i) for i, m in enumerate(models)}
for name, (title, xy) in views.items():
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
        a.scatter(*np.median(xy[k], axis=0), s=120, marker="X", color="black")
        a.set_title(m, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}: each dot is one passage, black X = the model's median", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(ROOT / "figures" / f"models_2d_{name}.png"), dpi=110)
    print("saved", str(ROOT / "figures" / f"models_2d_{name}.png"))
