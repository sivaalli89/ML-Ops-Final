"""Find which passages answer the same prompt (cluster by topic)."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

N_MODELS = 18   # every model answers every prompt once, so a full group has 18 passages
MIN_BATCH = 54  # below this (3 prompts) the topic words are too few to cluster reliably


def cluster_passages(texts, n_groups=None, n_models=N_MODELS, seed=42):
    """Return one group number per passage.

    Only rare content words are used, so passages cluster by subject and not by author.
    """
    texts = list(texts)
    if n_groups is None:
        n_groups = max(1, round(len(texts) / n_models))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        sublinear_tf=True,
        min_df=2,
        max_df=0.05,
    )
    word_weights = vectorizer.fit_transform(texts)

    # squash to at most 200 numbers per passage (this step is called LSA)
    n_parts = min(200, word_weights.shape[1] - 1, word_weights.shape[0] - 1)
    topic_vectors = normalize(TruncatedSVD(n_parts, random_state=seed).fit_transform(word_weights))

    clusterer = AgglomerativeClustering(n_clusters=n_groups, metric="cosine", linkage="average")
    return clusterer.fit_predict(topic_vectors)


def cluster_quality(groups, models=None, prompt_ids=None):
    """How good are the groups? A good group has about 18 passages and never the same model twice."""
    groups = np.asarray(groups)
    sizes = np.bincount(groups)
    quality = {"n_groups": len(sizes), "median_size": float(np.median(sizes))}
    if models is not None:
        models = np.asarray(models)
        repeats = [len(set(models[groups == g])) < (groups == g).sum() for g in np.unique(groups)]
        quality["share_with_repeat"] = float(np.mean(repeats))
    if prompt_ids is not None:
        quality["ari"] = float(adjusted_rand_score(prompt_ids, groups))
    return quality
