"""Turn passages into numbers: style features (TF-IDF) and language-model embeddings."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDING_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def make_skeleton(text, common_words):
    """Keep common words and punctuation, replace every other word with W.

    "The cat sat; it was warm." becomes "the W W ; it was W ."
    """
    pieces = []
    for token in re.findall(r"[A-Za-z']+|[^\s A-Za-z']", text):
        lower = token.lower()
        if lower in common_words:
            pieces.append(lower)
        elif token[0].isalpha() or token[0] == "'":
            pieces.append("W")
        else:
            pieces.append(token)   # punctuation stays as it is
    return " ".join(pieces)


class StyleFeatures:
    """Characters + words + skeleton, all as TF-IDF. Learn from training text, then apply to anything."""

    def __init__(self, n_common_words=150):
        self.n_common_words = n_common_words

    def fit(self, texts):
        texts = list(texts)
        all_words = [w.lower() for t in texts for w in re.findall(r"[A-Za-z']+", t)]
        self.common_words = {word for word, count in Counter(all_words).most_common(self.n_common_words)}

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=20000, dtype=np.float32
        ).fit(texts)
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=15000, dtype=np.float32
        ).fit(texts)
        self.skeleton_vectorizer = TfidfVectorizer(
            ngram_range=(1, 4), token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
            min_df=3, max_features=20000, dtype=np.float32,
        ).fit(self._skeletons(texts))
        return self

    def transform(self, texts):
        texts = list(texts)
        blocks = [
            self.char_vectorizer.transform(texts),
            self.word_vectorizer.transform(texts),
            self.skeleton_vectorizer.transform(self._skeletons(texts)),
        ]
        return np.hstack([block.toarray() for block in blocks])

    def _skeletons(self, texts):
        return [make_skeleton(t, self.common_words) for t in texts]


def embed_passages(texts, cache_file=None):
    """DistilBERT numbers (768 per passage). Needs sentence-transformers; saved to cache_file so it runs once."""
    if cache_file is not None and Path(cache_file).exists():
        return np.load(cache_file)

    from sentence_transformers import SentenceTransformer
    language_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    language_model.max_seq_length = 512
    embeddings = language_model.encode(list(texts), batch_size=16, normalize_embeddings=True)

    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
    return embeddings
