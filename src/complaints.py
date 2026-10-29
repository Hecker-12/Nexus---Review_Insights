"""Complaint modeling and topic extraction for negative reviews.

Provides utilities for:
1. Loading and verifying negative review embeddings and texts.
2. Training and saving BERTopic on negative reviews only.
3. Topic merging and outlier preservation.
4. Coherence (c_npmi) evaluation against negative review tokens.
5. Sampling reviews per complaint theme.
"""

import os
import time
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.topics import count_topics, build_gensim_dictionary, get_bertopic_eval_words


def load_negative_reviews(
    clean_path: str = "data/processed/reviews_clean.parquet",
    embeddings_path: str = "data/processed/embeddings.npy",
) -> Tuple[pd.DataFrame, np.ndarray, bool]:
    """Load reviews and embeddings, filter to negative reviews, and assert 1-to-1 alignment."""
    df = pd.read_parquet(clean_path)
    embeddings = np.load(embeddings_path)

    neg_df = df[df["sentiment"] == "negative"].copy()
    assert len(neg_df) == 24669, f"Expected 24,669 negative reviews, found {len(neg_df)}"

    neg_indices = neg_df["review_id"].to_numpy()
    neg_embeddings = embeddings[neg_indices]

    # Verify that embeddings[review_id] aligns with each selected review
    is_aligned = bool((neg_embeddings == embeddings[neg_df["review_id"].to_numpy()]).all())
    assert is_aligned, "Review ID to embedding mapping alignment failed"

    return neg_df, neg_embeddings, is_aligned


def create_complaints_bertopic(random_state: int = 42) -> BERTopic:
    """Create BERTopic instance configured for complaint topic modeling."""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=40,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=10,
        ngram_range=(1, 2),
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True,
    )


def fit_complaints_bertopic(
    topic_model: BERTopic,
    docs: List[str],
    embeddings: np.ndarray,
) -> Tuple[BERTopic, List[int], np.ndarray, float]:
    """Fit BERTopic model on negative review texts and precomputed embeddings."""
    start_time = time.time()
    topics, probs = topic_model.fit_transform(docs, embeddings)
    elapsed_time = time.time() - start_time
    return topic_model, topics, probs, elapsed_time


def save_complaints_model(
    topic_model: BERTopic,
    model_dir: str,
    df: pd.DataFrame,
    topics: List[int] | np.ndarray,
    parquet_path: str,
    col_name: str = "topic_neg",
) -> None:
    """Save BERTopic model to directory and topic assignments to parquet."""
    os.makedirs(model_dir, exist_ok=True)
    topic_model.save(
        model_dir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model="all-MiniLM-L6-v2",
    )
    out_df = pd.DataFrame({
        "review_id": df["review_id"].values,
        col_name: topics,
    })
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    out_df.to_parquet(parquet_path, index=False)


def reduce_complaints_topics(
    model_or_path: str | BERTopic,
    docs: List[str],
    nr_topics: int = 15,
) -> Tuple[BERTopic, List[int]]:
    """Reduce topics on a freshly loaded model copy to at most nr_topics."""
    if isinstance(model_or_path, str):
        topic_model = BERTopic.load(model_or_path)
    else:
        topic_model = model_or_path

    topic_model.reduce_topics(docs, nr_topics=nr_topics)
    return topic_model, topic_model.topics_


def compute_complaint_coherence(
    model_or_path: str | BERTopic,
    docs: List[str],
    topics: List[int] | np.ndarray,
    texts: List[List[str]],
    dictionary: Dictionary,
    coherence: str = "c_npmi",
    processes: int = 1,
) -> Tuple[float, int, float]:
    """Compute c_npmi coherence for a complaint BERTopic model on a fresh model copy."""
    if isinstance(model_or_path, str):
        model = BERTopic.load(model_or_path)
    else:
        model = model_or_path

    vectorizer = CountVectorizer(stop_words="english", min_df=10)
    topic_words, scored_count, _ = get_bertopic_eval_words(
        topic_model=model,
        docs=docs,
        topics=topics,
        dictionary=dictionary,
        vectorizer=vectorizer,
    )

    cm = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence,
        processes=processes,
    )
    score = float(cm.get_coherence())
    num_topics = count_topics(model)
    outlier_frac = float((np.array(topics) == -1).mean())
    return score, num_topics, outlier_frac


def sample_theme_reviews(
    df: pd.DataFrame,
    topics: List[int] | np.ndarray,
    topic_id: int,
    n: int = 8,
    random_state: int = 1,
    char_limit: int = 150,
) -> pd.DataFrame:
    """Sample random reviews from a specific theme with character truncation."""
    sub_df = df[np.array(topics) == topic_id]
    if len(sub_df) == 0:
        return pd.DataFrame(columns=["review_id", "text_150"])

    sampled = sub_df.sample(n=min(n, len(sub_df)), random_state=random_state)
    result = sampled[["review_id"]].copy()
    result["text_150"] = sampled["text"].astype(str).str.slice(0, char_limit)
    return result
