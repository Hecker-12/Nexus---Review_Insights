import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def count_topics(topic_model: BERTopic) -> int:
    """Count valid topics excluding the outlier topic -1.
    
    Per project standard: len([t for t in model.get_topics() if t != -1])
    """
    return len([t for t in topic_model.get_topics() if t != -1])


def get_default_vectorizer() -> CountVectorizer:
    """Return default vectorizer used across BERTopic models."""
    return CountVectorizer(
        stop_words="english",
        min_df=10,
        ngram_range=(1, 2),
    )


def reduce_outliers_embeddings(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int] | np.ndarray,
    embeddings: np.ndarray,
    vectorizer: CountVectorizer | None = None,
) -> tuple[BERTopic, list[int]]:
    """Reduce outliers using embeddings strategy and update topic representations."""
    if vectorizer is None:
        vectorizer = get_default_vectorizer()

    new_topics = topic_model.reduce_outliers(
        docs,
        topics,
        strategy="embeddings",
        embeddings=embeddings,
    )
    topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)
    return topic_model, new_topics


def sample_reassigned_reviews(
    df: pd.DataFrame,
    orig_topics: list[int] | np.ndarray,
    new_topics: list[int] | np.ndarray,
    topic_model: BERTopic,
    n: int = 8,
    random_state: int = 1,
) -> pd.DataFrame:
    """Sample reviews that were previously outliers (-1) and were reassigned.
    
    Uses np.random.RandomState for reproducible selection.
    """
    orig_arr = np.array(orig_topics)
    new_arr = np.array(new_topics)
    reassigned_mask = (orig_arr == -1) & (new_arr != -1)

    reassigned_indices = np.where(reassigned_mask)[0]
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(reassigned_indices, size=min(n, len(reassigned_indices)), replace=False)

    topic_names = topic_model.get_topic_info().set_index("Topic")["Name"].to_dict()

    sample_df = df.iloc[sample_indices].copy()
    sample_df["new_topic_id"] = new_arr[sample_indices]
    sample_df["new_topic_name"] = [
        topic_names.get(tid, f"Topic {tid}") for tid in sample_df["new_topic_id"]
    ]
    sample_df["text_200"] = sample_df["text"].str.slice(0, 200)
    return sample_df[["review_id", "new_topic_id", "new_topic_name", "text_200"]]


def merge_topics(
    topic_model: BERTopic,
    docs: list[str],
    nr_topics: int = 30,
) -> tuple[BERTopic, list[int]]:
    """Reduce topics to a target number using agglomerative clustering.
    
    Mutates topic_model in-place and returns (topic_model, topics_).
    """
    topic_model.reduce_topics(docs, nr_topics=nr_topics)
    return topic_model, topic_model.topics_


def get_topic_summary(topic_model: BERTopic, exclude_outliers: bool = True) -> pd.DataFrame:
    """Return DataFrame of topic IDs, Names, and Counts."""
    info = topic_model.get_topic_info()[["Topic", "Name", "Count"]].copy()
    if exclude_outliers:
        info = info[info["Topic"] != -1].reset_index(drop=True)
    return info


def create_bertopic_v2(random_state: int = 42) -> BERTopic:
    """Create BERTopic v2 model with looser clustering parameters (min_cluster_size=30, min_samples=5)."""
    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = get_default_vectorizer()

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=True,
    )


def fit_bertopic_v2(
    docs: list[str],
    embeddings: np.ndarray,
    random_state: int = 42,
) -> tuple[BERTopic, list[int], np.ndarray]:
    """Fit BERTopic v2 model and return (topic_model, topics, probs)."""
    model = create_bertopic_v2(random_state=random_state)
    topics, probs = model.fit_transform(docs, embeddings)
    return model, topics, probs


