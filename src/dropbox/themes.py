"""Theme matcher using pre-computed theme centroids from 29 merged themes."""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Cached singletons
_EMBEDDING_MODEL = None
_THEME_CENTROIDS = None  # shape: (29, 384)
_THEME_IDS = None        # list of 29 int IDs
_THEME_LABELS = None     # dict mapping theme_id (int) -> label string


def _init_theme_centroids():
    """Build and cache the 29 theme centroids and labels once."""
    global _EMBEDDING_MODEL, _THEME_CENTROIDS, _THEME_IDS, _THEME_LABELS
    if _THEME_CENTROIDS is None or _THEME_LABELS is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        embs_path = os.path.join(repo_root, "data", "processed", "embeddings.npy")
        topics_path = os.path.join(repo_root, "data", "processed", "topics_v1_merged.parquet")
        labels_path = os.path.join(repo_root, "data", "processed", "theme_labels_final.json")

        embeddings = np.load(embs_path)
        topics_df = pd.read_parquet(topics_path)

        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        # Exclude topic -1 (outliers)
        valid_mask = topics_df["topic_v1_merged"] != -1
        assert -1 not in topics_df.loc[valid_mask, "topic_v1_merged"].values, "Outlier topic -1 must be excluded."
        
        unique_topics = sorted(int(t) for t in topics_df.loc[valid_mask, "topic_v1_merged"].unique())
        assert len(unique_topics) == 29, f"Expected 29 themes, found {len(unique_topics)}"

        centroids_list = []
        theme_ids = []
        theme_labels = {}

        for t in unique_topics:
            idxs = topics_df[topics_df["topic_v1_merged"] == t].index.values
            t_embs = embeddings[idxs]
            centroid = np.mean(t_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids_list.append(centroid)
            theme_ids.append(t)
            theme_labels[t] = raw_labels.get(str(t), f"Theme {t}")

        _THEME_CENTROIDS = np.vstack(centroids_list)  # (29, 384)
        _THEME_IDS = theme_ids
        _THEME_LABELS = theme_labels

    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return _EMBEDDING_MODEL, _THEME_CENTROIDS, _THEME_IDS, _THEME_LABELS


def match_themes(texts: List[str], batch_size: int = 64) -> List[str]:
    """Match texts to the 29 themes based on centroid cosine similarity.

    Thresholds: cosine >= 0.30 and margin >= 0.03 over second best.
    Returns: List of assigned theme labels (or 'Unassigned').
    """
    if not texts:
        return []

    model, centroids, theme_ids, theme_labels = _init_theme_centroids()

    # Encode with unit normalization
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False
    )

    # Cosine similarities: shape (N, 29) since both texts and centroids are unit-normalized
    similarities = np.dot(embeddings, centroids.T)

    assigned_themes = []
    for sim_row in similarities:
        # Sort descending
        sorted_indices = np.argsort(sim_row)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        top_sim = float(sim_row[top_idx])
        second_sim = float(sim_row[second_idx])

        margin = top_sim - second_sim

        if top_sim >= 0.30 and margin >= 0.03:
            theme_id = theme_ids[top_idx]
            assigned_themes.append(theme_labels[theme_id])
        else:
            assigned_themes.append("Unassigned")

    return assigned_themes


def get_theme_info() -> Dict[str, Any]:
    """Return theme metadata and diagnostic information."""
    _, centroids, theme_ids, theme_labels = _init_theme_centroids()
    return {
        "num_themes": len(theme_ids),
        "theme_ids": theme_ids,
        "theme_labels": theme_labels
    }
