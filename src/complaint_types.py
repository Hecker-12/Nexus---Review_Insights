"""Complaint-type taxonomy and semantic classification for negative reviews.

Provides utilities for:
1. Loading and verifying negative review embeddings and texts.
2. Embedding prototype complaint sentences with all-MiniLM-L6-v2.
3. Zero-shot classification via prototype centroid cosine similarity with margin thresholding.
4. Sampling reviews by complaint type.
5. Cross-tabulating complaint types across product themes.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.complaints import load_negative_reviews


def load_complaint_prototypes(config_path: str = "config/complaint_types.json") -> Dict[str, List[str]]:
    """Load complaint type prototypes from configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_type_vectors(
    prototypes: Dict[str, List[str]],
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[List[str], np.ndarray]:
    """Embed prototype sentences and compute unit-normalized centroid vectors per type."""
    model = SentenceTransformer(model_name)
    type_names = list(prototypes.keys())
    type_vectors = []

    for name in type_names:
        sentences = prototypes[name]
        embs = model.encode(sentences, normalize_embeddings=True)
        mean_vec = embs.mean(axis=0)
        norm_vec = mean_vec / np.linalg.norm(mean_vec)
        type_vectors.append(norm_vec)

    return type_names, np.array(type_vectors)


def classify_negative_reviews(
    neg_embeddings: np.ndarray,
    type_names: List[str],
    type_vectors: np.ndarray,
    min_similarity: float = 0.30,
    min_margin: float = 0.03,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Classify negative reviews by cosine similarity and margin against type prototypes.
    
    Returns (assigned_types, top_similarities, margins).
    """
    # Dot product of unit-norm vectors equals cosine similarity
    sims = neg_embeddings @ type_vectors.T

    # Sort similarities per row descending
    sorted_indices = np.argsort(-sims, axis=1)
    top_indices = sorted_indices[:, 0]
    runner_up_indices = sorted_indices[:, 1]

    top_sims = np.take_along_axis(sims, top_indices[:, None], axis=1).squeeze(axis=1)
    runner_up_sims = np.take_along_axis(sims, runner_up_indices[:, None], axis=1).squeeze(axis=1)
    margins = top_sims - runner_up_sims

    assigned_types = []
    for i in range(len(neg_embeddings)):
        if top_sims[i] >= min_similarity and margins[i] >= min_margin:
            assigned_types.append(type_names[top_indices[i]])
        else:
            assigned_types.append("Other / unclear")

    return assigned_types, top_sims, margins


def sample_complaint_reviews(
    df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    complaint_type: str,
    n: int = 10,
    random_state: int = 1,
    char_limit: int = 200,
) -> pd.DataFrame:
    """Sample assigned reviews for a specific complaint type with truncated text."""
    merged = df.merge(assignments_df, on="review_id")
    matching = merged[merged["complaint_type"] == complaint_type]

    if len(matching) == 0:
        return pd.DataFrame(columns=["review_id", "similarity", "margin", "text_200"])

    sampled = matching.sample(n=min(n, len(matching)), random_state=random_state)
    result = sampled[["review_id", "similarity", "margin"]].copy()
    result["text_200"] = sampled["text"].astype(str).str.slice(0, char_limit)
    return result


def cross_complaint_by_product_theme(
    assignments_df: pd.DataFrame,
    topics_path: str = "data/processed/topics_v1_merged.parquet",
    labels_path: str = "data/processed/theme_labels_final.json",
    min_theme_count: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-tabulate complaint types by product themes as share of theme negative reviews.
    
    Returns (full_crosstab_shares, top_pairs_filtered).
    """
    topics_df = pd.read_parquet(topics_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        theme_labels = json.load(f)

    # Merge negative reviews assignments with product themes
    merged = assignments_df.merge(topics_df, on="review_id", how="inner")
    merged["product_theme"] = merged["topic_v1_merged"].apply(
        lambda tid: theme_labels.get(str(tid), "Unclustered" if tid == -1 else f"Theme {tid}")
    )

    # Compute contingency table (counts)
    counts_table = pd.crosstab(merged["product_theme"], merged["complaint_type"])
    theme_totals = counts_table.sum(axis=1)

    # Convert to shares per theme (row-wise division)
    shares_table = counts_table.div(theme_totals, axis=0)

    # Identify themes with at least min_theme_count negative reviews
    eligible_themes = theme_totals[theme_totals >= min_theme_count].index

    # Find top theme-and-type pairs by share among eligible themes
    pairs = []
    for theme in eligible_themes:
        total_n = int(theme_totals[theme])
        for ctype in counts_table.columns:
            if ctype == "Other / unclear":
                continue
            cnt = int(counts_table.loc[theme, ctype])
            shr = float(shares_table.loc[theme, ctype])
            pairs.append({
                "product_theme": theme,
                "complaint_type": ctype,
                "theme_negative_count": total_n,
                "complaint_count": cnt,
                "share": shr,
            })

    top_pairs_df = pd.DataFrame(pairs).sort_values("share", ascending=False).reset_index(drop=True)
    return shares_table, top_pairs_df


def compute_type_transitions(
    v1_df: pd.DataFrame,
    v2_df: pd.DataFrame,
) -> Tuple[int, pd.DataFrame]:
    """Compute transitions between v1 and v2 complaint classifications.
    
    Returns (total_changed, transitions_table).
    """
    merged = v1_df[["review_id", "complaint_type"]].merge(
        v2_df[["review_id", "complaint_type"]],
        on="review_id",
        suffixes=("_v1", "_v2"),
    )
    changed_mask = merged["complaint_type_v1"] != merged["complaint_type_v2"]
    total_changed = int(changed_mask.sum())

    changed_df = merged[changed_mask]
    transitions = (
        changed_df.groupby(["complaint_type_v1", "complaint_type_v2"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return total_changed, transitions
