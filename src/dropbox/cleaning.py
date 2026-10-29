"""Data cleaning utilities for dropbox text reviews."""

import pandas as pd
from typing import Tuple, Dict, Any


def clean_reviews(
    df: pd.DataFrame,
    text_col: str,
    rating_col: str | None = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean reviews by stripping whitespace, dropping empty strings, short texts (<3 words),

    and exact duplicates.

    Returns:
        (df_cleaned, removal_counts_dict)
    """
    total_in = len(df)
    
    # 1. Strip whitespace and identify empty texts
    raw_series = df[text_col]
    stripped = raw_series.fillna("").astype(str).str.strip()
    is_empty = stripped == ""
    removed_empty = int(is_empty.sum())
    
    df_step1 = df[~is_empty].copy()
    stripped_step1 = stripped[~is_empty]

    # 2. Identify texts under 3 words
    word_counts = stripped_step1.apply(lambda s: len(s.split()))
    is_under_3_words = word_counts < 3
    removed_under_3 = int(is_under_3_words.sum())

    df_step2 = df_step1[~is_under_3_words].copy()
    df_step2[text_col] = stripped_step1[~is_under_3_words]

    # 3. Exact duplicates
    is_duplicate = df_step2[text_col].duplicated(keep="first")
    removed_duplicates = int(is_duplicate.sum())

    df_clean = df_step2[~is_duplicate].reset_index(drop=True)

    counts = {
        "removed_empty": removed_empty,
        "removed_under_3_words": removed_under_3,
        "removed_duplicates": removed_duplicates,
        "rows_used": len(df_clean)
    }

    return df_clean, counts
