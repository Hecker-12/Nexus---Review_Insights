"""End-to-end pipeline for scoring and analyzing uploaded review data in memory."""

import time
import pandas as pd
from typing import Optional, Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score

from src.dropbox.loader import load_file
from src.dropbox.cleaning import clean_reviews
from src.dropbox.pii import scrub_pii
from src.dropbox.scorer import score_texts
from src.dropbox.themes import match_themes

WARNING_MESSAGES = [
    "Models were trained on Amazon product reviews; accuracy on other text may be lower.",
    "Themes are the 29 Amazon-product themes found in this project; they may not fit your data."
]


def sanitize_formula(val: str) -> str:
    """Guard against CSV / spreadsheet formula injection."""
    if val and isinstance(val, str) and val[0] in ("=", "+", "-", "@", "\t"):
        return "'" + val
    return val


def analyze(
    file_bytes: bytes,
    filename: str,
    text_column: Optional[str] = None,
    include_themes: bool = True
) -> Dict[str, Any]:
    """Execute end-to-end analysis on uploaded file bytes without writing to disk.

    Returns:
        Structured result dictionary.
    """
    t0_total = time.time()
    timings = {}

    # 1. Load file
    t0 = time.time()
    df_raw, detected_text_col, detected_rating_col, rows_in, sampled = load_file(
        file_bytes=file_bytes,
        filename=filename,
        text_column=text_column
    )
    timings["load_seconds"] = round(time.time() - t0, 4)

    # 2. Clean reviews
    t0 = time.time()
    df_clean, removal_counts = clean_reviews(
        df=df_raw,
        text_col=detected_text_col,
        rating_col=detected_rating_col
    )
    timings["clean_seconds"] = round(time.time() - t0, 4)

    rows_used = len(df_clean)
    if rows_used == 0:
        return {
            "report": {
                "rows_in": rows_in,
                "rows_used": 0,
                "removed_empty": removal_counts["removed_empty"],
                "removed_under_3_words": removal_counts["removed_under_3_words"],
                "removed_duplicates": removal_counts["removed_duplicates"],
                "sampled": sampled,
                "detected_text_column": detected_text_col,
                "detected_rating_column": detected_rating_col,
                "timings": timings,
            },
            "summary": {"positive": 0, "negative": 0, "pct_positive": 0.0, "pct_negative": 0.0},
            "accuracy_vs_ratings": None,
            "themes_table": [],
            "preview": [],
            "rows": [],
            "warnings": WARNING_MESSAGES
        }

    # 3. PII Scrubbing
    t0 = time.time()
    raw_texts = df_clean[detected_text_col].tolist()
    scrubbed_texts = [scrub_pii(t) for t in raw_texts]
    timings["pii_seconds"] = round(time.time() - t0, 4)

    # 4. Sentiment Scoring
    t0 = time.time()
    scored_results = score_texts(scrubbed_texts)
    timings["score_seconds"] = round(time.time() - t0, 4)

    # 5. Themes matching (optional)
    t0 = time.time()
    if include_themes:
        theme_labels = match_themes(scrubbed_texts)
    else:
        theme_labels = ["Unassigned"] * rows_used
    timings["theme_seconds"] = round(time.time() - t0, 4)

    timings["total_seconds"] = round(time.time() - t0_total, 4)

    # Assemble rows list
    all_rows = []
    positive_count = 0
    negative_count = 0

    for idx, (scrubbed, (label, prob), theme) in enumerate(
        zip(scrubbed_texts, scored_results, theme_labels)
    ):
        if label == "positive":
            positive_count += 1
        else:
            negative_count += 1

        all_rows.append({
            "row_id": idx,
            "scrubbed_text": sanitize_formula(scrubbed),
            "label": label,
            "probability": prob,
            "theme": theme
        })

    # Summary KPIs
    pct_pos = round((positive_count / rows_used) * 100, 2) if rows_used > 0 else 0.0
    pct_neg = round((negative_count / rows_used) * 100, 2) if rows_used > 0 else 0.0
    summary = {
        "positive": positive_count,
        "negative": negative_count,
        "pct_positive": pct_pos,
        "pct_negative": pct_neg
    }

    # Accuracy vs Ratings (if rating column detected)
    accuracy_vs_ratings = None
    if detected_rating_col is not None:
        raw_ratings = pd.to_numeric(df_clean[detected_rating_col], errors="coerce")
        # 1-2 negative, 4-5 positive, ignore 3
        eval_mask = raw_ratings.isin([1, 2, 4, 5])
        ignored_3_count = int((raw_ratings == 3).sum())
        eval_count = int(eval_mask.sum())

        if eval_count > 0:
            y_true = ["negative" if r in [1, 2] else "positive" for r in raw_ratings[eval_mask]]
            y_pred = [all_rows[i]["label"] for i in range(rows_used) if eval_mask.iloc[i]]
            acc = round(accuracy_score(y_true, y_pred), 4)
            macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
            accuracy_vs_ratings = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "evaluated_count": eval_count,
                "ignored_rating_3_count": ignored_3_count,
                "note": "Ratings 1-2 treated as negative, 4-5 as positive; neutral rating 3 ignored."
            }

    # Themes table
    themes_table = []
    if include_themes:
        theme_counts: Dict[str, Dict[str, int]] = {}
        for row in all_rows:
            th = row["theme"]
            if th not in theme_counts:
                theme_counts[th] = {"total": 0, "negative": 0}
            theme_counts[th]["total"] += 1
            if row["label"] == "negative":
                theme_counts[th]["negative"] += 1

        for th, stats in theme_counts.items():
            tot = stats["total"]
            neg = stats["negative"]
            share = round((tot / rows_used) * 100, 2)
            pct_neg_theme = round((neg / tot) * 100, 2) if tot > 0 else 0.0
            themes_table.append({
                "label": th,
                "count": tot,
                "share": share,
                "pct_negative": pct_neg_theme
            })

        themes_table.sort(key=lambda x: x["count"], reverse=True)

    preview = all_rows[:20]

    return {
        "report": {
            "rows_in": rows_in,
            "rows_used": rows_used,
            "removed_empty": removal_counts["removed_empty"],
            "removed_under_3_words": removal_counts["removed_under_3_words"],
            "removed_duplicates": removal_counts["removed_duplicates"],
            "sampled": sampled,
            "detected_text_column": detected_text_col,
            "detected_rating_column": detected_rating_col,
            "timings": timings
        },
        "summary": summary,
        "accuracy_vs_ratings": accuracy_vs_ratings,
        "themes_table": themes_table,
        "preview": preview,
        "rows": all_rows,
        "warnings": WARNING_MESSAGES
    }
