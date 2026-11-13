import os
import sqlite3
import threading
from typing import Optional, Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()

DB_PATH = os.path.join("data", "processed", "insights.db")
EMBEDDINGS_PATH = os.path.join("data", "processed", "embeddings.npy")
RESULTS_DIR = "results"

_embeddings_lock = threading.Lock()
_embeddings_array: Optional[np.ndarray] = None


def get_readonly_db():
    """Connect to insights.db in read-only mode."""
    abs_db = os.path.abspath(DB_PATH)
    uri = f"file:{abs_db}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_embeddings(request: Optional[Request] = None) -> np.ndarray:
    """Load embeddings once with mmap_mode='r' under a lock, or reuse from app.state."""
    global _embeddings_array
    with _embeddings_lock:
        if _embeddings_array is None:
            # Check if app.state already holds embeddings
            if (
                request is not None
                and hasattr(request.app, "state")
                and hasattr(request.app.state, "embeddings")
                and request.app.state.embeddings is not None
            ):
                _embeddings_array = request.app.state.embeddings
            else:
                _embeddings_array = np.load(EMBEDDINGS_PATH, mmap_mode="r")

            # Assert embeddings row count equals reviews table row count and print both
            conn = get_readonly_db()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM reviews;")
                reviews_count = cur.fetchone()[0]
            finally:
                conn.close()

            emb_count = _embeddings_array.shape[0]
            print(f"ExploreRouter: Embeddings rows={emb_count}, Reviews table rows={reviews_count}")
            assert emb_count == reviews_count, (
                f"Mismatch between embeddings ({emb_count}) and reviews table ({reviews_count})"
            )

        return _embeddings_array


# --- Pydantic Response Models ---

class ReviewDetail(BaseModel):
    review_id: int
    text: str
    sentiment_true: str
    theme_id: int
    theme_label: str


class SimilarItem(BaseModel):
    review_id: int
    score: float
    text: str
    sentiment_true: str
    theme_label: str


# --- Whitelist of Methodology Files ---

METHODOLOGY_WHITELIST = [
    "sentiment_comparison.csv",
    "topic_comparison.csv",
    "evaluation_upgrade.csv",
    "mcnemar_tests.csv",
    "calibration.csv",
    "calibration_summary.csv",
    "slice_analysis.csv",
    "complaint_type_counts_v2.csv",
    "complaint_by_theme_v2.csv",
]


# --- Endpoints ---

@router.get("/review/{review_id}", response_model=ReviewDetail)
def get_review(
    review_id: int = Path(..., ge=0, description="Non-negative integer review ID")
):
    """Fetch review text, sentiment_true, theme_id, and theme_label by review_id."""
    conn = get_readonly_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.review_id, r.text, r.sentiment_true, r.topic_id, t.label as theme_label
            FROM reviews r
            JOIN themes t ON r.topic_id = t.topic_id
            WHERE r.review_id = ?;
            """,
            (review_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Review ID {review_id} not found"
            )
        return ReviewDetail(
            review_id=row["review_id"],
            text=row["text"],
            sentiment_true=row["sentiment_true"],
            theme_id=row["topic_id"],
            theme_label=row["theme_label"],
        )
    finally:
        conn.close()


@router.get("/similar/{review_id}", response_model=list[SimilarItem])
def get_similar_reviews(
    request: Request,
    review_id: int = Path(..., ge=0, description="Target review ID"),
    k: int = Query(10, ge=1, le=20, description="Number of neighbours (1 to 20)"),
):
    """Return top-k cosine similarity neighbours for a review from precomputed embeddings."""
    embeddings = get_embeddings(request)

    if review_id < 0 or review_id >= embeddings.shape[0]:
        raise HTTPException(
            status_code=404,
            detail=f"Review ID {review_id} not found"
        )

    # Confirm review exists in database
    conn = get_readonly_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT review_id FROM reviews WHERE review_id = ?;", (review_id,))
        if cur.fetchone() is None:
            raise HTTPException(
                status_code=404,
                detail=f"Review ID {review_id} not found in reviews database"
            )

        # Compute cosine similarity
        target_vec = embeddings[review_id]
        similarities = np.dot(embeddings, target_vec).copy()
        # Exclude the review itself
        similarities[review_id] = -np.inf

        # Top-k indices sorted descending
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_ids = [int(idx) for idx in top_k_indices]

        # Fetch metadata
        placeholders = ",".join("?" for _ in top_k_ids)
        cur.execute(
            f"""
            SELECT r.review_id, r.text, r.sentiment_true, t.label as theme_label
            FROM reviews r
            JOIN themes t ON r.topic_id = t.topic_id
            WHERE r.review_id IN ({placeholders});
            """,
            top_k_ids,
        )
        rows_by_id = {row["review_id"]: row for row in cur.fetchall()}

        results = []
        for idx in top_k_ids:
            if idx in rows_by_id:
                r = rows_by_id[idx]
                results.append(
                    SimilarItem(
                        review_id=r["review_id"],
                        score=round(float(similarities[idx]), 4),
                        text=r["text"][:300],
                        sentiment_true=r["sentiment_true"],
                        theme_label=r["theme_label"],
                    )
                )

        return results
    finally:
        conn.close()


@router.get("/methodology")
def get_methodology():
    """Return structured JSON for whitelisted methodology CSV files.
    Missing files return null for that key (200 OK).
    NaN and inf values are converted to null.
    """
    output = {}

    for filename in METHODOLOGY_WHITELIST:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(filepath):
            output[filename] = None
            continue

        try:
            df = pd.read_csv(filepath)
            df = df.replace([np.inf, -np.inf], np.nan)
            columns = list(df.columns)

            clean_rows = []
            for row in df.itertuples(index=False):
                clean_row = [
                    None if (pd.isna(val) or val is None) else val
                    for val in row
                ]
                clean_rows.append(clean_row)

            output[filename] = {
                "columns": columns,
                "rows": clean_rows,
            }
        except Exception:
            output[filename] = None

    return output

