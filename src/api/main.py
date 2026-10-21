import os
import sqlite3
import json
from contextlib import asynccontextmanager
from typing import Optional, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join("data", "processed", "insights.db")
EMBEDDINGS_PATH = os.path.join("data", "processed", "embeddings.npy")


# --- Pydantic Models ---

class OverallSentiment(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    pct_negative: float


class ThemeSentiment(BaseModel):
    topic_id: int
    label: str
    count: int
    positive_count: int
    negative_count: int
    pct_negative: float


class SentimentResponse(BaseModel):
    overall: OverallSentiment
    themes: list[ThemeSentiment]


class ThemeItem(BaseModel):
    topic_id: int
    label: str
    count: int
    pct_negative: float


class SearchResultItem(BaseModel):
    review_id: int
    text: str
    sentiment_true: str
    topic_id: int
    topic_label: str
    similarity_score: float


class ThemeSummaryResponse(BaseModel):
    theme_id: int
    theme_label: str
    sample_size: int = 30
    sample_review_ids: list[int] = Field(default_factory=list)
    summary: Any
    praises: list[Any] = Field(default_factory=list)
    complaints: list[Any] = Field(default_factory=list)
    claims: list[Any] = Field(default_factory=list)
    flags: list[Any] = Field(default_factory=list)


# --- Lifespan Context Manager (Startup / Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model and embeddings once at startup
    print("Loading SentenceTransformer('all-MiniLM-L6-v2') at startup...")
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Loading precomputed embeddings from {EMBEDDINGS_PATH}...")
    app.state.embeddings = np.load(EMBEDDINGS_PATH)
    app.state.db_path = DB_PATH
    yield


app = FastAPI(
    title="Amazon Review Insights API",
    description="FastAPI backend for sentiment analysis, theme extraction, and grounded summaries.",
    version="1.0.0",
    lifespan=lifespan
)


def get_db():
    """Database dependency yielding a SQLite connection."""
    conn = sqlite3.connect(app.state.db_path if hasattr(app.state, "db_path") else DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# --- Endpoints ---

@app.get("/sentiment", response_model=SentimentResponse)
def get_sentiment(db: sqlite3.Connection = Depends(get_db)):
    """Return overall sentiment distribution plus per-theme sentiment distribution."""
    # 1. Overall sentiment
    cur = db.cursor()
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN sentiment_true = 'positive' THEN 1 ELSE 0 END) as pos,
            SUM(CASE WHEN sentiment_true = 'negative' THEN 1 ELSE 0 END) as neg
        FROM reviews;
    """)
    overall_row = cur.fetchone()
    total = overall_row["total"] or 0
    pos = overall_row["pos"] or 0
    neg = overall_row["neg"] or 0
    pct_neg = round((neg / total) * 100.0, 2) if total > 0 else 0.0

    overall = OverallSentiment(
        total_reviews=total,
        positive_count=pos,
        negative_count=neg,
        pct_negative=pct_neg
    )

    # 2. Per-theme sentiment (including Unclustered -1)
    cur.execute("""
        SELECT 
            r.topic_id,
            t.label,
            COUNT(*) as count,
            SUM(CASE WHEN r.sentiment_true = 'positive' THEN 1 ELSE 0 END) as pos,
            SUM(CASE WHEN r.sentiment_true = 'negative' THEN 1 ELSE 0 END) as neg,
            t.pct_negative
        FROM reviews r
        JOIN themes t ON r.topic_id = t.topic_id
        GROUP BY r.topic_id, t.label, t.pct_negative
        ORDER BY r.topic_id;
    """)
    theme_rows = cur.fetchall()

    theme_list = [
        ThemeSentiment(
            topic_id=row["topic_id"],
            label=row["label"],
            count=row["count"],
            positive_count=row["pos"] or 0,
            negative_count=row["neg"] or 0,
            pct_negative=row["pct_negative"]
        )
        for row in theme_rows
    ]

    return SentimentResponse(overall=overall, themes=theme_list)


@app.get("/themes", response_model=list[ThemeItem])
def get_themes(db: sqlite3.Connection = Depends(get_db)):
    """Return all 29 themes plus Unclustered (-1)."""
    cur = db.cursor()
    cur.execute("SELECT topic_id, label, count, pct_negative FROM themes ORDER BY topic_id;")
    rows = cur.fetchall()
    return [
        ThemeItem(
            topic_id=row["topic_id"],
            label=row["label"],
            count=row["count"],
            pct_negative=row["pct_negative"]
        )
        for row in rows
    ]


@app.get("/summary/{theme_id}", response_model=ThemeSummaryResponse)
def get_summary(theme_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Return grounded summary for a specific theme. Returns 404 for unknown theme IDs."""
    cur = db.cursor()
    cur.execute("SELECT summary_json FROM summaries WHERE topic_id = ?;", (theme_id,))
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Theme summary for theme_id {theme_id} not found")

    try:
        data = json.loads(row["summary_json"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode summary JSON: {e}")


@app.get("/search", response_model=list[SearchResultItem])
def search_reviews(
    q: Optional[str] = Query(None, description="Search query string"),
    db: sqlite3.Connection = Depends(get_db)
):
    """Semantic search against reviews using all-MiniLM-L6-v2 cosine similarity.
    Rejects empty or whitespace-only queries with 400.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Search query 'q' cannot be empty")

    query_str = q.strip()

    # Get model and embeddings from app state
    if not hasattr(app.state, "model") or not hasattr(app.state, "embeddings"):
        # Fallback in case app was initialized without lifespan running
        app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
        app.state.embeddings = np.load(EMBEDDINGS_PATH)

    # 1. Embed query
    query_emb = app.state.model.encode(query_str, normalize_embeddings=True)

    # 2. Compute cosine similarity against all 50,000 unit-norm review embeddings
    similarities = np.dot(app.state.embeddings, query_emb)

    # 3. Get top 10 review indices
    top10_indices = np.argsort(similarities)[-10:][::-1]
    top10_ids = [int(idx) for idx in top10_indices]

    # 4. Fetch review text and metadata from insights.db
    placeholders = ",".join("?" for _ in top10_ids)
    cur = db.cursor()
    cur.execute(f"""
        SELECT r.review_id, r.text, r.sentiment_true, r.topic_id, t.label as topic_label
        FROM reviews r
        JOIN themes t ON r.topic_id = t.topic_id
        WHERE r.review_id IN ({placeholders});
    """, top10_ids)
    rows_by_id = {row["review_id"]: row for row in cur.fetchall()}

    # 5. Format results in descending order of similarity
    results = []
    for idx in top10_ids:
        if idx in rows_by_id:
            r = rows_by_id[idx]
            results.append(SearchResultItem(
                review_id=r["review_id"],
                text=r["text"],
                sentiment_true=r["sentiment_true"],
                topic_id=r["topic_id"],
                topic_label=r["topic_label"],
                similarity_score=round(float(similarities[idx]), 4)
            ))

    return results
