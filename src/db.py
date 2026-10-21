import os
import sqlite3
import json
import pandas as pd


def build_insights_database(
    db_path: str = os.path.join("data", "processed", "insights.db"),
    reviews_path: str = os.path.join("data", "processed", "reviews_clean.parquet"),
    ft_preds_path: str = os.path.join("data", "processed", "ft_preds.parquet"),
    topics_merged_path: str = os.path.join("data", "processed", "topics_v1_merged.parquet"),
    topics_reduced_path: str = os.path.join("data", "processed", "topics_v1_reduced.parquet"),
    theme_labels_path: str = os.path.join("data", "processed", "theme_labels_final.json"),
    theme_summaries_path: str = os.path.join("data", "processed", "theme_summaries.json"),
) -> sqlite3.Connection:
    """Build and populate the SQLite database for Review Insights."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Remove existing db file if present to guarantee clean schema & data
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Create tables
    cur.execute("""
    CREATE TABLE reviews (
        review_id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        sentiment_true TEXT NOT NULL,
        pred_ft TEXT,
        topic_id INTEGER NOT NULL,
        topic_id_reduced INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE themes (
        topic_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        count INTEGER NOT NULL,
        pct_negative REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE summaries (
        topic_id INTEGER PRIMARY KEY,
        summary_json TEXT NOT NULL
    );
    """)

    # 2. Populate reviews table
    print("Loading data for 'reviews' table...")
    df_reviews = pd.read_parquet(reviews_path)
    df_reviews = df_reviews.rename(columns={"sentiment": "sentiment_true"})

    # Left-join fine-tuned predictions (exists only for 10,000 test reviews)
    df_ft = pd.read_parquet(ft_preds_path)[["review_id", "pred_ft"]]
    df_merged = df_reviews.merge(df_ft, on="review_id", how="left")

    # Merge topic assignments
    df_topics_m = pd.read_parquet(topics_merged_path)[["review_id", "topic_v1_merged"]]
    df_topics_m = df_topics_m.rename(columns={"topic_v1_merged": "topic_id"})
    df_merged = df_merged.merge(df_topics_m, on="review_id", how="left")

    df_topics_r = pd.read_parquet(topics_reduced_path)[["review_id", "topic_v1_reduced"]]
    df_topics_r = df_topics_r.rename(columns={"topic_v1_reduced": "topic_id_reduced"})
    df_merged = df_merged.merge(df_topics_r, on="review_id", how="left")

    # Ensure correct column ordering
    cols_order = ["review_id", "text", "sentiment_true", "pred_ft", "topic_id", "topic_id_reduced"]
    df_reviews_final = df_merged[cols_order].copy()
    
    # Insert reviews into SQLite
    df_reviews_final.to_sql("reviews", conn, if_exists="append", index=False)
    print(f"Inserted {len(df_reviews_final)} rows into 'reviews'.")

    # 3. Populate themes table
    print("Loading data for 'themes' table...")
    with open(theme_labels_path, "r", encoding="utf-8") as f:
        theme_labels = json.load(f)

    # Compute review counts and pct_negative per topic using sentiment_true across all 50k reviews
    theme_rows = []
    # All topic IDs: -1 (Unclustered) and 0 through 28
    all_topic_ids = [-1] + sorted([int(k) for k in theme_labels.keys() if int(k) != -1])

    for tid in all_topic_ids:
        label = "Unclustered" if tid == -1 else theme_labels[str(tid)]
        sub = df_reviews_final[df_reviews_final["topic_id"] == tid]
        cnt = len(sub)
        neg_cnt = (sub["sentiment_true"] == "negative").sum()
        pct_neg = round((neg_cnt / cnt) * 100.0, 2) if cnt > 0 else 0.0

        theme_rows.append((tid, label, cnt, pct_neg))

    cur.executemany(
        "INSERT INTO themes (topic_id, label, count, pct_negative) VALUES (?, ?, ?, ?);",
        theme_rows
    )
    print(f"Inserted {len(theme_rows)} rows into 'themes'.")

    # 4. Populate summaries table
    print("Loading data for 'summaries' table...")
    with open(theme_summaries_path, "r", encoding="utf-8") as f:
        theme_summaries = json.load(f)

    summary_rows = []
    for tid_str, t_data in theme_summaries.items():
        tid = int(tid_str)
        # Store full summary JSON
        summary_rows.append((tid, json.dumps(t_data, ensure_ascii=False)))

    cur.executemany(
        "INSERT INTO summaries (topic_id, summary_json) VALUES (?, ?);",
        summary_rows
    )
    print(f"Inserted {len(summary_rows)} rows into 'summaries'.")

    # 5. Create indexes
    print("Creating indexes on reviews table...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_topic_id ON reviews(topic_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_review_id ON reviews(review_id);")

    conn.commit()
    print("Database build complete.")
    return conn
