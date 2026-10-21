# Project Notes & Methodological Details

## 1. Grounded Summaries & Faithfulness Evaluation (Phase 7)
- **Claim Evaluation Statistics**:
  - **168 praise and complaint claims** evaluated across 29 themes.
  - **6 claims not checked** (Theme 24: Spanish-language reviews, skipped for English word-overlap).
  - **1 ID dropped** across all themes (Review ID `2706` in Theme 2 was an out-of-sample citation).
  - **2 flagged claims** (word-overlap false alarms in Theme 14 and Theme 28).
- **Summary Overviews**:
  - Overview paragraphs are **uncited by design** and reported separately from praise and complaint claims.

## 2. Topic Modeling & Coherence (Phases 0–4)
- **Topic Counts**:
  - The merged BERTopic model has **29 real themes** plus an outlier group `-1` (total `nr_topics=30` including `-1`).
  - Standard topic count excludes the outlier group: `len([t for t in model.get_topics() if t != -1])`.
- **Coherence (NPMI)**:
  - *Coherence depends on topic count and outlier handling; BERTopic NPMI is computed on clustered topics only, excluding the outlier group.*

## 3. Database Architecture & Temporal Limitations (Phase 8)
- **Database Location**: `data/processed/insights.db`
- **Tables**:
  - `reviews(review_id, text, sentiment_true, pred_ft, topic_id, topic_id_reduced)`
  - `themes(topic_id, label, count, pct_negative)`
  - `summaries(topic_id, summary_json)`
- **Temporal Analysis**:
  - **No dates in the dataset, so no trends table**: The Amazon reviews dataset used in this project does not contain review timestamp or date attributes. Therefore, longitudinal trend tracking and time-series aggregations cannot be performed, and no trends table is included in the database.
- **Sentiment & Predictions**:
  - `sentiment_true` contains ground truth sentiment for all 50,000 reviews.
  - `pred_ft` contains fine-tuned DistilBERT predictions for the 10,000 test reviews (and `NULL` for the remaining 40,000 reviews).
  - `pct_negative` in `themes` is calculated using `sentiment_true` across all reviews in each topic.
