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

### Highest Negative Sentiment Query
Top 5 themes with the highest `pct_negative` (excluding `Unclustered` and themes under 100 reviews):

```sql
SELECT topic_id, label, count, pct_negative
FROM themes
WHERE topic_id != -1 AND count >= 100
ORDER BY pct_negative DESC
LIMIT 5;
```

| Topic ID | Theme Label | Review Count | Pct Negative |
|:---:|:---|:---:|:---:|
| **17** | Cables and TV accessories | 209 | 69.38% |
| **18** | Baby bottles and diapers | 209 | 68.90% |
| **9** | Computer hardware and storage | 504 | 68.45% |
| **7** | Batteries and chargers | 772 | 63.08% |
| **8** | Headphones and phones | 565 | 61.95% |

## 4. FastAPI Backend (Phase 9)
- **Endpoints**:
  - `GET /sentiment`: Overall sentiment metrics plus per-theme sentiment breakdowns.
  - `GET /themes`: List of all 29 themes plus `Unclustered` (-1).
  - `GET /summary/{theme_id}`: Grounded summary with praises, complaints, and citations (returns 404 for unknown or uncited themes).
  - `GET /search?q=`: Semantic vector search using `all-MiniLM-L6-v2` against `embeddings.npy` returning top 10 reviews (rejects empty `q` with HTTP 400).
- **No `/trends` Endpoint**:
  - The dataset contains no review timestamps or date metadata, so temporal aggregation is not possible and no `/trends` endpoint is provided.

## 5. Streamlit Dashboard (Phase 10)
- **Location**: `app/dashboard.py`
- **Architecture**: Consumes FastAPI endpoints over HTTP (no direct database access). API base URL is read from `API_BASE_URL` environment variable, defaulting to `http://127.0.0.1:8000`.
- **Launch Commands**:
  ```bash
  # Terminal 1: Start the FastAPI backend
  uvicorn src.api.main:app --reload

  # Terminal 2: Start the Streamlit dashboard
  streamlit run app/dashboard.py
  ```
- **Tabs**:
  1. **📈 Overview**: Total reviews (50,000), positive/negative counts, % negative metric cards. Horizontal bar chart of `pct_negative` per theme sorted descending (Unclustered excluded by default, toggleable via checkbox). Expandable full theme sentiment table. Info note: *"45% of reviews are unclustered; themes cover the rest."*
  2. **🔍 Theme Explorer**: Dropdown to select any theme. Metric cards for label, review count, % negative. Grounded summary with praises and complaints, each showing cited review IDs. Caution banner on themes 5, 20, and 28: *"Mixed theme: summary may blend product types."*
  3. **🔎 Search**: Free-text semantic search box. Top 10 results displayed with sentiment emoji (🟢/🔴), review ID, theme label, similarity score, and full review text.
- **Caching**: All API calls cached with `st.cache_data` (5-minute TTL for themes/sentiment, 1-minute TTL for search).
- **Error Handling**: API-down errors caught and displayed as a clear user-facing message with instructions, not a Python traceback.
- **Dependencies Added**: `streamlit==1.66.1`, `requests==2.34.2` (pinned in `requirements.txt`).

