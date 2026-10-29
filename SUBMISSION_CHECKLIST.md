# Submission Checklist — Nexus Customer Review Insights Engine

This checklist outlines the contents, execution instructions, data regeneration steps, and quality control auditing for the project submission.

---

## 1. Submission Package Contents

The repository and submission archive contain the following core components:

- **`README.md`**: Complete system architecture, evaluation results, domain warnings, Drop-your-data feature guide, screenshots, and run documentation.
- **`SUBMISSION_CHECKLIST.md`**: Submission verification, execution instructions, and component auditing (this document).
- **`requirements.txt`**: Pinned Python dependencies for reproducible execution with Python 3.11.
- **`Dockerfile` & `Dockerfile.dashboard`**: Container specifications for FastAPI backend and Streamlit dashboard.
- **`docker-compose.yml`**: Multi-container orchestration definition for API and Dashboard services.
- **`.dockerignore` & `.gitignore`**: Git and Docker ignore configurations excluding datasets, models, temporary files, and environments.
- **`src/`**: Core application logic:
  - `src/api/main.py`: FastAPI backend implementing `/sentiment`, `/themes`, `/summary/{id}`, and `/search`.
  - `src/api/dropbox_router.py`: FastAPI endpoints for `GET /dropbox/limits` and `POST /dropbox/analyze`.
  - `src/dropbox/`: Complete Drop-your-data pipeline (loader, cleaning, regex PII redaction, TF-IDF sentiment scoring, centroid-based theme matcher, error models, and pipeline runner).
  - `src/cleaning.py`, `src/complaints.py`, `src/complaint_types.py`, `src/db.py`, `src/llm.py`, `src/sentiment.py`, `src/summaries.py`, `src/topics.py`: Data ingestion, database population, and modeling utilities.
- **`app/`**:
  - `app/dashboard.py`: Interactive Streamlit dashboard with 4 tabs: Overview, Theme Explorer, Search, and Drop your data.
- **`tests/`**: Pytest test suite (36 passing tests):
  - `tests/test_api.py`: 7 integration tests for core REST endpoints.
  - `tests/test_dashboard.py`: 2 Streamlit `AppTest` test cases.
  - `tests/test_dropbox_api.py`: 15 integration tests for `/dropbox` endpoints.
  - `tests/dropbox/test_dropbox.py`: 12 unit tests for the dropbox package.
- **`scripts/`**:
  - `scripts/acceptance_dropbox.py`: Automated end-to-end acceptance test script covering cold-start, held-out accuracy, non-Amazon text, error handling, PII/formula guard, AppTest, and snapshot parity.
- **`results/`**: Analytical outputs, documentation, and visual evidence:
  - `results/dropbox_notes.md`: Profiling timings, architectural constraints, acceptance check outputs, and snapshot lines.
  - `results/notes.md`: Detailed methodological documentation.
  - `results/verification_report.md`: Grounded summary citation audit and quality assessment.
  - `results/sentiment_comparison.csv` & `sentiment_comparison.png`: Baseline and fine-tuned sentiment evaluations.
  - `results/topic_comparison.csv`: LDA and BERTopic coherence metrics.
  - `results/complaint_by_theme.csv`, `complaint_by_theme_v2.csv`, `complaint_type_counts.csv`, `complaint_type_counts_v2.csv`, `complaint_topic_summary.csv`: Complaint taxonomy outputs.
  - `results/screenshots/`: Full-page dashboard captures (`01_overview.png`, `02_theme_explorer.png`, `03_mixed_theme_warning.png`, `04_search_results.png`, `05_drop_your_data_empty.png`, `06_drop_your_data_results.png`).
- **`config/`**:
  - `config/complaint_types.json` and `config/complaint_types_v2.json`: Taxonomy definitions and prototype sentences.
- **`notebooks/`**: Research and development notebooks (`01_explore.ipynb` through `10_complaint_types.ipynb`).

---

## 2. How to Run

### (a) Python Virtual Environment Setup

The system requires **Python 3.11**:

```bash
# Clone and enter the repository
git clone https://github.com/Hecker-12/Nexus---Review_Insights.git
cd Nexus---Review_Insights

# Create virtual environment
python -m venv .venv311

# Activate virtual environment
# Windows:
.venv311\Scripts\activate
# Linux/macOS:
source .venv311/bin/activate

# Install all pinned dependencies
pip install -r requirements.txt
```

### (b) Running the FastAPI Backend and Streamlit Dashboard

```bash
# Terminal 1: Launch FastAPI REST backend (port 8000)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Streamlit Dashboard (port 8501)
streamlit run app/dashboard.py
```

- Dashboard UI: `http://localhost:8501`
- Interactive API Documentation: `http://localhost:8000/docs`

### (c) Running Tests

```bash
# Run the full 36-test suite
pytest tests/ -v

# Run the automated acceptance test script
python scripts/acceptance_dropbox.py
```

### (d) Running with Docker Compose

```bash
docker compose up
```

> **Note on Docker Images**: Docker images predate the Drop-your-data feature and were not rebuilt.

---

## 3. Data & Model Exclusions and Regeneration Order

The `data/` and `models/` directories are intentionally excluded from version control to prevent committing large binary artifacts (unit embeddings, model checkpoints, and SQLite databases).

To regenerate all processed data and model artifacts from scratch:

1. **`notebooks/01_cleaning.ipynb`**: Cleans and normalizes raw reviews, creating `data/processed/reviews_clean.parquet`.
2. **`notebooks/02_sentiment.ipynb` & `03_distilbert_ft.ipynb`**: Evaluates sentiment baselines and fine-tunes DistilBERT, outputting `data/processed/test_preds.parquet`.
3. **`notebooks/04_embeddings.ipynb`**: Computes 384-dimensional unit-normalized embeddings via `all-MiniLM-L6-v2`, saving `data/processed/embeddings.npy`.
4. **`notebooks/05_lda.ipynb` & `06_bertopic.ipynb`**: Trains LDA and initial BERTopic v1/v2 models.
5. **`notebooks/08_bertopic.ipynb`**: Hierarchically merges micro-clusters into 29 themes, creating `data/processed/topics_v1_merged.parquet`.
6. **`src/summaries.py`**: Calls Gemini to generate citation-grounded theme summaries with automated validation, saving `data/processed/theme_summaries.json`. *(Requires `${GEMINI_API_KEY}` set in `.env`)*.
7. **`src/db.py`**: Builds SQLite database `data/processed/insights.db` with relational tables (`reviews`, `themes`, `summaries`).

---

## 4. Quality Control Audit: Who Checked What

| Component / Task | Automatic Checks | LLM Reviewer | Human (me) |
|:---|:---|:---|:---|
| **Sentiment Model Evaluation** | Accuracy, macro F1, confusion matrices on the held-out 10,000-review test split | Error sample inspections | Not done |
| **Grounded Summaries** | Citation parsing, word overlap validation (2 false alarms noted), 6 not checked | 18 claims reviewed (14 supported, 4 partly supported, 0 unsupported) | Not done |
| **Theme Labels** | None | Gemini drafted initial labels; Claude suggested 22 adjustments, I approved them | Approval only |
| **Topic Quality** | None | Spot-check of 5 thematic clusters | Not done |
| **Complaint Taxonomy** | Cosine similarity ($\ge 0.30$) and margin ($\ge 0.03$) threshold enforcement | Inspection of ~10 sample reviews per complaint type | Not done |
| **Core API Endpoints** | 7 automated tests in `tests/test_api.py` | None | Not done |
| **Drop-Your-Data Feature** | 36 passing tests + `scripts/acceptance_dropbox.py` (held-out accuracy: 89.00% on 300 rows, 88.70% on 1,000 rows) | None | Not done |
