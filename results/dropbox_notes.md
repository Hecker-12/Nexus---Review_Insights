# Drop-Your-Data Feature Notes and Acceptance Verification

This document records the acceptance verification, performance benchmarks, architectural limits, and evaluation notes for the **Drop-your-data** feature (FastAPI `/dropbox` endpoints and Streamlit dashboard tab).

---

## 1. Test Suite (36 Tests by File)

All 36 tests passed in 28.93 s on `feature/dropbox` (commit `4290969`):

### (a) `tests/dropbox/test_dropbox.py` (12 unit tests)
1. `test_pii_8_examples`: Validates regex masking of 8 diverse email, phone, and URL patterns with `<EMAIL>`, `<PHONE>`, and `<URL>`.
2. `test_loader_empty_file`: Rejects 0-byte input with `EmptyFileError`.
3. `test_loader_oversized_file`: Rejects files exceeding 5 MB with `FileSizeLimitError`.
4. `test_loader_invalid_file_type`: Rejects unsupported file extensions (e.g., `.json`) with `InvalidFileTypeError`.
5. `test_loader_valid_csv_and_text_detection`: Correctly identifies `review_text` and `score` columns in CSV.
6. `test_loader_valid_xlsx`: Correctly parses `.xlsx` and identifies `customer_feedback` and `rating` columns.
7. `test_loader_override_column`: Respects user-specified text column override and raises `ColumnNotFoundError` for missing columns.
8. `test_loader_sampling_at_3000`: Randomly samples CSV files with >3,000 rows down to exactly 3,000 rows (`random_state=42`).
9. `test_cleaning_drops`: Drops empty rows, reviews under 3 words, and exact duplicates.
10. `test_scorer_classes_and_sentences`: Verifies binary classes (`negative`, `positive`) and scoring logic on prototype inputs.
11. `test_themes_count_and_matching`: Verifies 29 unit-normalized centroids and matches coffee machine query to `Coffee makers`.
12. `test_pipeline_synthetic_200_rows`: Validates end-to-end `analyze()` pipeline on 200 synthetic rows with PII masking and rating evaluation.

### (b) `tests/test_api.py` (7 core API tests)
1. `test_sentiment_endpoint`: Verifies `/sentiment` response schema, 50,000 total reviews, and 30 theme groups.
2. `test_themes_endpoint`: Verifies `/themes` returns all 29 themes plus Unclustered (-1).
3. `test_summary_valid_endpoint`: Verifies grounded summaries for Theme 4 (Coffee makers), 26, and 5.
4. `test_summary_404_cases`: Confirms HTTP 404 for non-existent theme 999 and Unclustered (-1).
5. `test_search_valid`: Tests semantic search with `all-MiniLM-L6-v2` cosine similarity.
6. `test_search_returns_exactly_10_sorted_descending`: Verifies top-10 descending similarity ordering.
7. `test_search_empty_query_cases`: Verifies HTTP 400 for empty or whitespace-only search queries.

### (c) `tests/test_dashboard.py` (2 Streamlit AppTest tests)
1. `test_dashboard_tabs_and_overview_metrics`: Verifies all 4 dashboard tabs load with 0 exceptions, displays `50,000` / `25,331` / `24,669`, renders file uploader, and shows Amazon review domain warnings.
2. `test_dashboard_results_rendering_and_session_state`: Pre-sets `st.session_state["dropbox_results"]` and verifies KPI cards, accuracy display, theme breakdown chart, confidence annotations, and "Download Scored CSV" button.

### (d) `tests/test_dropbox_api.py` (15 integration tests)
1. `test_dropbox_limits`: Verifies `GET /dropbox/limits` returns correct limits.
2. `test_dropbox_valid_csv`: Verifies `POST /dropbox/analyze` with valid CSV returns 200 and complete schema.
3. `test_dropbox_valid_xlsx`: Verifies `POST /dropbox/analyze` with valid Excel spreadsheet.
4. `test_dropbox_rating_column_and_3star_ignored`: Verifies 1-2 mapped to negative, 4-5 mapped to positive, and 3-star neutral reviews ignored.
5. `test_dropbox_no_rating_column`: Verifies `accuracy_vs_ratings` is `None` when no ratings column is present.
6. `test_dropbox_bad_extension`: Verifies unsupported extensions return HTTP 415.
7. `test_dropbox_empty_file`: Verifies empty files return HTTP 400.
8. `test_dropbox_oversized_file`: Verifies files over 5 MB return HTTP 413.
9. `test_dropbox_no_text_column`: Verifies numeric-only tables return HTTP 400.
10. `test_dropbox_text_column_override`: Verifies explicit `text_column` form field overrides auto-detection.
11. `test_dropbox_pii_scrubbing_in_returned_rows`: Verifies PII redaction in returned rows.
12. `test_dropbox_formula_injection_guard`: Verifies prefixing `'` for `=, +, -, @, \t`.
13. `test_dropbox_deterministic`: Verifies identical outputs across repeated requests.
14. `test_dropbox_500_no_traceback`: Verifies unexpected internal exceptions return HTTP 500 without tracebacks.
15. `test_dropbox_no_repo_files_written`: Verifies no temporary or result files are leaked into the git tree.

---

## 2. Cold-Start Timings

The drop-your-data pipeline relies on lazy, thread-safe model initialization upon the first request to `/dropbox/analyze`.

- **Measured Step 2A Timings (live server start)**:
  - First `POST /dropbox/analyze` (300-row sample from `test.parquet`, cold start): **10.17 s**
  - Second `POST /dropbox/analyze` (identical 300-row sample, warm call): **6.06 s**
- **Component Breakdown from Earlier Profiling**:
  - TF-IDF vectorizer + Logistic Regression loading: **15.7 s**
  - Theme centroid loading & unit normalization (29 centroids): **0.09 s**
  - Embedding model initialization (`all-MiniLM-L6-v2`): **7.2 s**
  - First 300-row analysis: **10.2 s**

---

## 3. Limits and Constraints

The endpoint enforces strict resource boundaries:
- **Maximum file size**: **5.0 MB** (`5 * 1024 * 1024` bytes). Oversized files return HTTP 413.
- **Maximum row cap**: **3,000 rows**.
  - CSV files with >3,000 rows are randomly sampled to 3,000 rows with fixed seed `random_state=42`.
  - Excel (`.xlsx`) files with >3,000 rows are rejected before loading with HTTP 400 (`SheetLimitError`).
- **Maximum column cap**: **50 columns**. Sheets exceeding 50 columns are rejected with HTTP 400.
- **Supported extensions**: `.csv` and `.xlsx` only. All other file extensions return HTTP 415.

---

## 4. Acceptance Check Results (Steps 2B, 2C, 2D, 2E, 2G)

### Step 2B: Held-Out Accuracy Check (from `data/processed/test.parquet`)
- **300 rows** (`random_state=3`, mapped `positive -> 5`, `negative -> 1`):
  - HTTP Status: **200**
  - Rows Used: **300** (pass requirement: $\ge 290$)
  - Accuracy: **0.8900** (0.890, pass requirement: between 0.86 and 0.94)
  - Macro F1: **0.8887** (0.889)
  - Theme assigned share: **87.0%** (261 / 300 assigned, 39 Unassigned)
- **1,000 rows** (`random_state=4`, mapped `positive -> 5`, `negative -> 1`):
  - HTTP Status: **200**
  - Rows Used: **1,000**
  - Accuracy: **0.8870** (0.887)
  - Macro F1: **0.8867** (0.887)

*Methodological note*: The 40-row synthetic test from early unit testing (accuracy 1.0) and the 87.8% in-sample theme assignment rate from earlier development checks were NOT evaluations; they were sanity checks. The held-out checks above on 300 and 1,000 rows from `test.parquet` (not used for model training) represent the actual automatic accuracy checks.

### Step 2C: Not-Amazon Text Check (40 Short Non-Product Texts)
Evaluated on 40 generated short texts (tweets, support messages, news comments) of mixed sentiment:
- HTTP Status: **200** (no crash)
- Both warnings present: **True**
  - *"Models were trained on Amazon product reviews; accuracy on other text may be lower."*
  - *"Themes are the 29 Amazon-product themes found in this project; they may not fit your data."*
- Sentiment split: **29 positive (72.5%)**, **11 negative (27.5%)**
- Share "Unassigned": **80.0%** (32 / 40)
- Share with probability $< 0.65$: **67.5%** (27 / 40)
- **5 Example Rows**:
  1. Row 0: `[NEGATIVE 0.5835]` Theme: `Unassigned` | *"traffic on the interstate is completely standstill today, absolutely furious with road work"*
  2. Row 1: `[POSITIVE 0.7979]` Theme: `Unassigned` | *"loving the crisp autumn morning air and gorgeous colorful trees today!"*
  3. Row 2: `[NEGATIVE 0.5930]` Theme: `Batteries and chargers` | *"customer support at my local electric utility resolved my billing question in five minutes"*
  4. Row 3: `[POSITIVE 0.7614]` Theme: `Unassigned` | *"the new commuter train schedule makes my morning commute twice as long and stressful"*
  5. Row 4: `[POSITIVE 0.6320]` Theme: `Unassigned` | *"congratulations to our neighborhood soccer team on advancing to the state championship!"*

### Step 2D: Error Handling Checks
- **1. `.txt` file**: Status **415**, Detail: `Unsupported file format '.txt'. Only .csv and .xlsx files are supported.`
- **2. Empty CSV**: Status **400**, Detail: `Uploaded file is empty (0 bytes).`
- **3. Numeric-only CSV**: Status **400**, Detail: `Could not automatically detect a text column. Please specify a text column name.`
- **4. 6 MB file**: Status **413**, Detail: `File size exceeds maximum allowed limit of 5.00 MB.`
- **5. XLSX with 3,500 rows**: Status **400**, Detail: `Excel file exceeds maximum allowed limit of 3000 data rows.`
- *Traceback check*: **PASS** (no response body contains the word "Traceback").

### Step 2E: Privacy and Formula Guard
- **PII Redaction**: Raw text with email (`confidential_agent_test987@securemail.org`), phone (`800-555-0199`), and URL (`https://classifiedportal.testnet.internal/auth`) returned with `<EMAIL>`, `<PHONE>`, `<URL>`. None of the raw strings appeared in returned payload.
- **Formula Injection Guard**: Raw text starting with `=1+1` returned as `'=1+1`.
- **Repository Cleanliness**: Git status showed no untracked data files or temp files. No uploaded text leaked into any repository `.log` file.

### Step 2G: Snapshot Parity Lines
Comparing live endpoints against `scratch/api_snapshot.json`:
- `/sentiment`: **True**
- `/themes`: **True**
- `/summary/4`: **True**
- `/summary/26`: **True**

