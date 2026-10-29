# Nexus — Customer Review Insights Engine

At scale, thousands of customer reviews go unread, leaving critical product flaws, recurring failure modes, and customer delight signals buried in unstructured text. Product and customer experience teams cannot manually sift through tens of thousands of reviews to extract actionable intelligence. **Nexus** addresses this problem on a dataset of **50,000 Amazon product reviews** by providing an end-to-end analytical pipeline: fine-tuned transformer sentiment classification, outlier-aware topic discovery and reduction to **29 core themes**, hallucination-guarded grounded summaries that passed automatic checks, sub-10ms semantic vector search, relational SQLite storage, and an interactive FastAPI + Streamlit dashboard.

---

## 1. Pipeline Architecture

```mermaid
flowchart TD
    subgraph Ingestion & Preprocessing
        RAW[50,000 Amazon Reviews] --> CLEAN[Text Cleaning & Normalization]
        CLEAN --> EMB[all-MiniLM-L6-v2 Embeddings]
    end

    subgraph Sentiment Classification
        CLEAN --> VADER[VADER Rule Baseline]
        CLEAN --> TFIDF[TF-IDF + Logistic Regression]
        CLEAN --> BERT_ZS[DistilBERT Zero-Shot]
        CLEAN --> BERT_FT[DistilBERT Fine-Tuned]
    end

    subgraph Topic Discovery
        EMB --> LDA[LDA Baseline k=10, 15]
        EMB --> BERTopic_v1[BERTopic v1: 109 Topics + Outliers]
        BERTopic_v1 --> MERGED[Hierarchical Merge: 29 Themes + Unclustered]
    end

    subgraph LLM Intelligence
        MERGED --> LLM_LABEL[LLM Noun-Phrase Theme Labeling]
        MERGED --> LLM_SUM[Grounded Theme Summaries: Praises & Complaints]
        LLM_SUM --> AUTO_CHECK[Automated Checks: Citations & Content Overlap]
    end

    subgraph Storage & Serving
        MERGED & BERT_FT & LLM_SUM --> SQLITE[(insights.db SQLite)]
        EMB & SQLITE --> FASTAPI[FastAPI REST API: /sentiment, /themes, /summary, /search]
        FASTAPI --> STREAMLIT[Streamlit Dashboard: Overview, Theme Explorer, Search]
    end
```

---

## 2. Dataset & Limitations

The engine analyzes **50,000 Amazon product reviews** with the following structural limitations:
- **Binary Sentiment Only**: Reviews are categorized strictly as `positive` or `negative`. There is no neutral class to capture hedged or ambivalent reviews.
- **No Dates (No Trends)**: The dataset contains no timestamp or date metadata. Longitudinal trend tracking, time-series aggregations, and temporal spike detection cannot be performed.
- **Star-Derived Label Noise**: Ground truth labels derived from 1–5 star ratings introduce noise (e.g. sarcastic reviews with 5 stars or mildly critical reviews rated 1 star).
- **Mixed Products**: Reviews span diverse product categories (electronics, media, baby products, apparel, personal care) without category partition metadata, leading to composite clusters.

---

## 3. Results

All metrics below are drawn directly from `results/sentiment_comparison.csv`, `results/topic_comparison.csv`, `results/notes.md`, and `results/verification_report.md`.

### (a) Sentiment Classification (10,000 Test Reviews)

| Model | Accuracy | Macro F1 | True Negative | False Positive | False Negative | True Positive |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **VADER default** | 0.7189 | 0.7041 | 2,477 | 2,457 | 354 | 4,712 |
| **VADER tuned (0.7)** | 0.7707 | 0.7706 | 3,727 | 1,207 | 1,086 | 3,980 |
| **TF-IDF + LogReg** | 0.9017 | 0.9017 | 4,426 | 508 | 475 | 4,591 |
| **DistilBERT zero-shot** | 0.8980 | 0.8980 | 4,506 | 428 | 592 | 4,474 |
| **DistilBERT fine-tuned** | **0.9468** | **0.9468** | **4,661** | **273** | **259** | **4,807** |

### (b) Topic Modeling & NPMI Coherence

| Method | Topics | Outliers | NPMI |
|:---|:---:|:---:|:---:|
| **LDA (k=10)** | 10 | 0.0% | 0.0420 |
| **LDA (k=15)** | 15 | 0.0% | 0.0305 |
| **BERTopic v1** | 109 | 45.2% | 0.0983 |
| **BERTopic v2** | 248 | 46.7% | 0.0593 |
| **BERTopic v1 (merged 29 themes)** | 29 | 45.2% | 0.0401 |

> **Note on NPMI**: *Coherence depends on topic count and outlier handling; BERTopic NPMI is computed on clustered topics only, excluding the outlier group.*

### (c) Summary & Claim Evaluation

| Metric | Count | Details / Methodological Status |
|:---|:---:|:---|
| **Total claims evaluated** | 168 | Across all 29 themes |
| **Claims not checked** | 6 | Theme 24 (Spanish-language reviews, skipped for English word-overlap) |
| **Out-of-sample citations dropped** | 1 | Review ID `2706` in Theme 2 |
| **Flagged claims (automatic checks)** | 2 | Word-overlap false alarms in Theme 14 and Theme 28 (passed automatic checks upon review) |
| **Claims read by an LLM reviewer** | 18 | 14 supported, 4 partly supported, 0 unsupported |
| **Overview summaries** | 29 | Uncited by design, reported separately from praise and complaint claims |

---

## 4. Additional Exploratory Analysis: What Negative Reviews Are About

*(Exploratory, not part of the evaluated pipeline)*

An exploratory investigation was conducted on the subset of **24,669 negative reviews** to determine whether complaint-specific failure modes could be isolated.

### (a) BERTopic on Negative Reviews
Fitting BERTopic directly on negative reviews using precomputed `all-MiniLM-L6-v2` embeddings clustered primarily by product category rather than complaint type:
- **Unmerged (92 topics, 33.38% outliers)**: NPMI **0.0568**
- **Merged (14 themes, 33.38% outliers)**: NPMI **-0.0015**

Because the merged model showed negative coherence and remained split along product lines (e.g., books, movies, audio equipment) rather than failure modes, those 14 themes were not labeled or summarized.

### (b) Predefined Complaint-Type Taxonomy
To categorize complaints across product boundaries, a predefined 11-type taxonomy was constructed with customer-phrased prototype sentences (`config/complaint_types_v2.json`) and embedded using `all-MiniLM-L6-v2`. Using centroid cosine similarity with conservative thresholds ($\text{similarity} \ge 0.30$, $\text{margin} \ge 0.03$), **52.29%** (12,899 / 24,669) of negative reviews matched a complaint type:
- **Disliked the content (plot, writing, acting, music)**: **35.77%** (8,825 reviews), capturing creative dissatisfaction in media categories (books, movies, music).
- **Product-problem types**: **15.50%** (**3,823 reviews**) matched concrete product failure categories:
  - *Poor quality or broke easily*: 764 (3.10%)
  - *Stopped working or defective*: 675 (2.74%)
  - *Wrong item or not as described or pictured*: 422 (1.71%)
  - *Delivery problem: late, never arrived, damaged in shipping*: 417 (1.69%)
  - *Bad taste, smell or skin reaction*: 411 (1.67%)
  - *Hard to install, set up, or not compatible*: 384 (1.56%)
  - *Leaks or spills*: 340 (1.38%)
  - *Poor fit or wrong size*: 205 (0.83%)
  - *Poor customer service, refund or return problems*: 205 (0.83%)
- **Unreliable noise type**: The type *Too loud or noisy* matched 251 reviews (1.02%); its 251 matches (1.02%) are unreliable (about 3 to 4 of 10 sampled looked correct to an LLM reviewer) and remain in the saved v2 file. (Including this type would sum to 4,074 reviews or 16.51%, but **15.5%** is the headline product-problem figure).
- **Other / unclear**: **47.71%** (11,770 reviews) fell below the similarity or margin threshold.

### (c) Exploratory Methodological Limits
- The taxonomy is predefined rather than unsupervised.
- Prototypes were revised once after viewing samples, so sample-based observations are not independent.
- Qualitative assessments relied on a small LLM-reviewed sample (about 10 reviews per type).
- The noise type: its 251 matches (1.02%) are unreliable (about 3 to 4 of 10 sampled looked correct to an LLM reviewer) and remain in the saved v2 file.
- The shares are lower bounds because 47.7% (11,770 reviews) remain "Other / unclear".

---

## 5. Key Findings

1. **Fine-Tuning Gain**: Fine-tuning DistilBERT yielded **94.68% accuracy** and **0.9468 Macro F1**, outperforming the zero-shot baseline (89.80%) by **+4.88%** and cutting false positives from 428 to 273.
2. **Domain Shift & Linear Baseline**: TF-IDF + Logistic Regression achieved **90.17% accuracy**, proving to be a strong baseline on in-domain Amazon review vocabulary compared to tuned VADER (77.07%).
3. **Coherence vs Human Judgment**: While BERTopic v1 achieved higher NPMI (0.0983) with 109 micro-clusters, 109 topics is overwhelming for decision-makers. Hierarchically merging to 29 themes (NPMI 0.0401) created human-interpretable product themes.
4. **Outlier Reassignment Noise**: Reassigning the 45.2% HDBSCAN outliers via cosine similarity introduced noise; spot-checks revealed clear thematic alignment in only 4 of 8 examined cases.
5. **LLM Labels Needing Correction**: Raw cluster labels required manual refinement: 22 of 29 labels were adjusted, with about 10 fixing real miscategorizations where raw keywords were overly generic or skewed by dominant words.
6. **Themes with Highest Negative Sentiment**: Querying `insights.db` for themes with $\ge 100$ reviews (excluding Unclustered) identified the top 5 pain points:
   - **Cables and TV accessories** (Theme 17): **69.38% negative** (209 reviews)
   - **Baby bottles and diapers** (Theme 18): **68.90% negative** (209 reviews)
   - **Computer hardware and storage** (Theme 9): **68.45% negative** (504 reviews)
   - **Batteries and chargers** (Theme 7): **63.08% negative** (772 reviews)
   - **Headphones and phones** (Theme 8): **61.95% negative** (565 reviews)

---

## 6. Error Analysis

In error analysis across the 10,000-review test split:
- **Both Models Wrong**: 12 of 15 examined reviews where both TF-IDF and fine-tuned DistilBERT failed were determined by an LLM reviewer to be mislabeled in the ground truth or genuinely unresolvable due to conflicting sentiment.
  - *Example (Review 16741)*: Sarcastic review rating 5 stars ("The only thing it gave me was stage 4 metastatic cancer. Overall 5 stars..."), true label `positive`, predicted `negative` by both models.
- **Fine-Tuned Model Failure Modes**: The fine-tuned transformer struggled on:
  1. *Hedged Sentiment*: Reviews containing both strong praise and severe criticism in different paragraphs (e.g. "great pots, lousy lids").
  2. *Non-English Reviews*: French reviews (e.g. Review 17881 "PAS DE VERSION STF: MALGRE VOS INDICATIONS...") misclassified due to English training distribution.

---

## 7. Topic Quality Assessment

Spot-checks conducted across merged themes revealed clear distinctions in topic coherence:
- **Coherent Themes (LLM-reviewed)**:
  - **Theme 0 (Books, 9,231 reviews)**: Highly cohesive book reviews citing reading experience, authors, and writing style.
  - **Theme 13 (Filters, pumps and cleaners, 344 reviews)**: Consistently groups water filters, vacuum cleaners, and humidifiers around mechanical durability.
- **Mixed Themes (LLM-reviewed)**:
  - **Theme 5 (Supplements, food and personal care, 907 reviews)**: Blends protein powders, herbal supplements (Kava Kava), and hair color kits.
  - **Theme 20 (Mixed: Gun cases and backpacks, 159 reviews)**: Blends rifle slings/holsters with laptop backpacks and camera cases.
  - **Theme 28 (Fans, 58 reviews)**: Blends room circulation fans, car fans, ceiling fans, and exhaust components.

---

## 8. Dashboard Screenshots

The interactive dashboard consists of three tabs and contextual safety banners:

| Overview Tab | Theme Explorer Tab |
|:---:|:---:|
| ![Overview Tab](results/screenshots/01_overview.png) | ![Theme Explorer](results/screenshots/02_theme_explorer.png) |
| *Metrics, unclustered notice, and % negative bar chart* | *Selected theme metrics, grounded summary, praises & complaints* |

| Mixed-Theme Caution Banner | Semantic Search Results |
|:---:|:---:|
| ![Mixed Theme Warning](results/screenshots/03_mixed_theme_warning.png) | ![Search Results](results/screenshots/04_search_results.png) |
| *Caution for composite themes (5, 20, 28)* | *Top 10 cosine-similarity results for "coffee maker leaking"* |

---

## 9. How to Run

### (a) Local Environment Setup

Requires Python 3.11:

```bash
# Clone and enter directory
git clone https://github.com/Hecker-12/Nexus---Review_Insights.git
cd Nexus---Review_Insights

# Create and activate virtual environment
python -m venv .venv311
# Windows:
.venv311\Scripts\activate
# Linux/macOS:
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### (b) Data Regeneration Order

Note: `data/` and `models/` are excluded from version control. To regenerate all processed artifacts:
1. `notebooks/01_cleaning.ipynb` — Clean and tokenize raw reviews to `data/processed/reviews_clean.parquet`.
2. `notebooks/02_sentiment.ipynb` & `03_distilbert_ft.ipynb` — Evaluate baselines and fine-tune DistilBERT, outputting `data/processed/test_preds.parquet`.
3. `notebooks/04_embeddings.ipynb` — Generate 384-dim unit-normalized `all-MiniLM-L6-v2` embeddings (`data/processed/embeddings.npy`).
4. `notebooks/05_lda.ipynb` & `06_bertopic.ipynb` — Train LDA baselines and BERTopic v1/v2 models.
5. `notebooks/08_bertopic.ipynb` — Execute hierarchical reduction to 29 themes (`data/processed/topics_v1_merged.parquet`).
6. `src/summaries.py` — Generate grounded summaries via Gemini, enforce automated validation checks, and save `data/processed/theme_summaries.json`.
7. `src/db.py` — Populate relational tables in `data/processed/insights.db`.

> **Environment Variables**: Set `${GEMINI_API_KEY}` and optionally `LLM_MODEL=gemini-1.5-flash` in `.env` before running LLM generation scripts.

### (c) Running API & Dashboard

```bash
# Terminal 1: Start FastAPI REST Backend
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Streamlit Dashboard
streamlit run app/dashboard.py
```

Access the dashboard at `http://localhost:8501` and interactive Swagger docs at `http://localhost:8000/docs`.

### (d) Running with Docker Compose

Docker files provided; build and run not re-tested at submission time.

---

## 10. Limitations & Future Roadmap

### Known Limitations
1. **Binary Classification**: Neutral or mixed customer sentiment is forced into binary buckets (`positive` / `negative`).
2. **Missing Dates**: The dataset lacks timestamp attributes, precluding velocity, trend, or seasonality tracking.
3. **45.2% Unclustered Reviews in Topic Discovery**: HDBSCAN leaves 22,623 reviews in topic `-1` to preserve cluster purity.
4. **Weakly Supervised Star Labels**: Ground truth labels carry natural reviewer bias and sarcasm.
5. **Predefined Complaint Taxonomy**: Zero-shot complaint matching leaves 47.71% of negative reviews unassigned in "Other / unclear".

### Future Work
- **Score-Your-Text Box**: An interactive tool to score arbitrary single reviews with calibrated sentiment probabilities.
- **Upload and Analyze**: Batch processing for uploaded CSV files to categorize sentiment and failure modes, with strict client-side data privacy.
- **Complaint Summaries**: Generating grounded, citation-backed summaries specifically for each complaint category.
- **Topics on Negative Reviews with a Better Method**: Exploring hierarchical clustering with adaptive density estimation or contrastive clustering to extract granular failure modes from negative reviews without product-category entanglement.
- **Dated Dataset for Trends**: Augmenting with timestamped e-commerce review streams to detect defect spikes and emerging failure trends over time.
- **Human Labeling of Summaries**: Conducting human review of generated claims and praise/complaint summaries alongside automated checks.

---

## 11. Verification & Quality Control: Who Checked What

| Component / Task | Automatic Checks | LLM Reviewer | Human (me) |
|:---|:---|:---|:---|
| **Sentiment Model Evaluation** | `pytest` test suite, confusion matrix, macro F1 calculations | None | Ran evaluation suite; TODO(me): inspect individual edge-case errors |
| **Grounded Summaries: Citations** | Parser validated citations against sample IDs; dropped 1 out-of-sample ID (`2706`) | LLM reviewer evaluated 18 sample claims (14 supported, 4 partly) | TODO(me): spot-check full claim texts against original reviews |
| **Grounded Summaries: Word Overlap** | Regex word-overlap checks against cited review text; flagged 2 false alarms | LLM reviewer checked flagged claims | TODO(me): review non-English Theme 24 claims |
| **Theme Cluster Labels** | Heuristic keyword overlap check | LLM generated candidate labels (22/29 adjusted) | TODO(me): finalize remaining composite theme names |
| **Sentiment Error Analysis** | Identification of 15 consensus failure cases | LLM reviewer assessed label noise vs model failure on 15 cases | TODO(me): verify ground-truth label noise in test set |
| **Complaint Prototypes & Samples** | Threshold filtering ($\text{sim} \ge 0.30$, $\text{margin} \ge 0.03$) | LLM reviewer evaluated small samples (~10 reviews per type); flagged noise type | TODO(me): conduct manual blind validation of complaint categories |
| **API Endpoints & Integration** | 7 passing FastAPI integration tests in `tests/test_api.py` | None | Tested endpoint responses; TODO(me): test under high concurrent load |

---

## 12. Methods & Transparency Note

- **LLM Used**: Grounded summaries were generated using `gemini-1.5-flash` (via the Gemini free tier) with temperature $0.2$, structured JSON schema enforcement, and stratified sample prompts.
- **LLM Reviewer**: Claim and topic quality reviews designated as "LLM reviewer" were conducted via automated Gemini evaluation prompts and have not been validated by a human.
- TODO(me): Conduct manual human review of sampled claims, misclassified reviews, and topic coherence.