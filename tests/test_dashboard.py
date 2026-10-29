"""
Tests for Streamlit dashboard tabs and components using AppTest.
"""

import os
import sys
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def mock_requests_get(url, *args, **kwargs):
    """Route requests.get to FastAPI TestClient."""
    path = url.replace("http://127.0.0.1:8000", "")
    params = kwargs.get("params")
    return client.get(path, params=params)


def mock_requests_post(url, *args, **kwargs):
    """Route requests.post to FastAPI TestClient."""
    path = url.replace("http://127.0.0.1:8000", "")
    data = kwargs.get("data")
    files = kwargs.get("files")
    return client.post(path, data=data, files=files)


def test_dashboard_tabs_and_overview_metrics():
    """Verify that all tabs load with 0 exceptions, Overview shows 50,000 / 25,331 / 24,669,

    and the new 'Drop your data' tab renders the uploader and warnings.
    """
    dashboard_path = os.path.abspath("app/dashboard.py")
    with patch("requests.get", side_effect=mock_requests_get), \
         patch("requests.post", side_effect=mock_requests_post):
        at = AppTest.from_file(dashboard_path, default_timeout=30)
        at.run()

        # 1. 0 exceptions
        assert len(at.exception) == 0, f"Dashboard threw exceptions: {[e.value for e in at.exception]}"

        # 2. Overview metrics: 50,000 / 25,331 / 24,669
        metric_values = [m.value for m in at.metric]
        assert "50,000" in metric_values, f"Expected '50,000' in metrics, got {metric_values}"
        assert "25,331" in metric_values, f"Expected '25,331' in metrics, got {metric_values}"
        assert "24,669" in metric_values, f"Expected '24,669' in metrics, got {metric_values}"

        # 3. New tab renders file uploader
        uploader_labels = [u.label for u in at.file_uploader]
        assert any("Upload CSV or Excel" in lbl for lbl in uploader_labels), (
            f"Expected uploader in {uploader_labels}"
        )

        # 4. New tab renders the warnings
        warning_texts = [w.value for w in at.warning]
        warning_combined = " ".join(warning_texts)
        assert "Amazon product reviews" in warning_combined, "Missing warning about Amazon reviews training"
        assert "29 Amazon-product themes" in warning_combined, "Missing warning about 29 themes"

        # 5. Check privacy note and limits text
        all_markdown = " ".join(m.value for m in at.markdown)
        all_captions = " ".join(c.value for c in at.caption)
        combined_text = all_markdown + " " + all_captions
        assert "Processed in memory by this app's API, not stored, no LLM used." in combined_text
        assert "CSV files over 3,000 rows are randomly sampled; Excel files over 3,000 rows are rejected." in combined_text


def test_dashboard_results_rendering_and_session_state():
    """Verify that when dropbox_results is in session_state, results (KPIs, accuracy,

    themes, preview with low confidence flag, and download button) render with 0 exceptions.
    """
    dashboard_path = os.path.abspath("app/dashboard.py")
    sample_result = {
        "report": {
            "rows_in": 10,
            "rows_used": 10,
            "removed_empty": 0,
            "removed_under_3_words": 0,
            "removed_duplicates": 0,
            "sampled": False,
            "detected_text_column": "review",
            "detected_rating_column": "rating",
            "timings": {"total_seconds": 0.5}
        },
        "summary": {
            "positive": 6,
            "negative": 4,
            "pct_positive": 60.0,
            "pct_negative": 40.0
        },
        "accuracy_vs_ratings": {
            "accuracy": 0.875,
            "macro_f1": 0.850,
            "evaluated_count": 8,
            "ignored_rating_3_count": 2,
            "note": "Ratings 1-2 treated as negative, 4-5 as positive; neutral rating 3 ignored."
        },
        "themes_table": [
            {"label": "Coffee makers", "count": 6, "share": 60.0, "pct_negative": 33.3},
            {"label": "Headphones and phones", "count": 4, "share": 40.0, "pct_negative": 50.0}
        ],
        "warnings": [
            "Models were trained on Amazon product reviews; accuracy on other text may be lower.",
            "Themes are the 29 Amazon-product themes found in this project; they may not fit your data."
        ],
        "preview": [
            {"row_id": 0, "scrubbed_text": "Great coffee maker", "label": "positive", "probability": 0.95, "theme": "Coffee makers"},
            {"row_id": 1, "scrubbed_text": "Average device", "label": "positive", "probability": 0.55, "theme": "Coffee makers"}
        ],
        "rows": [
            {"row_id": 0, "scrubbed_text": "Great coffee maker", "label": "positive", "probability": 0.95, "theme": "Coffee makers"}
        ]
    }

    with patch("requests.get", side_effect=mock_requests_get), \
         patch("requests.post", side_effect=mock_requests_post):
        at = AppTest.from_file(dashboard_path, default_timeout=30)
        at.session_state["dropbox_results"] = sample_result
        at.run()

        assert len(at.exception) == 0

        # Check KPIs rendered
        metric_values = [m.value for m in at.metric]
        assert "60.0%" in metric_values
        assert "40.0%" in metric_values
        assert "0.875" in metric_values

        # Check low confidence sentence
        all_info = " ".join(inf.value for inf in at.info)
        assert "The model has no neutral class; scores near 0.5 mean the text has no clear sentiment." in all_info

        # Check download button
        download_btns = [btn.label for btn in at.download_button]
        assert any("Download Scored CSV" in b for b in download_btns)
