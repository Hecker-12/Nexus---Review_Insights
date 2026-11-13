"""
Tests for v1.2 Streamlit dashboard tabs and components using AppTest.
Verifies all 6 tabs load with 0 exceptions, Overview retains core metrics,
Key Insights card renders, and Methodology and Complaints tabs render.
"""

import os
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


def test_dashboard_v12_all_tabs_and_components():
    """Verify all 6 tabs load with 0 exceptions, Overview metrics and Key Insights

    render correctly, and Methodology and Complaints tabs render without errors.
    """
    dashboard_path = os.path.abspath("app/dashboard.py")
    with patch("requests.get", side_effect=mock_requests_get), \
         patch("requests.post", side_effect=mock_requests_post):
        at = AppTest.from_file(dashboard_path, default_timeout=30)
        at.run()

        # 1. 0 exceptions
        assert len(at.exception) == 0, f"Dashboard threw exceptions: {[e.value for e in at.exception]}"

        # 2. Exactly 6 tabs in expected order
        tab_labels = [t.label for t in at.tabs]
        assert len(tab_labels) == 6, f"Expected 6 tabs, found {len(tab_labels)}: {tab_labels}"
        assert tab_labels[0] == "📈 Overview"
        assert tab_labels[1] == "🔍 Theme Explorer"
        assert tab_labels[2] == "🔎 Search"
        assert tab_labels[3] == "Drop your data"
        assert tab_labels[4] == "Methodology"
        assert tab_labels[5] == "Complaints (exploratory)"

        # 3. Overview metrics: 50,000 / 25,331 / 24,669
        metric_values = [m.value for m in at.metric]
        assert "50,000" in metric_values, f"Expected '50,000' in metrics, got {metric_values}"
        assert "25,331" in metric_values, f"Expected '25,331' in metrics, got {metric_values}"
        assert "24,669" in metric_values, f"Expected '24,669' in metrics, got {metric_values}"

        # 4. Key Insights card rendered
        all_markdown = " ".join(m.value for m in at.markdown)
        all_subheaders = " ".join(s.value for s in at.subheader)
        assert "Key Insights" in all_subheaders, "Missing 'Key Insights' subheader"
        assert "Overall Negative Sentiment" in all_markdown, "Missing overall negative sentiment bullet"

        # 5. Methodology tab rendered
        assert "Methodology & Evaluation Rigor" in all_subheaders, "Missing Methodology subheader"
        assert "Model Performance" in all_markdown, "Missing Model Performance section in Methodology"
        assert "McNemar Exact Tests" in all_markdown, "Missing McNemar section in Methodology"

        # 6. Complaints tab rendered
        assert "Complaint Taxonomy Analysis" in all_subheaders, "Missing Complaints subheader"
        assert "Overall Distribution by Complaint Type" in all_markdown, "Missing Complaint distribution section"

