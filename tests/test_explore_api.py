"""
Integration tests for explore endpoints:
- GET /review/{review_id}
- GET /similar/{review_id}?k=10
- GET /methodology
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_review_endpoint_valid():
    """Verify /review/{review_id} returns 200 and expected fields."""
    response = client.get("/review/0")
    assert response.status_code == 200
    data = response.json()
    assert data["review_id"] == 0
    assert isinstance(data["text"], str) and len(data["text"]) > 0
    assert data["sentiment_true"] in ["positive", "negative"]
    assert isinstance(data["theme_id"], int)
    assert isinstance(data["theme_label"], str) and len(data["theme_label"]) > 0


def test_review_endpoint_404_not_found():
    """Verify unknown review_id returns 404 with JSON detail."""
    response = client.get("/review/999999")
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "not found" in response.json()["detail"].lower()


def test_review_endpoint_validation_errors():
    """Verify negative and non-integer review_id return 422 without traceback."""
    # Negative ID
    r_neg = client.get("/review/-1")
    assert r_neg.status_code == 422
    assert "detail" in r_neg.json()
    assert "Traceback" not in r_neg.text

    # String ID
    r_str = client.get("/review/not_an_int")
    assert r_str.status_code == 422
    assert "detail" in r_str.json()
    assert "Traceback" not in r_str.text


def test_similar_endpoint_valid():
    """Verify /similar/{review_id} returns top-k neighbours sorted descending."""
    response = client.get("/similar/0?k=5")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 5

    # Check structure
    for item in items:
        assert isinstance(item["review_id"], int)
        assert isinstance(item["score"], float)
        assert isinstance(item["text"], str)
        assert len(item["text"]) <= 300
        assert item["sentiment_true"] in ["positive", "negative"]
        assert isinstance(item["theme_label"], str)

    # Check review itself is excluded
    assert all(item["review_id"] != 0 for item in items)

    # Check scores sorted descending
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)


def test_similar_endpoint_k_bounds():
    """Verify k out of range (k < 1 or k > 20) returns 422."""
    r_low = client.get("/similar/0?k=0")
    assert r_low.status_code == 422
    assert "Traceback" not in r_low.text

    r_high = client.get("/similar/0?k=25")
    assert r_high.status_code == 422
    assert "Traceback" not in r_high.text


def test_similar_endpoint_404():
    """Verify unknown review_id in /similar returns 404."""
    response = client.get("/similar/999999?k=5")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_methodology_endpoint():
    """Verify /methodology returns whitelisted tables without error."""
    response = client.get("/methodology")
    assert response.status_code == 200
    data = response.json()

    expected_keys = [
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

    for key in expected_keys:
        assert key in data, f"Missing key {key} in /methodology"
        table = data[key]
        assert table is not None, f"Table {key} should not be None"
        assert "columns" in table
        assert "rows" in table
        assert isinstance(table["columns"], list)
        assert isinstance(table["rows"], list)
        assert len(table["rows"]) > 0

    # Confirm no client path query accepted (endpoint takes no query/body parameter)
    # Testing that /methodology?file=../../etc returns 200 with the same whitelisted payload
    r_traversal = client.get("/methodology?file=../../etc/passwd")
    assert r_traversal.status_code == 200
    assert r_traversal.json() == data

