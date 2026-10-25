import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance within the app's lifespan context."""
    with TestClient(app) as c:
        yield c


def test_sentiment_endpoint(client):
    """Test GET /sentiment returns overall metrics and all theme distributions."""
    response = client.get("/sentiment")
    assert response.status_code == 200
    data = response.json()

    # Check overall
    assert "overall" in data
    overall = data["overall"]
    assert overall["total_reviews"] == 50000
    assert overall["positive_count"] + overall["negative_count"] == 50000
    assert 0.0 <= overall["pct_negative"] <= 100.0

    # Check themes
    assert "themes" in data
    themes = data["themes"]
    assert len(themes) == 30  # 29 real themes + topic -1 (Unclustered)

    topic_ids = [t["topic_id"] for t in themes]
    assert -1 in topic_ids
    assert 0 in topic_ids
    assert 28 in topic_ids


def test_themes_endpoint(client):
    """Test GET /themes returns all 29 themes plus Unclustered (-1)."""
    response = client.get("/themes")
    assert response.status_code == 200
    themes = response.json()

    assert len(themes) == 30
    unclustered = next((t for t in themes if t["topic_id"] == -1), None)
    assert unclustered is not None
    assert unclustered["label"] == "Unclustered"
    assert unclustered["count"] == 22623

    coffee = next((t for t in themes if t["topic_id"] == 4), None)
    assert coffee is not None
    assert coffee["label"] == "Coffee makers"
    assert coffee["count"] == 1193


def test_summary_valid_endpoint(client):
    """Test GET /summary/{theme_id} for valid themes."""
    # Test theme 4 (Coffee makers)
    response = client.get("/summary/4")
    assert response.status_code == 200
    data = response.json()
    assert data["theme_id"] == 4
    assert data["theme_label"] == "Coffee makers"
    assert "summary" in data
    assert len(data["praises"]) > 0
    assert len(data["complaints"]) > 0

    # Test theme 26 (GPS navigation devices)
    res_26 = client.get("/summary/26")
    assert res_26.status_code == 200
    assert res_26.json()["theme_id"] == 26

    # Test theme 5 (Supplements, food and personal care)
    res_5 = client.get("/summary/5")
    assert res_5.status_code == 200
    assert res_5.json()["theme_id"] == 5


def test_summary_404_cases(client):
    """Test GET /summary/{theme_id} returns 404 for unknown or unsummarized themes."""
    # Non-existent theme ID
    res_unknown = client.get("/summary/999")
    assert res_unknown.status_code == 404
    assert "not found" in res_unknown.json()["detail"].lower()

    # Topic -1 (Unclustered is uncited/unsummarized by design)
    res_unclustered = client.get("/summary/-1")
    assert res_unclustered.status_code == 404
    assert "not found" in res_unclustered.json()["detail"].lower()


def test_search_valid(client):
    """Test GET /search?q=... returns top 10 most relevant reviews with scores."""
    response = client.get("/search?q=coffee+maker+leaking+water")
    assert response.status_code == 200
    results = response.json()

    assert len(results) == 10
    for r in results:
        assert "review_id" in r
        assert "text" in r
        assert len(r["text"]) > 0
        assert r["sentiment_true"] in ["positive", "negative"]
        assert "topic_label" in r
        assert isinstance(r["similarity_score"], float)

    # Verify descending similarity score order
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_exactly_10_sorted_descending(client):
    """Test that search returns exactly 10 results with strictly descending similarity scores."""
    response = client.get("/search?q=battery+died+quickly")
    assert response.status_code == 200
    results = response.json()

    # Exactly 10 results
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    # Each result has a similarity_score
    scores = [r["similarity_score"] for r in results]
    assert all(isinstance(s, float) for s in scores)

    # Strictly descending order (no ties at same rank)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score at index {i} ({scores[i]}) is not >= score at index {i+1} ({scores[i+1]})"
        )

    # Overall: sorted descending
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query_cases(client):
    """Test GET /search rejects empty or whitespace-only queries with HTTP 400."""
    # Empty query parameter
    res1 = client.get("/search?q=")
    assert res1.status_code == 400
    assert "cannot be empty" in res1.json()["detail"]

    # Whitespace-only query
    res2 = client.get("/search?q=   ")
    assert res2.status_code == 400
    assert "cannot be empty" in res2.json()["detail"]

    # Missing query parameter
    res3 = client.get("/search")
    assert res3.status_code == 400
    assert "cannot be empty" in res3.json()["detail"]
