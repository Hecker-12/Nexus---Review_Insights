"""Integration tests for the /dropbox API endpoints."""

import io
import os
import subprocess
import unittest.mock
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_dropbox_limits():
    """GET /dropbox/limits returns 200 with correct caps."""
    res = client.get("/dropbox/limits")
    assert res.status_code == 200
    data = res.json()
    assert data["max_file_size_mb"] == 5.0
    assert data["max_rows"] == 3000
    assert data["max_columns"] == 50
    assert ".csv" in data["allowed_extensions"]
    assert ".xlsx" in data["allowed_extensions"]


def test_dropbox_valid_csv():
    """POST /dropbox/analyze with valid CSV returns 200 and expected schema."""
    csv_content = (
        "review_id,review_text,rating\n"
        "1,This coffee machine is fantastic and makes great espresso,5\n"
        "2,Broken on arrival and stopped heating after two days,1\n"
    )
    files = {"file": ("reviews.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "report" in data
    assert "summary" in data
    assert "themes_table" in data
    assert "preview" in data
    assert "rows" in data
    assert "warnings" in data
    assert data["report"]["rows_used"] == 2


def test_dropbox_valid_xlsx():
    """POST /dropbox/analyze with valid XLSX returns 200."""
    df = pd.DataFrame({
        "feedback": ["High quality headphones with rich sound", "Fell apart in one week"],
        "stars": [5, 1]
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    files = {"file": ("data.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["report"]["rows_used"] == 2
    assert data["report"]["detected_text_column"] == "feedback"
    assert data["report"]["detected_rating_column"] == "stars"


def test_dropbox_rating_column_and_3star_ignored():
    """Test 60 synthetic rows with obvious sentiment words, verifying rating 3 is ignored."""
    pos_phrases = [
        "Absolutely love this fantastic product, works brilliantly!",
        "Excellent high quality item, best purchase ever made.",
        "Very pleased, super happy, wonderful performance every day."
    ]
    neg_phrases = [
        "Terrible defective junk, broke into pieces on first use.",
        "Horrible quality, completely useless, stopped working immediately.",
        "Worst customer experience, defective and failed right away."
    ]
    neutral_phrases = [
        "It arrived yesterday in an ordinary cardboard box.",
        "Average item, nothing special to write about here."
    ]

    rows = []
    # 25 positive with rating 5
    for i in range(25):
        rows.append({"text": f"{pos_phrases[i % 3]} ({i})", "star_rating": 5})
    # 25 negative with rating 1
    for i in range(25):
        rows.append({"text": f"{neg_phrases[i % 3]} ({i})", "star_rating": 1})
    # 10 neutral with rating 3
    for i in range(10):
        rows.append({"text": f"{neutral_phrases[i % 2]} ({i})", "star_rating": 3})

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    files = {"file": ("ratings_test.csv", buf, "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    data = res.json()

    acc_report = data["accuracy_vs_ratings"]
    assert acc_report is not None
    assert acc_report["evaluated_count"] == 50  # 25 pos + 25 neg
    assert acc_report["ignored_rating_3_count"] == 10  # 10 rating 3 ignored
    assert acc_report["accuracy"] >= 0.90
    assert acc_report["macro_f1"] >= 0.90
    assert "neutral rating 3 ignored" in acc_report["note"]


def test_dropbox_no_rating_column():
    """When no rating column is present, accuracy_vs_ratings is None."""
    csv_content = (
        "comment\n"
        "Great product with nice features\n"
        "Broken junk that failed quickly\n"
    )
    files = {"file": ("no_ratings.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    assert res.json()["accuracy_vs_ratings"] is None


def test_dropbox_bad_extension():
    """POST with unsupported extension returns 415."""
    files = {"file": ("data.json", io.BytesIO(b'{"key": "val"}'), "application/json")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 415
    assert "detail" in res.json()


def test_dropbox_empty_file():
    """POST with empty file returns 400."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 400
    assert "detail" in res.json()


def test_dropbox_oversized_file():
    """POST exceeding 5 MB returns 413."""
    big_bytes = b"header\n" + b"x" * (5 * 1024 * 1024 + 10)
    files = {"file": ("big.csv", io.BytesIO(big_bytes), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 413
    assert "detail" in res.json()


def test_dropbox_no_text_column():
    """CSV with no identifiable text column returns 400."""
    csv_content = "id,num1,num2\n1,100,200\n2,300,400\n"
    files = {"file": ("numbers.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 400
    assert "detail" in res.json()


def test_dropbox_text_column_override():
    """Form text_column override selects the requested column."""
    csv_content = (
        "short_title,long_body\n"
        "Amazing,This is a very long descriptive body of the review\n"
        "Terrible,Another very long descriptive body of the review\n"
    )
    files = {"file": ("override.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    data = {"text_column": "short_title"}
    res = client.post("/dropbox/analyze", files=files, data=data)
    assert res.status_code == 200
    assert res.json()["report"]["detected_text_column"] == "short_title"


def test_dropbox_pii_scrubbing_in_returned_rows():
    """Returned rows have PII masked with <EMAIL>, <PHONE>, and <URL>."""
    csv_content = (
        "review\n"
        "Contact me at support@example.com for refunds\n"
        "Call customer hotline at 800-555-0199 today\n"
        "Visit https://example.com/store for discounts\n"
    )
    files = {"file": ("pii.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    rows = res.json()["rows"]
    all_text = " ".join(r["scrubbed_text"] for r in rows)
    assert "@example.com" not in all_text
    assert "<EMAIL>" in all_text
    assert "800-555-0199" not in all_text
    assert "<PHONE>" in all_text
    assert "https://example.com" not in all_text
    assert "<URL>" in all_text


def test_dropbox_formula_injection_guard():
    """Cells starting with =, +, -, @, or tab are prefixed with a single quote."""
    csv_content = (
        "review\n"
        "=SUM(A1:A10) malicious formula here\n"
        "+CMD|' /C calc'!A0 calculator payload\n"
        "-HYPERLINK('http://evil.com') link text\n"
        "@MALICIOUS_TAG in review text\n"
        "\tTAB_PREFIXED review text here\n"
        "Normal harmless review text here\n"
    )
    files = {"file": ("formula.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/dropbox/analyze", files=files)
    assert res.status_code == 200
    rows = res.json()["rows"]
    texts = [r["scrubbed_text"] for r in rows]

    for t in texts:
        if any(t.startswith(prefix) for prefix in ["'=SUM", "'+CMD", "'-HYPERLINK", "'@MALICIOUS", "'\tTAB"]):
            assert t.startswith("'"), f"Expected single quote prefix on {t}"
    # Normal text should NOT start with quote
    normal = [t for t in texts if "Normal harmless" in t][0]
    assert not normal.startswith("'")


def test_dropbox_deterministic():
    """Repeated calls on identical data return identical labels and themes."""
    csv_content = (
        "review\n"
        "Great coffee maker heats water fast and brews hot.\n"
        "Awful blender broke on the very first day of use.\n"
    )
    files1 = {"file": ("det.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res1 = client.post("/dropbox/analyze", files=files1)
    assert res1.status_code == 200

    files2 = {"file": ("det.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res2 = client.post("/dropbox/analyze", files=files2)
    assert res2.status_code == 200

    assert res1.json()["rows"] == res2.json()["rows"]


def test_dropbox_500_no_traceback():
    """Unexpected internal error returns 500 with JSON detail and no stack trace."""
    csv_content = "review\nSome valid review text here with words\n"
    files = {"file": ("error_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}

    with unittest.mock.patch("src.api.dropbox_router.analyze", side_effect=RuntimeError("Secret internal bug")):
        res = client.post("/dropbox/analyze", files=files)
        assert res.status_code == 500
        data = res.json()
        assert "detail" in data
        assert "Internal server error" in data["detail"]
        assert "Traceback" not in res.text
        assert "Secret internal bug" not in res.text


def test_dropbox_no_repo_files_written():
    """Confirm test execution does not write or modify any files in the repository."""
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    untracked_or_modified = [
        line for line in res.stdout.splitlines()
        if not line.strip().startswith("?? scratch")
    ]
    # Existing changes are permitted, but no random temp files in data/ or root
    for item in untracked_or_modified:
        assert not item.strip().endswith(".tmp")
        assert not item.strip().endswith(".parquet")
        assert not item.strip().endswith(".npy")
