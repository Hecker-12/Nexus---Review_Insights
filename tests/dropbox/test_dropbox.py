"""Unit tests for the dropbox package."""

import io
import os
import pytest
import pandas as pd
import numpy as np

from src.dropbox.errors import (
    EmptyFileError,
    InvalidFileTypeError,
    FileSizeLimitError,
    NoTextColumnError,
    ColumnNotFoundError,
)
from src.dropbox.loader import load_file
from src.dropbox.cleaning import clean_reviews
from src.dropbox.pii import scrub_pii
from src.dropbox.scorer import score_texts, get_model_info
from src.dropbox.themes import match_themes, get_theme_info
from src.dropbox.pipeline import analyze


# ---------------------------------------------------------------------------
# 1. PII Scrubbing Tests (8 test examples)
# ---------------------------------------------------------------------------

def test_pii_8_examples():
    examples = [
        ("Contact me at user@example.com for info.", "Contact me at <EMAIL> for info."),
        ("Email John.Doe123@Sub.Domain.org today.", "Email <EMAIL> today."),
        ("Check out http://example.com/item/123 now.", "Check out <URL> now."),
        ("Visit https://www.amazon.com/product?id=99 for deals.", "Visit <URL> for deals."),
        ("Go to www.test.co.uk/page immediately.", "Go to <URL> immediately."),
        ("Call customer service at 555-867-5309.", "Call customer service at <PHONE>."),
        ("Reach our line at (800) 555-0199.", "Reach our line at <PHONE>."),
        ("International support at +1-800-555-1234.", "International support at <PHONE>.")
    ]
    for raw, expected in examples:
        assert scrub_pii(raw) == expected


# ---------------------------------------------------------------------------
# 2. Loader Tests
# ---------------------------------------------------------------------------

def test_loader_empty_file():
    with pytest.raises(EmptyFileError):
        load_file(b"", "test.csv")


def test_loader_oversized_file():
    big_bytes = b"x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(FileSizeLimitError):
        load_file(big_bytes, "big.csv")


def test_loader_invalid_file_type():
    with pytest.raises(InvalidFileTypeError):
        load_file(b"some,data\n1,2", "test.json")


def test_loader_valid_csv_and_text_detection():
    csv_bytes = b"id,review_text,score\n1,This is a fantastic product that I love,5\n2,Broke immediately after opening,1"
    df, text_col, rating_col, rows_in, sampled = load_file(csv_bytes, "sample.csv")
    assert rows_in == 2
    assert not sampled
    assert text_col == "review_text"
    assert rating_col == "score"


def test_loader_valid_xlsx(tmp_path):
    df_in = pd.DataFrame({
        "customer_feedback": ["Excellent quality and great battery life", "Poor build quality"],
        "rating": [5, 1]
    })
    buf = io.BytesIO()
    df_in.to_excel(buf, index=False)
    xlsx_bytes = buf.getvalue()

    df, text_col, rating_col, rows_in, sampled = load_file(xlsx_bytes, "feedback.xlsx")
    assert rows_in == 2
    assert text_col == "customer_feedback"
    assert rating_col == "rating"


def test_loader_override_column():
    csv_bytes = b"short,long_description\na,This is a very detailed review of the product\nb,Another very detailed review"
    # By default, long_description would be picked. Override with 'short':
    df, text_col, _, _, _ = load_file(csv_bytes, "override.csv", text_column="short")
    assert text_col == "short"

    # Missing override raises ColumnNotFoundError:
    with pytest.raises(ColumnNotFoundError):
        load_file(csv_bytes, "override.csv", text_column="nonexistent_column")


def test_loader_sampling_at_3000():
    # 3500 rows should sample to 3000
    rows = ["id,text,star\n"] + [f"{i},This is review text number {i},5\n" for i in range(3500)]
    csv_bytes = "".join(rows).encode("utf-8")
    df, text_col, rating_col, rows_in, sampled = load_file(csv_bytes, "large.csv")
    assert rows_in == 3500
    assert sampled is True
    assert len(df) == 3000


# ---------------------------------------------------------------------------
# 3. Cleaning Tests
# ---------------------------------------------------------------------------

def test_cleaning_drops():
    df = pd.DataFrame({
        "text": [
            "  Valid review text with enough words  ",  # keep (stripped)
            "   ",                                       # empty
            "Two words",                                 # under 3 words
            "Valid review text with enough words",       # duplicate
            "Another distinct valid review here"         # keep
        ]
    })
    df_clean, counts = clean_reviews(df, text_col="text")
    assert len(df_clean) == 2
    assert counts["removed_empty"] == 1
    assert counts["removed_under_3_words"] == 1
    assert counts["removed_duplicates"] == 1
    assert counts["rows_used"] == 2


# ---------------------------------------------------------------------------
# 4. Scorer Tests
# ---------------------------------------------------------------------------

def test_scorer_classes_and_sentences():
    info = get_model_info()
    assert info["classes"] == ["negative", "positive"]

    test_sentences = [
        "Terrible, stopped working after two days",
        "Absolutely love it, works perfectly",
        "It arrived"
    ]
    results = score_texts(test_sentences)
    assert len(results) == 3
    # Check sensible predictions for first two
    assert results[0][0] == "negative"
    assert results[1][0] == "positive"
    assert 0.0 <= results[2][1] <= 1.0


# ---------------------------------------------------------------------------
# 5. Themes Tests
# ---------------------------------------------------------------------------

def test_themes_count_and_matching():
    info = get_theme_info()
    assert info["num_themes"] == 29
    assert -1 not in info["theme_ids"]

    # Coffee maker query
    res = match_themes(["my coffee maker leaks water all over the counter"])
    assert len(res) == 1
    assert res[0] == "Coffee makers"


# ---------------------------------------------------------------------------
# 6. Pipeline Tests with 200 Synthetic Rows
# ---------------------------------------------------------------------------

def test_pipeline_synthetic_200_rows():
    # Generate 200 synthetic rows with ratings
    np.random.seed(42)
    sentences = [
        "Great quality coffee maker, brew is hot and delicious.",
        "Terrible defect, stopped working after three days of use.",
        "Fast shipping, excellent packaging, love this item very much.",
        "Completely useless item, broke on the first try, awful quality."
    ]
    ratings = [5, 1, 4, 1]

    data_rows = []
    for i in range(200):
        idx = i % 4
        # Add some PII in a few rows
        txt = sentences[idx]
        if i % 20 == 0:
            txt += f" Contact user{i}@example.com or 555-010-{i:04d}."
        data_rows.append({"review_body": txt, "star_rating": ratings[idx]})

    df_synth = pd.DataFrame(data_rows)
    buf = io.BytesIO()
    df_synth.to_csv(buf, index=False)
    csv_bytes = buf.getvalue()

    result = analyze(csv_bytes, filename="synthetic_reviews.csv")

    assert "report" in result
    assert "summary" in result
    assert "accuracy_vs_ratings" in result
    assert "themes_table" in result
    assert "preview" in result
    assert "rows" in result
    assert "warnings" in result

    # Check warnings
    assert len(result["warnings"]) == 2
    assert "Amazon product reviews" in result["warnings"][0]

    # Check PII was scrubbed
    pii_found = any("@example.com" in r["scrubbed_text"] for r in result["rows"])
    assert not pii_found, "Raw email should be scrubbed from rows"

    # Check accuracy_vs_ratings
    acc_res = result["accuracy_vs_ratings"]
    assert acc_res is not None
    assert acc_res["accuracy"] > 0.80
    assert acc_res["macro_f1"] > 0.80

    # Check preview length
    assert len(result["preview"]) <= 20
