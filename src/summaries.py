import os
import re
import json
from typing import Optional, Callable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.llm import generate


def sample_theme_reviews(
    df_merged: pd.DataFrame,
    df_reviews: pd.DataFrame,
    topic_id: int,
    max_n: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """Sample up to max_n reviews for a given theme, stratified by sentiment."""
    # Filter for the target topic
    sub_merged = df_merged[df_merged["topic_v1_merged"] == topic_id]
    if sub_merged.empty:
        raise ValueError(f"No reviews found for topic {topic_id}")

    # Merge with reviews to get text and sentiment
    # Avoid duplicate columns if review_id is already in both
    sub = sub_merged.merge(df_reviews[["review_id", "text", "sentiment"]], on="review_id")

    if len(sub) <= max_n:
        return sub.sort_values("review_id").reset_index(drop=True)

    # Stratified sampling by original sentiment column
    sample, _ = train_test_split(
        sub,
        train_size=max_n,
        stratify=sub["sentiment"],
        random_state=random_state
    )
    return sample.sort_values("review_id").reset_index(drop=True)


def build_summary_prompt(theme_label: str, sampled_reviews: pd.DataFrame) -> str:
    """Format reviews and the required prompt for theme summary generation."""
    reviews_text = "\n\n".join(
        [f"[{int(row.review_id)}] {row.text}" for _, row in sampled_reviews.iterrows()]
    )
    prompt = (
        f"Theme: {theme_label}\n\n"
        f"Reviews:\n{reviews_text}\n\n"
        f"Use ONLY the reviews provided. Every claim must cite review IDs in brackets (e.g. [1234]). "
        f"If something is not supported by the reviews, do not say it. "
        f"Return JSON with: summary (3 sentences), praises (up to 3), complaints (up to 3). "
        f"Each item has 'text' and 'review_ids'."
    )
    return prompt


def extract_bracket_ids(text: str) -> list[int]:
    """Extract integer review IDs found inside square brackets in text."""
    matches = re.findall(r"\[([0-9,\s]+)\]", text)
    ids = []
    for m in matches:
        for part in m.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    return list(dict.fromkeys(ids))  # preserve order without duplicates


def get_content_words(text: str) -> set[str]:
    """Extract content words (alphabetic, >=3 chars, not in stopwords)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in ENGLISH_STOP_WORDS}


def _clean_json_text(text: str) -> str:
    """Strip markdown code blocks or wrapping text to isolate JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        # Remove closing fence
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # If still not starting with {, search for the first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def parse_summary_json(
    raw_text: str,
    retry_func: Optional[Callable[[str], str]] = None
) -> dict:
    """Parse JSON output robustly, retrying once if parsing fails."""
    try:
        cleaned = _clean_json_text(raw_text)
        return json.loads(cleaned)
    except Exception as first_err:
        if retry_func is not None:
            fix_prompt = (
                f"The following text was supposed to be a valid JSON object with keys "
                f"'summary', 'praises', and 'complaints', but failed to parse ({first_err}):\n\n"
                f"{raw_text}\n\n"
                f"Return ONLY the fixed, valid JSON object, with no explanations or preamble."
            )
            try:
                fixed_raw = retry_func(fix_prompt)
                cleaned_fixed = _clean_json_text(fixed_raw)
                return json.loads(cleaned_fixed)
            except Exception as second_err:
                raise RuntimeError(
                    f"Failed to parse JSON after retry. Error 1: {first_err}, Error 2: {second_err}"
                )
        raise first_err


def _extract_ints(obj) -> list[int]:
    """Recursively extract all integer IDs from ints, strings, or nested lists."""
    res = []
    if isinstance(obj, int):
        res.append(obj)
    elif isinstance(obj, str):
        found = re.findall(r"\d+", obj)
        for f in found:
            res.append(int(f))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            res.extend(_extract_ints(item))
    return res


def normalize_summary_items(data: dict) -> list[dict]:
    """Extract standardized claim items from parsed summary data."""
    claims = []

    # 1. Summary
    if "summary" in data:
        sum_val = data["summary"]
        if isinstance(sum_val, dict):
            claims.append({
                "type": "summary",
                "text": sum_val.get("text", ""),
                "review_ids": sum_val.get("review_ids", [])
            })
        elif isinstance(sum_val, list):
            for item in sum_val:
                if isinstance(item, dict):
                    claims.append({
                        "type": "summary",
                        "text": item.get("text", ""),
                        "review_ids": item.get("review_ids", [])
                    })
                elif isinstance(item, str):
                    claims.append({
                        "type": "summary",
                        "text": item,
                        "review_ids": []
                    })
        elif isinstance(sum_val, str):
            claims.append({
                "type": "summary",
                "text": sum_val,
                "review_ids": []
            })

    # 2. Praises
    for item in data.get("praises", []):
        if isinstance(item, dict):
            claims.append({
                "type": "praise",
                "text": item.get("text", ""),
                "review_ids": item.get("review_ids", [])
            })
        elif isinstance(item, str):
            claims.append({
                "type": "praise",
                "text": item,
                "review_ids": []
            })

    # 3. Complaints
    for item in data.get("complaints", []):
        if isinstance(item, dict):
            claims.append({
                "type": "complaint",
                "text": item.get("text", ""),
                "review_ids": item.get("review_ids", [])
            })
        elif isinstance(item, str):
            claims.append({
                "type": "complaint",
                "text": item,
                "review_ids": []
            })

    # Normalize review_ids (ensure list of ints and combine with bracketed IDs in text)
    for c in claims:
        raw_ids = c["review_ids"]
        normalized_ids = _extract_ints(raw_ids)
        # Also extract from brackets in text
        text_ids = extract_bracket_ids(c["text"])
        combined_ids = list(dict.fromkeys(normalized_ids + text_ids))
        c["review_ids"] = combined_ids

    return claims


def check_faithfulness(
    theme_id: int,
    theme_label: str,
    claims: list[dict],
    sampled_reviews: pd.DataFrame
) -> tuple[list[dict], list[dict], list[dict]]:
    """Perform automated faithfulness checks:
    (1) For summary paragraphs: treated as uncited-by-design (not flagged for missing citations).
    (2) For praises and complaints: must cite at least one review ID.
    (3) Drop any cited ID that is not in the sample (e.g. theme 2 hallucinated ID).
    (4) Word-overlap check: flag claims whose cited reviews share no content words,
        skipping theme 24 (Spanish) and marking them as 'not_checked'.

    Returns:
        (flags, dropped_ids_records, not_checked_records)
    """
    sample_ids = set(sampled_reviews["review_id"].astype(int))
    review_text_by_id = dict(zip(sampled_reviews["review_id"].astype(int), sampled_reviews["text"]))

    flags = []
    dropped_ids_records = []
    not_checked_records = []

    for claim in claims:
        c_type = claim["type"]
        c_text = claim["text"]
        raw_ids = claim["review_ids"]

        # Drop any cited ID not in the sample
        valid_ids = []
        for rid in raw_ids:
            if rid in sample_ids:
                valid_ids.append(rid)
            else:
                dropped_ids_records.append({
                    "theme_id": theme_id,
                    "theme_label": theme_label,
                    "claim_type": c_type,
                    "dropped_id": rid,
                    "claim_text": c_text
                })
        # Update claim's review_ids with valid sample IDs only
        claim["review_ids"] = valid_ids

        # 1. Summary overview: treated as uncited-by-design
        if c_type == "summary":
            continue

        # 2. Praise and complaint claims
        reasons = []

        # Check: Every claim must cite at least one review
        if len(valid_ids) == 0:
            reasons.append("missing_citation")

        # Check: Word overlap
        if theme_id == 24:
            # Skip word-overlap check for Theme 24 (Spanish)
            not_checked_records.append({
                "theme_id": theme_id,
                "theme_label": theme_label,
                "claim_type": c_type,
                "claim_text": c_text,
                "cited_review_ids": valid_ids,
                "status": "not_checked"
            })
        else:
            if valid_ids:
                claim_words = get_content_words(c_text)
                cited_text = " ".join([review_text_by_id[rid] for rid in valid_ids])
                cited_words = get_content_words(cited_text)
                overlap = claim_words & cited_words
                if len(overlap) == 0:
                    reasons.append("no_content_word_overlap")

        if reasons:
            flags.append({
                "theme_id": theme_id,
                "theme_label": theme_label,
                "claim_type": c_type,
                "claim_text": c_text,
                "cited_review_ids": valid_ids,
                "reasons": reasons
            })

    return flags, dropped_ids_records, not_checked_records
