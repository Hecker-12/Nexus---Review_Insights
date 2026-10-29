"""PII redaction utilities for customer review texts."""

import re

URL_PATTERN = re.compile(r'(?:https?://[^\s]+|www\.[^\s]+)', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE)
PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b')


def scrub_pii(text: str) -> str:
    """Scrub sensitive PII (emails, phone numbers, URLs) from review text."""
    if not text:
        return ""
    # 1. Scrub URLs first to prevent overlap with emails or numbers
    text = URL_PATTERN.sub("<URL>", text)
    # 2. Scrub Emails
    text = EMAIL_PATTERN.sub("<EMAIL>", text)
    # 3. Scrub Phone numbers
    text = PHONE_PATTERN.sub("<PHONE>", text)
    return text
