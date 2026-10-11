import re
import html
import pandas as pd

RAW_PATH = "data/raw/train.ft.txt"
OUT_PATH = "data/processed/reviews_clean.parquet"
SAMPLE_SIZE = 50_000
SEED = 42

# a period/!/? followed directly by a capital letter: "mess.Imagine"
MISSING_SPACE_RE = re.compile(r"([.!?])([A-Z])")
WS_RE = re.compile(r"\s+")


def read_ft(path, n=None):
    """Read fastText-format file: '__label__X text...' per line."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            label, text = line.split(" ", 1)
            rows.append((label, text.strip()))
    return pd.DataFrame(rows, columns=["label", "text"])


def normalize_text(text):
    text = html.unescape(text)                # &amp; -> &
    text = MISSING_SPACE_RE.sub(r"\1 \2", text)  # "mess.Imagine" -> "mess. Imagine"
    text = WS_RE.sub(" ", text).strip()       # collapse extra whitespace
    return text


def clean_reviews(df):
    n0 = len(df)
    df = df.copy()

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text")

    df["sentiment"] = df["label"].map(
        {"__label__1": "negative", "__label__2": "positive"}
    )
    df = df.drop(columns="label")

    print(f"Kept {len(df)}/{n0} reviews")
    return df


if __name__ == "__main__":
    raw = read_ft(RAW_PATH, n=500_000)   # read first 500k, then sample
    clean = clean_reviews(raw)

    clean = clean.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    clean["review_id"] = clean.index

    clean.to_parquet(OUT_PATH, index=False)
    print(clean["sentiment"].value_counts())
    print(clean.head())