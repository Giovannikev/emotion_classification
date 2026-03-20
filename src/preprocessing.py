import re
from typing import Iterable, List

import pandas as pd


URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s]")


def basic_clean_text(text: str) -> str:
    lowered = text.lower()
    no_urls = URL_PATTERN.sub(" ", lowered)
    no_mentions = MENTION_PATTERN.sub(" ", no_urls)
    no_hashtags = HASHTAG_PATTERN.sub(r" \1 ", no_mentions)
    letters_only = NON_LETTER_PATTERN.sub(" ", no_hashtags)
    normalized_spaces = re.sub(r"\s+", " ", letters_only).strip()
    return normalized_spaces


def clean_text_series(texts: Iterable[str]) -> List[str]:
    return [basic_clean_text(t) for t in texts]


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    return df
