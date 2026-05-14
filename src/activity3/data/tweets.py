"""
Download and load Elon Musk's tweets from the HuggingFace dataset
`fdaudens/musk-tweets` (~78k tweets, 2013-03 to 2025-05) and aggregate
them by UTC day.

Was previously a Kaggle dataset capped at April 2021; HuggingFace covers
the post-2021 era when Musk became the most active mover of TSLA / DOGE.
"""
import os
import re
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from src.activity3.config import TWEETS_HF_REPO, TWEETS_HF_FILE


# Column aliases. The HF dataset uses CreatedTime / Message; we keep the
# old Kaggle aliases as fallback so old caches still work.
_DATE_ALIASES = ["CreatedTime", "created_at", "date", "Date", "datetime", "time", "Time"]
_TEXT_ALIASES = ["Message", "tweet", "text", "content", "Text", "Tweet"]

_URL_RE    = re.compile(r"https?://\S+")
_HANDLE_RE = re.compile(r"@\w+")
_WS_RE     = re.compile(r"\s+")


def download_musk_tweets() -> str:
    """Download the Musk tweets CSV from HuggingFace Hub and return its
    local path. HF Hub caches under ~/.cache/huggingface/."""
    return hf_hub_download(
        repo_id=TWEETS_HF_REPO,
        filename=TWEETS_HF_FILE,
        repo_type="dataset",
    )


def _pick_column(df: pd.DataFrame, aliases) -> str:
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(f"None of {aliases} found in columns: {list(df.columns)}")


def _clean_text(t):
    if not isinstance(t, str):
        return ""
    t = _URL_RE.sub("", t)
    t = _HANDLE_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def load_tweets(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      - date (datetime64[ns], naive UTC date)
      - text (cleaned)
    Sorted ascending by date.
    """
    if csv_path is None:
        csv_path = download_musk_tweets()

    # The HF CSV contains a few rows with stray separators that pandas can
    # choke on; on_bad_lines='skip' keeps us robust.
    df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")

    # Filter retweets and non-English rows when those columns exist.
    if "retweet" in df.columns:
        df = df[df["retweet"].fillna(False) == False].copy()
    if "language" in df.columns:
        df = df[df["language"].fillna("en") == "en"].copy()
    # Some HF dumps have an OriginalAuthor column on retweets: drop those.
    if "OriginalAuthor" in df.columns:
        df = df[df["OriginalAuthor"].isna() | (df["OriginalAuthor"] == "")].copy()

    date_col = _pick_column(df, _DATE_ALIASES)
    text_col = _pick_column(df, _TEXT_ALIASES)

    df = df[[date_col, text_col]].rename(columns={date_col: "date", text_col: "text"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)

    df["text"] = df["text"].astype(str).map(_clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def aggregate_by_day(df: pd.DataFrame, sep: str = " [SEP] ") -> pd.DataFrame:
    """Collapse multiple tweets per day into a single row."""
    grouped = (
        df.groupby("date", as_index=False)
          .agg(text_concat=("text", lambda xs: sep.join(xs)),
               n_tweets=("text", "size"))
    )
    return grouped
