"""
FinBERT (ProsusAI/finbert) sentiment scoring.

Produces per-day positive/negative/neutral averages of Musk tweets. The
backbone is used in inference mode only.
"""
import pandas as pd

from src.activity3.config import SENTIMENT_FEATURES, TEXT_BACKBONE_NAME
from src.activity3.data.text_extractors import cached_score, aggregate_daily
from src.activity3.data.tweets import load_tweets


# FinBERT id2label is positive, negative, neutral (we keep this order).
LABELS = ["sent_pos", "sent_neg", "sent_neu"]
CACHE_FILE = "sentiment_tweets.parquet"


def get_per_tweet_sentiment(force: bool = False) -> pd.DataFrame:
    tweets_df = load_tweets()
    return cached_score(tweets_df, CACHE_FILE, TEXT_BACKBONE_NAME, LABELS,
                        force=force)


def get_daily_sentiment(force: bool = False) -> pd.DataFrame:
    scored = get_per_tweet_sentiment(force=force)
    return aggregate_daily(scored, SENTIMENT_FEATURES)
