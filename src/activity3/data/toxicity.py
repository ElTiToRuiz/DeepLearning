"""
Toxic-BERT scoring (unitary/toxic-bert).

Produces per-day toxicity scores of Musk tweets across six axes:
toxic, severe_toxic, obscene, threat, insult, identity_hate.
"""
import pandas as pd

from src.activity3.config import TOXICITY_FEATURES, TOXICITY_MODEL_NAME
from src.activity3.data.text_extractors import cached_score, aggregate_daily
from src.activity3.data.tweets import load_tweets


LABELS = ["tox_toxic", "tox_severe_toxic", "tox_obscene", "tox_threat",
          "tox_insult", "tox_identity_hate"]
CACHE_FILE = "toxicity_tweets.parquet"


def get_per_tweet_toxicity(force: bool = False) -> pd.DataFrame:
    tweets_df = load_tweets()
    return cached_score(tweets_df, CACHE_FILE, TOXICITY_MODEL_NAME, LABELS,
                        force=force)


def get_daily_toxicity(force: bool = False) -> pd.DataFrame:
    scored = get_per_tweet_toxicity(force=force)
    return aggregate_daily(scored, TOXICITY_FEATURES, include_text=False)
