"""
Twitter-RoBERTa emotion scoring (cardiffnlp/twitter-roberta-base-emotion).

Produces per-day emotion averages of Musk tweets: joy, optimism, anger,
sadness. The backbone is used in inference mode only.
"""
import pandas as pd

from src.activity3.config import EMOTION_FEATURES, EMOTION_MODEL_NAME
from src.activity3.data.text_extractors import cached_score, aggregate_daily
from src.activity3.data.tweets import load_tweets


LABELS = ["emo_joy", "emo_optimism", "emo_anger", "emo_sadness"]
CACHE_FILE = "emotion_tweets.parquet"


def get_per_tweet_emotion(force: bool = False) -> pd.DataFrame:
    tweets_df = load_tweets()
    return cached_score(tweets_df, CACHE_FILE, EMOTION_MODEL_NAME, LABELS,
                        force=force)


def get_daily_emotion(force: bool = False) -> pd.DataFrame:
    scored = get_per_tweet_emotion(force=force)
    return aggregate_daily(scored, EMOTION_FEATURES, include_text=False)
