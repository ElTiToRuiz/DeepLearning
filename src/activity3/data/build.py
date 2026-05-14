"""
Build the per-ticker panel that joins prices + text-derived features + label,
materialize sliding-window tensors, and expose PyTorch Datasets +
DataLoaders for parts 1, 2, 3 and 4.

v5 design:
  Three text "feature sets" are extracted from the tweets by separate
  HuggingFace models (all in inference mode, no fine-tune):

    - sentiment (FinBERT): sent_pos, sent_neg, sent_neu        (3 cols)
    - emotion   (twitter-RoBERTa): emo_joy/optimism/anger/sadness (4 cols)
    - toxicity  (toxic-BERT): tox_toxic/severe_toxic/...        (6 cols)

  Each is cached as its own parquet so re-runs are instant.
  The `text_feature_set` flag picks which subset goes into the LSTM:

    - "none":      no text                  -> 8 cols  (LSTM A)
    - "sentiment": sentiment only           -> 11 cols (LSTM B)
    - "all":       sentiment+emo+toxicity   -> 21 cols (LSTM C)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.activity3.config import (
    ALL_TEXT_FEATURES, EMOTION_FEATURES, PRICE_FEATURES, RANDOM_SEED,
    SENTIMENT_FEATURES, TOXICITY_FEATURES, WINDOW,
)
from src.activity3.data.prices import load_ticker
from src.activity3.data.sentiment import get_daily_sentiment
from src.activity3.data.emotion import get_daily_emotion
from src.activity3.data.toxicity import get_daily_toxicity
from src.activity3.data.splits import temporal_split


CLASS_NAMES = ["normal", "shock"]


def feature_columns(text_feature_set: str) -> List[str]:
    base = list(PRICE_FEATURES)
    if text_feature_set == "none":
        return base
    if text_feature_set == "sentiment":
        return base + list(SENTIMENT_FEATURES)
    if text_feature_set == "all":
        return base + list(ALL_TEXT_FEATURES)
    raise ValueError(f"Unknown text_feature_set: {text_feature_set!r}")


# ---------------------------------------------------------------- daily text

def _load_all_daily_text() -> pd.DataFrame:
    """Outer-merge sentiment + emotion + toxicity daily frames so every
    column lives in one DataFrame indexed by date. Heavy lifting is cached
    inside the underlying functions."""
    sent  = get_daily_sentiment()                   # includes text_concat, n_tweets
    emo   = get_daily_emotion()
    tox   = get_daily_toxicity()
    merged = sent.merge(emo, on="date", how="outer") \
                 .merge(tox, on="date", how="outer")
    return merged


# ---------------------------------------------------------------- panel

def build_panel(ticker: str, text_daily: pd.DataFrame) -> pd.DataFrame:
    """Left-join prices with daily text features. Days without tweets get
    a fallback: 100% neutral sentiment, zero emotion, zero toxicity."""
    prices = load_ticker(ticker)
    panel = prices.merge(text_daily, on="date", how="left")

    panel["sent_pos"]    = panel["sent_pos"].fillna(0.0)
    panel["sent_neg"]    = panel["sent_neg"].fillna(0.0)
    panel["sent_neu"]    = panel["sent_neu"].fillna(1.0)
    for c in EMOTION_FEATURES + TOXICITY_FEATURES:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)
        else:
            panel[c] = 0.0

    panel["n_tweets"]    = panel["n_tweets"].fillna(0).astype(int)
    panel["text_concat"] = panel["text_concat"].fillna("")
    panel["has_text"]    = (panel["n_tweets"] > 0).astype(int)
    return panel


# ---------------------------------------------------------------- windows

@dataclass
class WindowedPanel:
    X_series: np.ndarray
    texts: list
    has_text: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.y)


def make_windows(panel: pd.DataFrame, columns: list) -> WindowedPanel:
    feats = panel[columns].to_numpy(dtype=np.float32)
    N = len(panel) - WINDOW + 1
    if N <= 0:
        raise ValueError(f"Panel too short for window {WINDOW} (got {len(panel)} rows)")

    X = np.empty((N, WINDOW, feats.shape[1]), dtype=np.float32)
    for i in range(N):
        X[i] = feats[i:i + WINDOW]

    end_idx = np.arange(WINDOW - 1, len(panel))
    y     = panel["y_shock"].to_numpy(dtype=np.int64)[end_idx]
    has   = panel["has_text"].to_numpy(dtype=np.int64)[end_idx]
    texts = panel["text_concat"].to_numpy()[end_idx].tolist()
    dates = pd.DatetimeIndex(panel["date"].to_numpy()[end_idx])
    return WindowedPanel(X_series=X, texts=texts, has_text=has, y=y, dates=dates)


def normalize_series(train_X, *others):
    mean = train_X.reshape(-1, train_X.shape[-1]).mean(axis=0)
    std  = train_X.reshape(-1, train_X.shape[-1]).std(axis=0) + 1e-8
    out = [(arr - mean) / std for arr in (train_X, *others)]
    return tuple(out), mean, std


# ---------------------------------------------------------------- datasets

class SeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------- bundles

@dataclass
class TickerBundle:
    ticker: str
    text_feature_set: str
    columns: list
    train: WindowedPanel
    val:   WindowedPanel
    test:  WindowedPanel
    feat_mean: Optional[np.ndarray] = None
    feat_std:  Optional[np.ndarray] = None
    panel: Optional[pd.DataFrame] = None


def _slice_windowed(wp, mask):
    return WindowedPanel(
        X_series=wp.X_series[mask],
        texts=[t for t, m in zip(wp.texts, mask) if m],
        has_text=wp.has_text[mask],
        y=wp.y[mask],
        dates=wp.dates[mask],
    )


def build_ticker_bundle(ticker: str,
                        text_daily: pd.DataFrame,
                        text_feature_set: str = "none") -> TickerBundle:
    """text_feature_set is one of {"none", "sentiment", "all"}."""
    panel = build_panel(ticker, text_daily)
    cols  = feature_columns(text_feature_set)

    windowed = make_windows(panel, cols)

    train_mask, val_mask, test_mask = temporal_split(windowed.dates)
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split for {ticker}: "
            f"train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
        )

    train = _slice_windowed(windowed, train_mask)
    val   = _slice_windowed(windowed, val_mask)
    test  = _slice_windowed(windowed, test_mask)

    (train_X, val_X, test_X), feat_mean, feat_std = normalize_series(
        train.X_series, val.X_series, test.X_series,
    )

    def _replace(wp, X):
        return WindowedPanel(X_series=X, texts=wp.texts, has_text=wp.has_text,
                             y=wp.y, dates=wp.dates)

    train = _replace(train, train_X)
    val   = _replace(val,   val_X)
    test  = _replace(test,  test_X)

    return TickerBundle(
        ticker=ticker, text_feature_set=text_feature_set, columns=cols,
        train=train, val=val, test=test,
        feat_mean=feat_mean, feat_std=feat_std,
        panel=panel,
    )


# Convenience: the main.py orchestrator caches `text_daily` once and
# passes it to every part to avoid re-merging on each build.
def get_all_daily_text() -> pd.DataFrame:
    return _load_all_daily_text()


# ---------------------------------------------------------------- loaders

def make_loaders(bundle, batch_size, num_workers: int = 0):
    loaders = []
    for split, shuffle in [(bundle.train, True), (bundle.val, False), (bundle.test, False)]:
        ds = SeriesDataset(split.X_series, split.y)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers))
    return tuple(loaders)


# ---------------------------------------------------------------- utility

def summarize_bundle(bundle: TickerBundle) -> Dict[str, object]:
    def _stats(wp):
        return {
            "n": len(wp),
            "pos_rate": float(wp.y.mean()) if len(wp) else float("nan"),
            "has_text_rate": float(wp.has_text.mean()) if len(wp) else float("nan"),
            "date_min": str(wp.dates.min()) if len(wp) else None,
            "date_max": str(wp.dates.max()) if len(wp) else None,
        }
    return {
        "ticker": bundle.ticker,
        "text_feature_set": bundle.text_feature_set,
        "n_features": len(bundle.columns),
        "train": _stats(bundle.train),
        "val":   _stats(bundle.val),
        "test":  _stats(bundle.test),
    }
