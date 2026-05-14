"""
Download daily OHLCV via yfinance, engineer features and add the
binary next-day-volatility label.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.activity3.config import (
    DATASET_DIR, LABEL_ROLLING_WINDOW, SHOCK_PERCENTILE,
    START_DATE, END_DATE,
)


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(DATASET_DIR, f"prices_{safe}.parquet")


def download_prices(ticker: str,
                    start: Optional[str] = None,
                    end: Optional[str] = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    Download (or load from local cache) daily OHLCV for `ticker`.
    Caches to dataset/prices_<TICKER>.parquet.

    Returns a DataFrame with columns: date, open, high, low, close, volume.
    """
    start = start or START_DATE
    end   = end   or END_DATE  # None => yfinance fetches up to today.

    os.makedirs(DATASET_DIR, exist_ok=True)
    cache = _cache_path(ticker)

    if use_cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                      progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance >=0.2 returns multiindex columns even for a single ticker.
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [c.lower() if isinstance(c, str) else c for c in raw.columns]
    raw = raw.rename(columns={"adj close": "adj_close"})

    keep = ["date", "open", "high", "low", "close", "volume"]
    df = raw[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    df.to_parquet(cache, index=False)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used by every model:
      - log_return        : log(Close_d / Close_{d-1})
      - intraday_range    : (High - Low) / Close
      - vol_5d, vol_20d   : rolling std of log_return
      - volume_ratio      : Volume_d / rolling_mean(Volume, 20)
      - rel_open/high/low : (X_d / Close_d) - 1
      - realized_vol      : |log_return| (used to build the label)
    Drops the first rows that lack history.
    """
    df = df.copy()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol_5d"]         = df["log_return"].rolling(5).std()
    df["vol_20d"]        = df["log_return"].rolling(20).std()
    df["volume_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()
    df["rel_open"]       = df["open"] / df["close"] - 1.0
    df["rel_high"]       = df["high"] / df["close"] - 1.0
    df["rel_low"]        = df["low"]  / df["close"] - 1.0
    df["realized_vol"]   = df["log_return"].abs()
    return df


def add_labels(df: pd.DataFrame,
               lookback: int = LABEL_ROLLING_WINDOW,
               percentile: float = SHOCK_PERCENTILE) -> pd.DataFrame:
    """
    Three next-day labels:
      y_shock  = 1 if realized_vol_{d+1} > rolling_quantile(realized_vol, lookback, q=percentile)
               (main task — "shock detector")
      y_dir    = 1 if log_return_{d+1} > 0   (auxiliary task — direction)
      y_mag    = |log_return_{d+1}|          (auxiliary task — magnitude, regression)

    The rolling quantile is computed on past data only (shifted by 1) to avoid
    lookahead leakage.

    The canonical column `y` is kept as an alias of `y_shock` so callers that
    only care about the main task don't need to change.
    """
    df = df.copy()
    rv = df["realized_vol"]
    threshold = rv.shift(1).rolling(lookback, min_periods=lookback).quantile(percentile)
    df["vol_threshold"] = threshold
    df["realized_vol_next"] = rv.shift(-1)
    df["log_return_next"]   = df["log_return"].shift(-1)

    df["y_shock"] = (df["realized_vol_next"] > df["vol_threshold"]).astype("Int64")
    df["y_dir"]   = (df["log_return_next"] > 0).astype("Int64")
    df["y_mag"]   = df["realized_vol_next"].astype(float)
    df["y"] = df["y_shock"]   # alias for single-task callers

    df = df.dropna(subset=["realized_vol_next", "vol_threshold",
                           "log_return_next"]).copy()
    for c in ("y_shock", "y_dir", "y"):
        df[c] = df[c].astype(int)
    return df


def load_ticker(ticker: str,
                start: Optional[str] = None,
                end: Optional[str] = None,
                use_cache: bool = True) -> pd.DataFrame:
    """Convenience wrapper: download + features + labels."""
    df = download_prices(ticker, start=start, end=end, use_cache=use_cache)
    df = add_features(df)
    df = add_labels(df)
    df = df.dropna(subset=["log_return", "vol_5d", "vol_20d", "volume_ratio"]).reset_index(drop=True)
    return df
