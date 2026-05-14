"""
Part 1: classical baselines per ticker.

For each ticker we train two models that do NOT use deep learning:
  - DummyClassifier (majority class predictor)
  - Logistic Regression on the concatenation of:
      * flattened windowed series features
      * TF-IDF features over the day's aggregated tweet text

The goal is to provide a defensible floor for the rest of the experiment:
if the DL models cannot beat these, deep learning does not justify itself.
"""
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from scipy.sparse import csr_matrix, hstack

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds

from src.activity3.config import (
    LOGREG_C, LOGREG_MAX_ITER, MODELS_DIR, RANDOM_SEED, RESULTS_DIR,
    TFIDF_MAX_FEATURES,
)
from src.activity3.data.build import (
    CLASS_NAMES, WindowedPanel, build_ticker_bundle, get_all_daily_text,
)
from src.activity3.plots import plot_confusion_matrix


def _flatten_series(wp: WindowedPanel) -> np.ndarray:
    """(N, WINDOW, F) -> (N, WINDOW*F)."""
    return wp.X_series.reshape(wp.X_series.shape[0], -1).astype(np.float32)


def _stack_features(series_X: np.ndarray, tfidf_X: csr_matrix) -> csr_matrix:
    """Concatenate dense series features with sparse TF-IDF features."""
    return hstack([csr_matrix(series_X), tfidf_X]).tocsr()


def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    return {
        "acc": float(accuracy_score(y_true, y_pred)) * 100.0,
        "f1":  float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "auc": float(auc),
    }


def run_for_ticker(ticker: str, text_daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Returns {model_name: {acc, f1, auc}} on the test split."""
    logger.info(f"=== Part 1 (classical) - {ticker} ===")

    bundle = build_ticker_bundle(ticker, text_daily, text_feature_set="none")
    train, val, test = bundle.train, bundle.val, bundle.test

    # --- 1a. Majority baseline (no text, just the label) ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(np.zeros((len(train), 1)), train.y)
    dummy_pred  = dummy.predict(np.zeros((len(test), 1)))
    dummy_proba = dummy.predict_proba(np.zeros((len(test), 1)))[:, 1]
    dummy_m = _compute_metrics(test.y, dummy_pred, dummy_proba)
    logger.info(f"Majority - test acc {dummy_m['acc']:.2f}% f1 {dummy_m['f1']:.3f}")

    # --- 1b. LogReg on series + TF-IDF ---
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    train_tfidf = tfidf.fit_transform(train.texts)
    val_tfidf   = tfidf.transform(val.texts)
    test_tfidf  = tfidf.transform(test.texts)

    train_X = _stack_features(_flatten_series(train), train_tfidf)
    val_X   = _stack_features(_flatten_series(val),   val_tfidf)
    test_X  = _stack_features(_flatten_series(test),  test_tfidf)

    # class_weight='balanced' compensates the ~80/20 imbalance of the
    # shock label so the LogReg is not driven by the majority class.
    logreg = LogisticRegression(
        C=LOGREG_C, max_iter=LOGREG_MAX_ITER, random_state=RANDOM_SEED,
        solver="liblinear", class_weight="balanced",
    )
    logreg.fit(train_X, train.y)
    val_pred  = logreg.predict(val_X)
    val_proba = logreg.predict_proba(val_X)[:, 1]
    test_pred  = logreg.predict(test_X)
    test_proba = logreg.predict_proba(test_X)[:, 1]
    logreg_m = _compute_metrics(test.y, test_pred, test_proba)
    val_m    = _compute_metrics(val.y,  val_pred,  val_proba)
    logger.info(
        f"LogReg - val acc {val_m['acc']:.2f}% f1 {val_m['f1']:.3f} "
        f"| test acc {logreg_m['acc']:.2f}% f1 {logreg_m['f1']:.3f} "
        f"auc {logreg_m['auc']:.3f}"
    )
    logger.info("\n" + classification_report(test.y, test_pred,
                                             target_names=CLASS_NAMES,
                                             zero_division=0))

    # Confusion matrices.
    plot_confusion_matrix(test.y, dummy_pred,  CLASS_NAMES,
                          f"Majority_{ticker}",  RESULTS_DIR)
    plot_confusion_matrix(test.y, test_pred,   CLASS_NAMES,
                          f"LogReg_{ticker}",    RESULTS_DIR)
    plot_confusion_matrix(test.y, test_pred,   CLASS_NAMES,
                          f"LogReg_{ticker}",    RESULTS_DIR, normalize=True)

    # Persist the fitted models for later inspection (optional).
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(dummy,  os.path.join(MODELS_DIR, f"majority_{ticker}.joblib"))
    joblib.dump({"tfidf": tfidf, "logreg": logreg,
                 "feat_mean": bundle.feat_mean, "feat_std": bundle.feat_std},
                os.path.join(MODELS_DIR, f"logreg_{ticker}.joblib"))

    # Also expose the test probabilities so the orchestrator can build a
    # single ROC plot with every model.
    return {
        "Majority": {**dummy_m,  "y_true": test.y, "y_score": dummy_proba},
        "LogReg":   {**logreg_m, "y_true": test.y, "y_score": test_proba},
    }


def run() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run part 1 over all tickers. Returns {ticker: {model: metrics}}."""
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)

    text_daily = get_all_daily_text()

    from src.activity3.config import TICKERS
    results = {}
    for ticker in TICKERS:
        results[ticker] = run_for_ticker(ticker, text_daily)
    return results
