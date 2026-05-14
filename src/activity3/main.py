"""
Activity 3 v5 orchestrator.

Pipeline:
  Stage 0: extract sentiment + emotion + toxicity per tweet with FinBERT,
           Twitter-RoBERTa-emotion and Toxic-BERT respectively (all in
           inference mode, all cached).
  Part 1:  classical baselines (Majority + LogReg with TF-IDF over tweets).
  Part 2:  LSTM A with 8 price-only features.
  Part 3:  LSTM B with 8 price features plus 3 sentiment features (11 cols).
  Part 4:  LSTM C with 8 price features plus 13 text features (21 cols).

Final SHAP analysis on LSTM C, ROC curves per ticker, sentiment time
series plots and a global bar chart.
"""
from typing import Dict

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds

from src.activity3.config import RANDOM_SEED, RESULTS_DIR, TICKERS
from src.activity3.data.build import build_ticker_bundle, get_all_daily_text
from src.activity3.plots import (
    plot_metrics_bar, plot_roc_curves, plot_sentiment_timeseries,
)

from src.activity3.part1_classical.pipeline import run as run_part1
from src.activity3.part2_lstm_price.pipeline import run as run_part2
from src.activity3.part3_lstm_sentiment.pipeline import run as run_part3
from src.activity3.part4_lstm_all_text.pipeline import run as run_part4
from src.activity3 import shap_analysis


def _collect_metrics(part1, part2, part3, part4) -> Dict:
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for ticker in TICKERS:
        models = {}
        for name, m in part1.get(ticker, {}).items():
            models[name] = m
        models["LSTM price"]      = part2[ticker]["metrics"]
        models["LSTM price+sent"] = part3[ticker]["metrics"]
        models["LSTM price+all"]  = part4[ticker]["metrics"]
        out[ticker] = models
    return out


def _strip_arrays(metrics: Dict) -> Dict:
    scalar = {}
    for ticker, models in metrics.items():
        scalar[ticker] = {}
        for name, m in models.items():
            scalar[ticker][name] = {k: m[k] for k in ("acc", "f1", "auc") if k in m}
    return scalar


def _print_summary(metrics):
    logger.info("=" * 78)
    logger.info("FINAL SUMMARY  test acc / F1 / AUC by model and ticker")
    logger.info("=" * 78)
    header = f"{'Model':<22}" + "".join(f"{t:>16}" for t in TICKERS)
    logger.info(header)
    logger.info("." * len(header))
    all_models = []
    for ticker in TICKERS:
        for m in metrics[ticker]:
            if m not in all_models:
                all_models.append(m)
    for model in all_models:
        row = f"{model:<22}"
        for ticker in TICKERS:
            m = metrics[ticker].get(model, {})
            acc = m.get("acc", float("nan"))
            f1  = m.get("f1",  float("nan"))
            auc = m.get("auc", float("nan"))
            row += f" {acc:5.1f}% {f1:.2f} {auc:.2f}"
        logger.info(row)
    logger.info("=" * 78)


def run():
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)

    logger.info("##### Stage 0: text feature extraction (cached) #####")
    text_daily = get_all_daily_text()
    logger.info(f"  Daily text rows: {len(text_daily)}, "
                f"date range: {text_daily.date.min()} -> {text_daily.date.max()}")

    for ticker in TICKERS:
        bundle = build_ticker_bundle(ticker, text_daily, text_feature_set="all")
        plot_sentiment_timeseries(bundle.panel, ticker, RESULTS_DIR)

    logger.info("##### Part 1: classical baselines (Majority + LogReg) #####")
    part1 = run_part1()

    logger.info("##### Part 2: LSTM on price-only features #####")
    part2 = run_part2()

    logger.info("##### Part 3: LSTM on price + sentiment features #####")
    part3 = run_part3()

    logger.info("##### Part 4: LSTM on price + sentiment + emotion + toxicity #####")
    part4 = run_part4()

    logger.info("##### SHAP feature importance on LSTM C (all text features) #####")
    shap_analysis.run(n_background=50, n_samples=100)

    all_metrics = _collect_metrics(part1, part2, part3, part4)
    plot_metrics_bar(_strip_arrays(all_metrics), RESULTS_DIR,
                     name="final_comparison")

    for ticker in TICKERS:
        roc_inputs = {
            name: (m["y_true"], m["y_score"])
            for name, m in all_metrics[ticker].items()
            if "y_true" in m and "y_score" in m
        }
        plot_roc_curves(roc_inputs, RESULTS_DIR, name=ticker)

    _print_summary(all_metrics)


if __name__ == "__main__":
    run()
