"""
SHAP feature importance for the LSTM that consumes all text features.

We run shap.GradientExplainer on a small background batch from the train
split and a sample of the test split, then aggregate the absolute SHAP
values across samples and time steps to get one number per input feature.

The resulting bar chart tells us which columns the LSTM actually leans on
when it decides shock vs normal.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import shap
import torch

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds
from src.shared.utils import load_model

from src.activity3.config import (
    MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TICKERS, get_device,
)
from src.activity3.data.build import build_ticker_bundle, get_all_daily_text
from src.activity3.part2_lstm_price.model import LSTMClassifier
from src.activity3.config import (
    LSTM_BIDIRECTIONAL, LSTM_DROPOUT, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
)


def shap_for_ticker(ticker: str, text_daily, n_background: int = 50,
                    n_samples: int = 100, device=None) -> None:
    logger.info(f"=== SHAP - LSTM_all on {ticker} ===")

    # SHAP plays better with CPU for LSTMs (autograd through MPS is flaky).
    device = torch.device("cpu") if device is None else device

    bundle = build_ticker_bundle(ticker, text_daily, text_feature_set="all")
    model = LSTMClassifier(
        n_features=len(bundle.columns),
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        bidirectional=LSTM_BIDIRECTIONAL,
    )
    load_model(model, f"LSTM_all_{ticker}", MODELS_DIR)
    model.to(device).eval()

    bg = torch.tensor(bundle.train.X_series[:n_background], dtype=torch.float32,
                      device=device)
    samples = torch.tensor(bundle.test.X_series[:n_samples], dtype=torch.float32,
                           device=device)

    explainer = shap.GradientExplainer(model, bg)
    shap_values = explainer.shap_values(samples)

    # shap_values may be a list (one entry per class) or a single array.
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        # Newer SHAP returns shape (n_samples, T, F, n_classes); pick class 1.
        if sv.ndim == 4:
            sv = sv[..., 1]

    # Mean of |SHAP| across samples and time -> one value per feature.
    importance = np.abs(sv).mean(axis=(0, 1))
    feature_names = bundle.columns

    order = np.argsort(importance)[::-1]
    ordered_names  = [feature_names[i] for i in order]
    ordered_values = [importance[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.3)))
    colors = []
    for name in ordered_names:
        if name.startswith("sent_"):
            colors.append("#0969da")
        elif name.startswith("emo_"):
            colors.append("#bf8700")
        elif name.startswith("tox_"):
            colors.append("#cf222e")
        else:
            colors.append("#1a7f37")
    ax.barh(ordered_names[::-1], ordered_values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"SHAP feature importance - LSTM with all text features ({ticker})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend by group.
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color="#1a7f37", label="price/volume"),
        mpatches.Patch(color="#0969da", label="sentiment"),
        mpatches.Patch(color="#bf8700", label="emotion"),
        mpatches.Patch(color="#cf222e", label="toxicity"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"shap_LSTM_all_{ticker}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved SHAP barchart for {ticker}")


def run(n_background: int = 50, n_samples: int = 100):
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)
    text_daily = get_all_daily_text()
    for ticker in TICKERS:
        try:
            shap_for_ticker(ticker, text_daily,
                            n_background=n_background, n_samples=n_samples)
        except Exception as e:
            logger.warning(f"SHAP failed for {ticker}: {e}")


if __name__ == "__main__":
    run()
