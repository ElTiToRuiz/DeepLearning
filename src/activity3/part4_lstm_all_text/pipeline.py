"""
Part 4: LSTM C trained on price + sentiment + emotion + toxicity (21 cols).

Same architecture and hyperparameters as parts 2 and 3; only the input
dimensionality grows. Used to test whether richer text representations
(emotion and toxicity on top of plain sentiment) help the LSTM predict
the shock label.
"""
import os
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds
from src.shared.utils import save_model, load_model

from src.activity3.config import (
    EARLY_STOP_PATIENCE, LSTM_BATCH, LSTM_BIDIRECTIONAL, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_HIDDEN_DIM, LSTM_LR, LSTM_NUM_LAYERS, LSTM_WD,
    MODELS_DIR, RANDOM_SEED, RESULTS_DIR, SHOCK_POS_WEIGHT, TICKERS,
    get_device,
)
from src.activity3.data.build import (
    CLASS_NAMES, build_ticker_bundle, make_loaders, get_all_daily_text,
)
from src.activity3.part2_lstm_price.model import LSTMClassifier
from src.activity3.plots import (
    plot_confusion_matrix, plot_history,
)
from src.activity3.train import fit, run_eval, unpack_series


TEXT_FEATURE_SET = "all"
MODEL_TAG = "LSTM_all"


def _build_model(n_features: int, device):
    model = LSTMClassifier(
        n_features=n_features,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        bidirectional=LSTM_BIDIRECTIONAL,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{MODEL_TAG} - parameters: {n_params:,}")
    return model


def run_for_ticker(ticker: str, text_daily: pd.DataFrame,
                   device: torch.device) -> Dict[str, object]:
    logger.info(f"=== Part 4 ({MODEL_TAG}) - {ticker} ===")

    bundle = build_ticker_bundle(ticker, text_daily,
                                 text_feature_set=TEXT_FEATURE_SET)
    train_loader, val_loader, test_loader = make_loaders(
        bundle, batch_size=LSTM_BATCH,
    )

    model = _build_model(n_features=len(bundle.columns), device=device)
    class_weights = torch.tensor([1.0, SHOCK_POS_WEIGHT], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_name = f"{MODEL_TAG}_{ticker}"
    checkpoint_path = os.path.join(MODELS_DIR, f"{checkpoint_name}.pth")
    skip_training   = os.environ.get("SKIP_TRAINING") == "1"
    history = None

    if skip_training and os.path.exists(checkpoint_path):
        logger.info("SKIP_TRAINING=1 - loading saved weights, no training.")
        load_model(model, checkpoint_name, MODELS_DIR)
    else:
        if skip_training:
            logger.warning(
                f"SKIP_TRAINING=1 but no checkpoint at {checkpoint_path}. "
                f"Falling back to full training."
            )
        optimizer = optim.AdamW(model.parameters(),
                                lr=LSTM_LR, weight_decay=LSTM_WD)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LSTM_LR,
            steps_per_epoch=len(train_loader), epochs=LSTM_EPOCHS,
            pct_start=0.3, anneal_strategy="cos",
        )
        history = fit(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler, device,
            epochs=LSTM_EPOCHS,
            model_label=f"{MODEL_TAG}-{ticker}",
            unpack_fn=unpack_series,
            early_stopping_patience=EARLY_STOP_PATIENCE,
            scheduler_per_batch=True,
        )
        save_model(model, checkpoint_name, MODELS_DIR)

    test_loss, test_acc, preds, probs, labels = run_eval(
        model, test_loader, criterion, device,
        unpack_fn=unpack_series, return_predictions=True,
    )
    logger.info(f"{MODEL_TAG} - test acc {test_acc:.2f}%")
    logger.info("\n" + classification_report(labels, preds,
                                             target_names=CLASS_NAMES,
                                             zero_division=0))

    base = f"{MODEL_TAG}_{ticker}"
    if history is not None:
        plot_history(history, base, RESULTS_DIR)
    plot_confusion_matrix(labels, preds, CLASS_NAMES, base, RESULTS_DIR)
    plot_confusion_matrix(labels, preds, CLASS_NAMES, base, RESULTS_DIR,
                          normalize=True)

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    metrics = {
        "acc": float(test_acc),
        "f1":  float(f1_score(labels, preds, average="binary", zero_division=0)),
        "auc": auc,
        "y_true": labels, "y_score": probs,
    }
    return {"history": history, "metrics": metrics, "bundle": bundle, "model": model}


def run() -> Dict[str, Dict[str, object]]:
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)
    device = get_device()
    logger.info(f"Selected device: {device}")

    text_daily = get_all_daily_text()
    results: Dict[str, Dict[str, object]] = {}
    for ticker in TICKERS:
        results[ticker] = run_for_ticker(ticker, text_daily, device)
    return results
