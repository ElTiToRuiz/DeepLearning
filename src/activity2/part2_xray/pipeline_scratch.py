import os

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds
from src.shared.utils import save_model, load_model

from src.activity2.config import (
    MODELS_DIR, RESULTS_DIR,
    XRAY_NUM_CLASSES, XRAY_IMG_SIZE, XRAY_BATCH_SIZE,
    XRAY_SCRATCH_EPOCHS, XRAY_SCRATCH_MAX_LR, XRAY_SCRATCH_WEIGHT_DECAY,
    RANDOM_SEED, get_device,
)
from src.activity2.train import fit, evaluate
from src.activity2.plots import plot_history, plot_confusion_matrix
from src.activity2.part2_xray.dataset import get_dataloaders, download_xray_dataset
from src.activity2.part2_xray.models.cnn_scratch import CNNScratch


def run():
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)

    device = get_device()
    logger.info(f"Selected device: {device}")

    data_dir = download_xray_dataset()
    logger.info(f"Dataset available at: {data_dir}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir, batch_size=XRAY_BATCH_SIZE, img_size=XRAY_IMG_SIZE,
    )
    logger.info(f"Classes: {classes}")

    model = CNNScratch(num_classes=XRAY_NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CNNScratch — parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SKIP_TRAINING=1 loads the saved checkpoint and skips training.
    skip_training = os.environ.get("SKIP_TRAINING") == "1"
    checkpoint_path = os.path.join(MODELS_DIR, "cnn_scratch_xray.pth")
    history = None

    if skip_training and os.path.exists(checkpoint_path):
        logger.info("SKIP_TRAINING=1 — loading saved weights, no training.")
        load_model(model, "cnn_scratch_xray", MODELS_DIR)
    else:
        if skip_training:
            logger.warning(
                f"SKIP_TRAINING=1 but no checkpoint at {checkpoint_path}. "
                f"Falling back to full training."
            )
        optimizer = optim.Adam(
            model.parameters(),
            lr=XRAY_SCRATCH_MAX_LR, weight_decay=XRAY_SCRATCH_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=XRAY_SCRATCH_MAX_LR,
            steps_per_epoch=len(train_loader), epochs=XRAY_SCRATCH_EPOCHS,
            pct_start=0.3, anneal_strategy="cos",
        )
        history = fit(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler, device,
            epochs=XRAY_SCRATCH_EPOCHS, model_label="CNNScratch-XRay",
            early_stopping_patience=5,
            scheduler_per_batch=True,
        )
        save_model(model, "cnn_scratch_xray", MODELS_DIR)

    # Final test evaluation
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, return_predictions=True,
    )
    logger.info(f"CNNScratch — test accuracy: {test_acc:.2f}%")
    logger.info("\n" + classification_report(labels, preds, target_names=classes))

    if history is not None:
        plot_history(history, "CNNScratch_XRay", RESULTS_DIR)
    plot_confusion_matrix(labels, preds, classes,
                          "CNNScratch_XRay", RESULTS_DIR)

    return history, test_acc
