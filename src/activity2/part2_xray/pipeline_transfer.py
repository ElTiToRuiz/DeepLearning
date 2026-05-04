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
    XRAY_TRANSFER_EPOCHS, XRAY_TRANSFER_LR,
    RANDOM_SEED, get_device,
)
from src.activity2.train import fit, evaluate
from src.activity2.plots import (
    plot_history, plot_confusion_matrix, plot_sample_predictions,
    sample_test_predictions,
)
from src.activity2.part2_xray.dataset import (
    get_dataloaders, download_xray_dataset, IMAGENET_MEAN, IMAGENET_STD,
)
from src.activity2.part2_xray.models.resnet18_transfer import (
    build_resnet18_transfer, count_params,
)


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

    model = build_resnet18_transfer(num_classes=XRAY_NUM_CLASSES).to(device)
    trainable, total = count_params(model)
    logger.info(
        f"ResNet18 transfer — parameters: {total:,} total, "
        f"{trainable:,} trainable ({100 * trainable / total:.2f}%)"
    )

    criterion = nn.CrossEntropyLoss()

    # SKIP_TRAINING=1 loads the saved checkpoint and skips training.
    skip_training = os.environ.get("SKIP_TRAINING") == "1"
    checkpoint_path = os.path.join(MODELS_DIR, "resnet18_transfer_xray.pth")
    history = None

    if skip_training and os.path.exists(checkpoint_path):
        logger.info("SKIP_TRAINING=1 — loading saved weights, no training.")
        load_model(model, "resnet18_transfer_xray", MODELS_DIR)
    else:
        if skip_training:
            logger.warning(
                f"SKIP_TRAINING=1 but no checkpoint at {checkpoint_path}. "
                f"Falling back to full training."
            )
        # Only optimize parameters with gradients (the new head).
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=XRAY_TRANSFER_LR,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5,
        )
        history = fit(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler, device,
            epochs=XRAY_TRANSFER_EPOCHS, model_label="ResNet18-Transfer-XRay",
            early_stopping_patience=4,
        )
        save_model(model, "resnet18_transfer_xray", MODELS_DIR)

    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, return_predictions=True,
    )
    logger.info(f"ResNet18 transfer — test accuracy: {test_acc:.2f}%")
    logger.info("\n" + classification_report(labels, preds, target_names=classes))

    if history is not None:
        plot_history(history, "ResNet18_Transfer_XRay", RESULTS_DIR)
    plot_confusion_matrix(labels, preds, classes,
                          "ResNet18_Transfer_XRay", RESULTS_DIR)
    plot_confusion_matrix(labels, preds, classes,
                          "ResNet18_Transfer_XRay", RESULTS_DIR, normalize=True)

    sample_imgs, sample_true, sample_pred = sample_test_predictions(
        model, test_loader.dataset, device, n_samples=16,
    )
    plot_sample_predictions(
        sample_imgs, sample_true, sample_pred,
        class_names=classes, model_name="ResNet18_Transfer_XRay",
        results_dir=RESULTS_DIR, n_samples=16,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    )

    return history, test_acc
