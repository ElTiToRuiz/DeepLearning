import os

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.shared.logger import setup_logger, logger
from src.shared.seeds import set_seeds
from src.shared.utils import save_model, load_model

from src.activity2.config import (
    CIFAR10_DIR, MODELS_DIR, RESULTS_DIR,
    CIFAR10_NUM_CLASSES, CIFAR10_BATCH_SIZE, CIFAR10_EPOCHS,
    CIFAR10_MAX_LR, CIFAR10_WEIGHT_DECAY,
    RANDOM_SEED, get_device,
)
from src.activity2.train import fit, evaluate
from src.activity2.plots import (
    plot_history, plot_confusion_matrix, plot_sample_predictions,
    sample_test_predictions,
)
from src.activity2.part1_cifar10.model import AlexNet


# Standard CIFAR-10 channel statistics for normalization.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def ensure_cifar10_root() -> str:
    """
    Return a directory that contains `cifar-10-batches-py/` ready for
    torchvision. Tries the expected location first; if missing, downloads
    the dataset via kagglehub (stable mirror used as a fallback when the
    Toronto server is down).
    """
    if os.path.isdir(os.path.join(CIFAR10_DIR, "cifar-10-batches-py")):
        return CIFAR10_DIR
    logger.info("CIFAR-10 not found locally. Downloading via kagglehub...")
    kh_root = kagglehub.dataset_download("pankrzysiu/cifar10-python")
    if not os.path.isdir(os.path.join(kh_root, "cifar-10-batches-py")):
        raise RuntimeError(
            f"kagglehub downloaded but cifar-10-batches-py not found at {kh_root}"
        )
    return kh_root


def get_cifar10_loaders():
    """
    Return (train_loader, test_loader). Standard CIFAR-10 setup with
    augmentation on train (horizontal flip + random crop) and only
    normalization on test.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    cifar_root = ensure_cifar10_root()
    train_set = torchvision.datasets.CIFAR10(
        root=cifar_root, train=True,  download=False, transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cifar_root, train=False, download=False, transform=transform_test,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=CIFAR10_BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=pin_memory, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=CIFAR10_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=pin_memory, persistent_workers=True,
    )

    return train_loader, test_loader


def run():
    setup_logger(RESULTS_DIR)
    set_seeds(RANDOM_SEED)

    device = get_device()
    logger.info(f"Selected device: {device}")

    train_loader, test_loader = get_cifar10_loaders()

    model = AlexNet(num_classes=CIFAR10_NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"AlexNet — parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SKIP_TRAINING=1 (env var) loads the saved checkpoint and skips training.
    skip_training = os.environ.get("SKIP_TRAINING") == "1"
    checkpoint_path = os.path.join(MODELS_DIR, "alexnet_cifar10.pth")
    history = None

    if skip_training and os.path.exists(checkpoint_path):
        logger.info("SKIP_TRAINING=1 — loading saved weights, no training.")
        load_model(model, "alexnet_cifar10", MODELS_DIR)
    else:
        if skip_training:
            logger.warning(
                f"SKIP_TRAINING=1 but no checkpoint at {checkpoint_path}. "
                f"Falling back to full training."
            )
        optimizer = optim.Adam(model.parameters(),
                               lr=CIFAR10_MAX_LR, weight_decay=CIFAR10_WEIGHT_DECAY)
        # OneCycleLR: per-batch warmup + cosine annealing.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CIFAR10_MAX_LR,
            steps_per_epoch=len(train_loader), epochs=CIFAR10_EPOCHS,
            pct_start=0.2, anneal_strategy="cos",
        )
        history = fit(
            model, train_loader, test_loader,
            criterion, optimizer, scheduler, device,
            epochs=CIFAR10_EPOCHS, model_label="AlexNet-CIFAR10",
            early_stopping_patience=10,
            scheduler_per_batch=True,
        )
        save_model(model, "alexnet_cifar10", MODELS_DIR)

    # Final test eval with predictions for the confusion matrix
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, return_predictions=True,
    )
    logger.info(f"Final test accuracy: {test_acc:.2f}% "
                f"(reference AlexNet on CIFAR-10: ~85-90%)")

    if history is not None:
        plot_history(history, "AlexNet_CIFAR10", RESULTS_DIR)
    plot_confusion_matrix(labels, preds, CIFAR10_CLASSES,
                          "AlexNet_CIFAR10", RESULTS_DIR)
    plot_confusion_matrix(labels, preds, CIFAR10_CLASSES,
                          "AlexNet_CIFAR10", RESULTS_DIR, normalize=True)

    # Random sample grid: shows actual images with true vs predicted labels.
    sample_imgs, sample_true, sample_pred = sample_test_predictions(
        model, test_loader.dataset, device, n_samples=16,
    )
    plot_sample_predictions(
        sample_imgs, sample_true, sample_pred,
        class_names=CIFAR10_CLASSES, model_name="AlexNet_CIFAR10",
        results_dir=RESULTS_DIR, n_samples=16,
        mean=CIFAR10_MEAN, std=CIFAR10_STD,
    )

    logger.info("Part 1 finished.")
    return history, test_acc
