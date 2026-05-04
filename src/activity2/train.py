import copy
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.shared.logger import logger


def train_epoch(model, loader: DataLoader, optimizer, criterion, device,
                desc: str = "Train",
                batch_scheduler=None) -> tuple[float, float]:
    """
    Single training pass.

    If `batch_scheduler` is provided (e.g. OneCycleLR), `.step()` is called
    after every batch. Returns (avg_loss, accuracy %).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader: DataLoader, criterion, device,
             return_predictions: bool = False, desc: str = "Eval"):
    """Evaluation with no gradients."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = 100.0 * correct / total

    if return_predictions:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc


def fit(model, train_loader: DataLoader, val_loader: DataLoader,
        criterion, optimizer, scheduler, device,
        epochs: int, model_label: str = "Model",
        early_stopping_patience: Optional[int] = None,
        track_best: bool = True,
        scheduler_per_batch: bool = False) -> dict:
    """
    Full training loop with progress bar, best-model tracking and early stopping.

    Args:
        scheduler_per_batch: if True, scheduler is stepped after every batch
            (OneCycleLR / CyclicLR). If False (default), it is stepped once
            per epoch with val_loss (ReduceLROnPlateau / CosineAnnealing).
        early_stopping_patience: if not None, stop after N epochs without
            improvement.
        track_best: restore the weights of the best epoch at the end.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    best_state    = None
    epochs_since_improvement = 0

    logger.info(f"--- Training {model_label} ({epochs} epochs, device={device}) ---")
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"Ep {epoch:02d}/{epochs} train",
            batch_scheduler=scheduler if scheduler_per_batch else None,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            desc=f"Ep {epoch:02d}/{epochs} val",
        )

        if scheduler is not None and not scheduler_per_batch:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if improved:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            if track_best:
                best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
            marker = " *"
        else:
            epochs_since_improvement += 1
            marker = ""

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"Val: loss {val_loss:.4f} acc {val_acc:.2f}%{marker}"
        )

        if (early_stopping_patience is not None
                and epochs_since_improvement >= early_stopping_patience):
            logger.info(
                f"Early stopping: val has not improved for "
                f"{early_stopping_patience} epochs. Stopping at epoch {epoch}."
            )
            break

    elapsed = time.time() - start
    logger.info(
        f"{model_label} trained in {elapsed/60:.1f} min. "
        f"Best epoch: {best_epoch} (val_acc {best_val_acc:.2f}%)"
    )

    if track_best and best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Weights restored to best epoch ({best_epoch}).")

    history["best_epoch"]    = best_epoch
    history["best_val_acc"]  = best_val_acc
    history["best_val_loss"] = best_val_loss
    return history
