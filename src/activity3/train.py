"""
Shared training loop for activity 3 v4.

Single-task only (the multi-task scaffolding from v3 is gone). Used by the
two LSTM pipelines (price-only and price+sentiment), which both predict
the shock label.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.shared.logger import logger


def unpack_series(batch, device):
    """Unpack (x_series, label) batches."""
    x, y = batch
    return (x.to(device, non_blocking=True),), y.to(device, non_blocking=True)


def train_epoch(model, loader, optimizer, criterion, device, unpack_fn,
                desc="Train", batch_scheduler=None):
    """Single training pass. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        inputs, labels = unpack_fn(batch, device)
        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{100.*correct/total:.1f}%")

    return total_loss / len(loader), 100.0 * correct / total


def run_eval(model, loader, criterion, device, unpack_fn,
             return_predictions=False, desc="Eval"):
    """Inference pass (model.eval() under torch.no_grad())."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, labels = unpack_fn(batch, device)
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = 100.0 * correct / total
    if return_predictions:
        return (avg_loss, acc,
                np.array(all_preds), np.array(all_probs), np.array(all_labels))
    return avg_loss, acc


evaluate = run_eval


def fit(model, train_loader, val_loader,
        criterion, optimizer, scheduler, device,
        epochs, model_label="Model",
        unpack_fn=unpack_series,
        early_stopping_patience=None,
        track_best=True,
        scheduler_per_batch=False):
    """Full training loop. Selects best model by val accuracy."""
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc": [],
        "val_f1":     [], "val_auc": [],
    }
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
            unpack_fn=unpack_fn,
            desc=f"Ep {epoch:02d}/{epochs} train",
            batch_scheduler=scheduler if scheduler_per_batch else None,
        )
        val_loss, val_acc, val_preds, val_probs, val_labels = run_eval(
            model, val_loader, criterion, device,
            unpack_fn=unpack_fn,
            return_predictions=True,
            desc=f"Ep {epoch:02d}/{epochs} val",
        )
        val_f1 = f1_score(val_labels, val_preds, average="binary", zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = float("nan")

        if scheduler is not None and not scheduler_per_batch:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

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
            f"Val: loss {val_loss:.4f} acc {val_acc:.2f}% "
            f"F1 {val_f1:.3f} AUC {val_auc:.3f}{marker}"
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
