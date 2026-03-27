import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
 
from src.config.config import RESULTS_DIR
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LOSS CURVES
# When: just after train_model()
# What it tells you: if the model is learning well, or if there is overfitting
# (train decreases but test increases) or underfitting (both remain high)
# ──────────────────────────────────────────────────────────────────────────────
def plot_losses(train_losses, test_losses, model_name="Model"):
 
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs  = range(1, len(train_losses) + 1)
 
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(epochs, test_losses,  label="Test Loss",  color="tomato",    linewidth=2)
 
    # MARK THE EPOCH WITH LOWEST TEST LOSS — beyond this, overfitting begins
    best_epoch = int(np.argmin(test_losses)) + 1
    best_loss  = min(test_losses)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Best epoch: {best_epoch}\nLoss: {best_loss:.1f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + len(epochs) * 0.05, best_loss * 1.1),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray")
    )
 
    ax.set_title(f"{model_name} — Loss Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# REAL VS PREDICTED
# When: after evaluate_model()
# What it tells you: how close the predictions are to the real values.
# Points should be close to the red diagonal (perfect prediction).
# If the orange trend line deviates significantly, there's a systematic bias.
# ──────────────────────────────────────────────────────────────────────────────
def plot_predictions(y_true, y_pred, model_name="Model"):
 
    fig, ax = plt.subplots(figsize=(7, 7))
 
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=20, label="Predictions")
 
    # DIAGONAL = perfect prediction
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="tomato", linewidth=2, linestyle="--", label="Perfect prediction")
 
    # REAL TREND LINE
    z        = np.polyfit(y_true, y_pred, 1)
    p        = np.poly1d(z)
    x_sorted = np.sort(y_true)
    ax.plot(x_sorted, p(x_sorted),
            color="orange", linewidth=1.5, linestyle="-", alpha=0.8, label="Real trend")
 
    ax.set_title(f"{model_name} — Real vs Predicted", fontsize=13, fontweight="bold")
    ax.set_xlabel("Real Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# RESIDUAL ANALYSIS PANEL (4 charts)
# When: after evaluate_model(), for deeper analysis
# What it tells you: if errors have systematic patterns that
# aggregate metrics (MAE, R²) don't reveal
# ──────────────────────────────────────────────────────────────────────────────
def plot_residuals(results: dict, model_name: str = "Model"):
 
    y_true    = results["y_true"]
    y_pred    = results["y_pred"]
    residuals = results["residuals"]   # y_true - y_pred
 
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{model_name} — Residual Analysis", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
 
    # ── G1: RESIDUALS vs PREDICTED ─────────────────────────────────────────
    # Looking for: random cloud around y=0
    # Bad: funnel (heteroscedasticity) or curve (systematic bias)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, color="steelblue", s=20)
    ax1.axhline(y=0, color="tomato", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs Predicted Values", fontweight="bold")
    ax1.set_xlabel("Predicted Value ($)")
    ax1.set_ylabel("Residual ($)")
    ax1.grid(True, alpha=0.3)
 
    # ── G2: RESIDUAL HISTOGRAM ───────────────────────────────────────────
    # Looking for: Gaussian bell curve centered at 0
    # Bad: right-skewed (underestimating high values) or bimodal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0,                    color="tomato", linestyle="--", linewidth=1.5, label="Zero")
    ax2.axvline(x=np.mean(residuals),   color="orange", linestyle="-",  linewidth=1.5,
                label=f"Mean: {np.mean(residuals):.0f}$")
    ax2.axvline(x=np.median(residuals), color="green",  linestyle="-",  linewidth=1.5,
                label=f"Median: {np.median(residuals):.0f}$")
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
 
    # ── G3: Q-Q PLOT ───────────────────────────────────────────────────────
    # Looking for: points following the red line
    # Bad: deviated tails = more extreme errors than expected
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax3.scatter(osm, osr, alpha=0.5, color="steelblue", s=20)
    ax3.plot(osm, slope * np.array(osm) + intercept, color="tomato", linewidth=1.5)
    ax3.set_title(f"Q-Q Plot  (R={r:.3f})", fontweight="bold")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Observed Quantiles")
    ax3.grid(True, alpha=0.3)
 
    # ── G4: ABSOLUTE ERROR BY PRICE QUARTILE ────────────────────────────
    # Looking for: similar boxplots in all quartiles
    # Bad: Q4 (expensive) with much larger errors = fails with outliers/smokers
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    bins       = np.percentile(y_true, [0, 25, 50, 75, 100])
    bin_labels = ["Q1\n(cheap)", "Q2", "Q3", "Q4\n(expensive)"]
    bin_errors = [
        abs_errors[(y_true >= bins[i]) & (y_true < bins[i + 1])]
        for i in range(len(bins) - 1)
    ]
    ax4.boxplot(bin_errors, labels=bin_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax4.set_title("Absolute Error by Price Quartile", fontweight="bold")
    ax4.set_xlabel("Real Price Quartile")
    ax4.set_ylabel("Absolute Error ($)")
    ax4.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.show()