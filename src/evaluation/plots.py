import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
 
from src.config.config import RESULTS_DIR
 
 
# ------------------------------------------------------------------------------
# LOSS CURVES
# This helps us see if the model is actually learning or just overfiting.
# If the test loss starts going up while train loss goes down, we've gone too far.
# ------------------------------------------------------------------------------
def plot_losses(train_losses, test_losses, model_name="Model"):
 
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs  = range(1, len(train_losses) + 1)
 
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(epochs, test_losses,  label="Test Loss",  color="tomato",    linewidth=2)
 
    # Find that sweet spot where test loss was at its lowest
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
 
    ax.set_title(f"{model_name} - Training Progress", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ------------------------------------------------------------------------------
# REAL VS PREDICTED
# Basically: how close did we get?
# We want the dots to be as close to the red line as possible.
# ------------------------------------------------------------------------------
def plot_predictions(y_true, y_pred, model_name="Model"):
 
    fig, ax = plt.subplots(figsize=(7, 7))
 
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=20, label="Our Predictions")
 
    # Diagonal line is the "perfect" score
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="tomato", linewidth=2, linestyle="--", label="Perfect Line")
 
    # See the actual trend of our predictions
    z        = np.polyfit(y_true, y_pred, 1)
    p        = np.poly1d(z)
    x_sorted = np.sort(y_true)
    ax.plot(x_sorted, p(x_sorted),
            color="orange", linewidth=1.5, linestyle="-", alpha=0.8, label="Actual Trend")
 
    ax.set_title(f"{model_name} - Real vs Predicted Prices", fontsize=13, fontweight="bold")
    ax.set_xlabel("Real Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ------------------------------------------------------------------------------
# RESIDUAL ANALYSIS PANEL
# Digging deeper into where we missed.
# Are we consistently high or low? Are errors worse for expensive items?
# ------------------------------------------------------------------------------
def plot_residuals(results: dict, model_name: str = "Model"):
 
    y_true    = results["y_true"]
    y_pred    = results["y_pred"]
    residuals = results["residuals"]   # the difference
 
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{model_name} - Deep Dive: Residual Analysis", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
 
    # G1: Error vs Price
    # We want a random cloud. If we see a pattern, there's something we're missing.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, color="steelblue", s=20)
    ax1.axhline(y=0, color="tomato", linestyle="--", linewidth=1.5)
    ax1.set_title("Errors across Price Range", fontweight="bold")
    ax1.set_xlabel("Predicted Price ($)")
    ax1.set_ylabel("Error ($)")
    ax1.grid(True, alpha=0.3)
 
    # G2: Histogram of errors
    # Should look like a neat bell curve centered on 0.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0,                    color="tomato", linestyle="--", linewidth=1.5, label="Perfect 0")
    ax2.axvline(x=np.mean(residuals),   color="orange", linestyle="-",  linewidth=1.5,
                label=f"Avg Error: {np.mean(residuals):.0f}$")
    ax2.axvline(x=np.median(residuals), color="green",  linestyle="-",  linewidth=1.5,
                label=f"Median Error: {np.median(residuals):.0f}$")
    ax2.set_title("How the errors are spread out", fontweight="bold")
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
 
    # G3: Q-Q Plot
    # Points on the line = normal errors. If they tail off, we have big "outlier" misses.
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax3.scatter(osm, osr, alpha=0.5, color="steelblue", s=20)
    ax3.plot(osm, slope * np.array(osm) + intercept, color="tomato", linewidth=1.5)
    ax3.set_title(f"Q-Q Plot (Quality Check, R={r:.3f})", fontweight="bold")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Observed Quantiles")
    ax3.grid(True, alpha=0.3)
 
    # G4: Absolute error by quartiles
    # Shows if we struggle more with cheap or expensive items.
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    bins       = np.percentile(y_true, [0, 25, 50, 75, 100])
    bin_labels = ["Q1 (Cheap)", "Q2", "Q3", "Q4 (Expensive)"]
    bin_errors = [
        abs_errors[(y_true >= bins[i]) & (y_true < bins[i + 1])]
        for i in range(len(bins) - 1)
    ]
    ax4.boxplot(bin_errors, labels=bin_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax4.set_title("Accuracy per Price Category", fontweight="bold")
    ax4.set_xlabel("Price Quartile")
    ax4.set_ylabel("Absolute Error ($)")
    ax4.grid(True, alpha=0.3)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.show()