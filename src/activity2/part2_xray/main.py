from src.shared.logger import logger

from src.activity2.config import RESULTS_DIR
from src.activity2.plots import plot_comparison
from src.activity2.part2_xray.pipeline_scratch import run as run_scratch
from src.activity2.part2_xray.pipeline_transfer import run as run_transfer


def run():
    """Train both models and compare their validation results."""
    scratch_history,  scratch_acc  = run_scratch()
    transfer_history, transfer_acc = run_transfer()

    # plot_comparison needs the history dicts; skip it if SKIP_TRAINING was set.
    if scratch_history is not None and transfer_history is not None:
        plot_comparison(
            {
                "CNN scratch":       scratch_history,
                "ResNet18 transfer": transfer_history,
            },
            results_dir=RESULTS_DIR,
            name="xray_comparison",
        )

    logger.info("=" * 50)
    logger.info("  FINAL COMPARISON — Chest X-Ray")
    logger.info("=" * 50)
    logger.info(f"  CNN from scratch:    {scratch_acc:.2f}% test accuracy")
    logger.info(f"  ResNet18 transfer:   {transfer_acc:.2f}% test accuracy")
    logger.info(f"  Delta (transfer - scratch): {transfer_acc - scratch_acc:+.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    run()
