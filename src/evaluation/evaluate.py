import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
from src.logger.logger import logger
 
 
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model and returns a dict with metrics and prediction arrays
    to be used in plotting.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
 
    y_true = y_test.cpu().numpy().flatten()
    y_pred = predictions.cpu().numpy().flatten()
 
    # ── METRICS ─────────────────────────────────────────────────────────────
 
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
 
    # MAPE — mean absolute percentage error, easy to communicate
    epsilon = 1e-8
    mape    = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
 
    # MedAE — median absolute error, robust to extreme outliers
    medae = np.median(np.abs(y_true - y_pred))
 
    # Max Error — the worst case
    max_error = np.max(np.abs(y_true - y_pred))
 
    # Residuals — used in plot_residuals
    residuals = y_true - y_pred
 
    logger.info(f"\n{'='*40}")
    logger.info(f"  {model_name} — Results")
    logger.info(f"{'='*40}")
    logger.info(f"  MAE:       {mae:>10.2f} $")
    logger.info(f"  RMSE:      {rmse:>10.2f} $")
    logger.info(f"  MedAE:     {medae:>10.2f} $")
    logger.info(f"  Max Error: {max_error:>10.2f} $")
    logger.info(f"  MAPE:      {mape:>10.2f} %")
    logger.info(f"  R²:        {r2:>10.4f}")
    logger.info(f"{'='*40}\n")
 
    return {
        "mae":       mae,
        "mse":       mse,
        "rmse":      rmse,
        "r2":        r2,
        "mape":      mape,
        "medae":     medae,
        "max_error": max_error,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "residuals": residuals,
    }
 
 
def compare_models(results_dict: dict):
    """
    Comparative table of several models.
    Usage: compare_models({"ShallowNN": shallow_results, "DeepNN": deep_results})
    """
    logger.info("\n" + "="*60)
    logger.info("  MODEL COMPARISON")
    logger.info("="*60)
    logger.info(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>8}")
    logger.info("-"*60)
 
    for name, res in results_dict.items():
        logger.info(
            f"{name:<15} "
            f"{res['mae']:>8.2f} "
            f"{res['rmse']:>8.2f} "
            f"{res['mape']:>7.2f}% "
            f"{res['r2']:>8.4f}"
        )
 
    logger.info("="*60 + "\n")