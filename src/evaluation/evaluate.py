import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
from src.logger.logger import logger
 
 
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Run the model on test data and grab all the important metrics.
    Returns a dict with the results so we can use them for plotting.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
 
    # Flatten everything so numpy doesn't complain
    y_true = y_test.cpu().numpy().flatten()
    y_pred = predictions.cpu().numpy().flatten()
 
    # Metrics calculation
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
 
    # MAPE (percentage error) - always good for business talk
    epsilon = 1e-8
    mape    = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
 
    # MedAE - median absolute error (ignores outliers better than MAE)
    medae = np.median(np.abs(y_true - y_pred))
 
    # Max Error - just to see the worst miss
    max_error = np.max(np.abs(y_true - y_pred))
 
    # Residuals for the error distribution plots
    residuals = y_true - y_pred
 
    # Log the summary in a nice block
    logger.info(f"\n{'='*40}")
    logger.info(f"  {model_name} - Stats Summary")
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
    Little table to compare models side-by-side.
    Usage: compare_models({"ModelA": resultsA, "ModelB": resultsB})
    """
    logger.info("\n" + "="*60)
    logger.info("  SIDE-BY-SIDE MODEL COMPARISON")
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