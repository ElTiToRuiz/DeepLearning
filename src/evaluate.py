# import numpy as np
# import torch
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # FUNCIÓN DE EVALUACIÓN
# def evaluate_model(model, X_test, y_test, model_name="Model"):
    
#     # MODO EVALUACION
#     model.eval()
    
#     # DESACTIVAMOS GRADIENTES YA QUE SOLO QUEREMOS PREDECIR Y NO ENTRENAR
#     with torch.no_grad():
#         predictions = model(X_test)
    
#     # CONVERTIMOS TENSORES A ARRAYS
#     y_true = y_test.cpu().numpy()
#     y_pred = predictions.cpu().numpy()
    
#     # ERROR ABSOLUTO MEDIO
#     mae = mean_absolute_error(y_true, y_pred)
#     # ERROR CUADRATICO MEDIO
#     mse = mean_squared_error(y_true, y_pred)    
#     # RAIZ DEL MSE
#     rmse = np.sqrt(mse)
#     # DE LA VARIABILIDAD CUANTO EXPLICA EL MODELO
#     r2 = r2_score(y_true, y_pred)
    
#     # RESULTADOS
#     print(f"\n{model_name} Evaluation Metrics")
#     print("-" * 30)
#     print(f"MAE:  {mae:.4f}")
#     print(f"MSE:  {mse:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R²:   {r2:.4f}")
    
#     # DEVOLVEMOS METRISCAS Y PREDICCIONES PARA LUEGO PODER HACER LOS GRAFICOS
#     return {
#         "mae": mae,
#         "mse": mse,
#         "rmse": rmse,
#         "r2": r2,
#         "y_true": y_true,
#         "y_pred": y_pred
#     }

# src/evaluate.py

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import logger


def evaluate_model(model, X_test, y_test, model_name="Model"):

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

    y_true = y_test.cpu().numpy().flatten()
    y_pred = predictions.cpu().numpy().flatten()

    # MÉTRICAS BÁSICAS
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    # MAPE — ERROR PORCENTUAL MEDIO
    # MUY ÚTIL PARA COMUNICAR RESULTADOS A NO TÉCNICOS
    # EVITAMOS DIVIDIR POR CERO CON UN PEQUEÑO EPSILON
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # MEDAE — MEDIANA DEL ERROR ABSOLUTO
    # MÁS ROBUSTA QUE MAE ANTE OUTLIERS EXTREMOS
    medae = np.median(np.abs(y_true - y_pred))

    # MAX ERROR — EL PEOR CASO
    max_error = np.max(np.abs(y_true - y_pred))

    # RESIDUOS (DIFERENCIA ENTRE REAL Y PREDICHO)
    residuals = y_true - y_pred

    # LOGGING DE RESULTADOS
    logger.info(f"\n{'='*40}")
    logger.info(f"  {model_name} — Resultados de Evaluación")
    logger.info(f"{'='*40}")
    logger.info(f"  MAE:        {mae:>10.2f} $")
    logger.info(f"  RMSE:       {rmse:>10.2f} $")
    logger.info(f"  MedAE:      {medae:>10.2f} $")
    logger.info(f"  Max Error:  {max_error:>10.2f} $")
    logger.info(f"  MAPE:       {mape:>10.2f} %")
    logger.info(f"  R²:         {r2:>10.4f}")
    logger.info(f"{'='*40}\n")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "medae": medae,
        "max_error": max_error,
        "y_true": y_true,
        "y_pred": y_pred,
        "residuals": residuals
    }


def compare_models(results_dict: dict):
    """
    Recibe un diccionario {nombre_modelo: results} y muestra
    una tabla comparativa de todos los modelos.
    """
    logger.info("\n" + "="*60)
    logger.info("  COMPARATIVA DE MODELOS")
    logger.info("="*60)
    logger.info(f"{'Modelo':<15} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>8}")
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