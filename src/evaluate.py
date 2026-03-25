import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# FUNCIÓN DE EVALUACIÓN
def evaluate_model(model, X_test, y_test, model_name="Model"):
    
    # MODO EVALUACION
    model.eval()
    
    # DESACTIVAMOS GRADIENTES YA QUE SOLO QUEREMOS PREDECIR Y NO ENTRENAR
    with torch.no_grad():
        predictions = model(X_test)
    
    # CONVERTIMOS TENSORES A ARRAYS
    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    # ERROR ABSOLUTO MEDIO
    mae = mean_absolute_error(y_true, y_pred)
    
    # ERROR CUADRATICO MEDIO
    mse = mean_squared_error(y_true, y_pred)
    
    # RAIZ DEL MSE
    rmse = np.sqrt(mse)
    
    # DE LA VARIABILIDAD CUANTO EXPLICA EL MODELO
    r2 = r2_score(y_true, y_pred)
    
    # RESULTADOS
    print(f"\n{model_name} Evaluation Metrics")
    print("-" * 30)
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # DEVOLVEMOS METRISCAS Y PREDICCIONES PARA LUEGO PODER HACER LOS GRAFICOS
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_true,
        "y_pred": y_pred
    }