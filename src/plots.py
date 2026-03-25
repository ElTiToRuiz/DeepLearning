import matplotlib.pyplot as plt

# GRÁFICA DE LOSS
def plot_losses(train_losses, test_losses, model_name="Model"):
    
    # FIGURA NUEVA
    plt.figure(figsize=(8, 5))
    
    # DIBUJAMOS LA LOSS DE ENTRENAMIENTO
    plt.plot(train_losses, label="Train Loss")
    
    # DIBUJAMOS LA LOSS DE TEST
    plt.plot(test_losses, label="Test Loss")
    
    # AÑADIMOS ETIQUETAS
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# GRÁFICA REAL VS PREDICHO
def plot_predictions(y_true, y_pred, model_name="Model"):
    
    # FIGURA NUEVA
    plt.figure(figsize=(7, 7))
    
    # SCATTER PLOT
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # DIBUJAMOS LINEA DIAGONAL
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    
    # AÑADIMOS ETIQUETAS
    plt.title(f"{model_name}: Real vs Predicted")
    plt.xlabel("Real Charges")
    plt.ylabel("Predicted Charges")
    plt.grid(True)
    plt.show()