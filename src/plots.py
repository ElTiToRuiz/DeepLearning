# src/plots.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from src.config import RESULTS_DIR


# ─────────────────────────────────────────────────────────────
# LOSS CURVES
# Cuándo usarla: justo después de train_model()
# Qué te dice: si el modelo está aprendiendo bien, si hay
# overfitting (train baja pero test sube) o underfitting
# (ambas se quedan altas y no convergen)
# ─────────────────────────────────────────────────────────────
def plot_losses(train_losses, test_losses, model_name="Model"):

    fig, ax = plt.subplots(figsize=(9, 5))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(epochs, test_losses,  label="Test Loss",  color="tomato",    linewidth=2)

    # MARCAMOS EL PUNTO DONDE EL TEST LOSS FUE MÍNIMO
    # ESE ES EL EPOCH ÓPTIMO, MÁS ALLÁ EMPIEZA EL OVERFITTING
    best_epoch = int(np.argmin(test_losses)) + 1
    best_loss  = min(test_losses)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Best epoch: {best_epoch}\nLoss: {best_loss:.1f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + len(epochs) * 0.05, best_loss * 1.1),
        fontsize=9,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray")
    )

    ax.set_title(f"{model_name} — Curvas de Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# REAL VS PREDICHO
# Cuándo usarla: después de evaluate_model()
# Qué te dice: qué tan cerca están tus predicciones del valor
# real. Los puntos deberían estar pegados a la línea diagonal.
# Si están muy dispersos arriba o abajo de la línea, el modelo
# está sobreestimando o subestimando sistemáticamente en ese
# rango de precios.
# ─────────────────────────────────────────────────────────────
def plot_predictions(y_true, y_pred, model_name="Model"):

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=20, label="Predicciones")

    # LÍNEA DIAGONAL = PREDICCIÓN PERFECTA
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="tomato", linewidth=2, linestyle="--", label="Predicción perfecta")

    # LÍNEA DE TENDENCIA REAL DE TUS PREDICCIONES
    # SI SE ALEJA MUCHO DE LA DIAGONAL, HAY SESGO SISTEMÁTICO
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(y_true)
    ax.plot(x_sorted, p(x_sorted),
            color="orange", linewidth=1.5, linestyle="-", alpha=0.8, label="Tendencia real")

    ax.set_title(f"{model_name} — Real vs Predicho", fontsize=13, fontweight="bold")
    ax.set_xlabel("Precio Real ($)")
    ax.set_ylabel("Precio Predicho ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# PANEL DE RESIDUOS (4 gráficas en una)
# Cuándo usarla: después de evaluate_model(), como análisis
# más profundo después de ver el Real vs Predicho
# Qué te dice: si los errores tienen algún patrón sistemático
# que las métricas agregadas (MAE, R²) no te cuentan
# ─────────────────────────────────────────────────────────────
def plot_residuals(results: dict, model_name: str = "Model"):

    y_true    = results["y_true"]
    y_pred    = results["y_pred"]
    residuals = results["residuals"]  # y_true - y_pred

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{model_name} — Análisis de Residuos", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── GRÁFICA 1: RESIDUOS vs PREDICHOS ──────────────────────────────────
    # QUÉ BUSCAS: puntos distribuidos aleatoriamente alrededor de y=0
    # QUÉ ES MALO: si ves una curva o embudo, el modelo tiene sesgo
    # sistemático o los errores crecen con el precio (heteroscedasticidad)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, color="steelblue", s=20)
    ax1.axhline(y=0, color="tomato", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuos vs Valores Predichos", fontweight="bold")
    ax1.set_xlabel("Valor Predicho ($)")
    ax1.set_ylabel("Residuo ($)")
    ax1.grid(True, alpha=0.3)

    # ── GRÁFICA 2: HISTOGRAMA DE RESIDUOS ─────────────────────────────────
    # QUÉ BUSCAS: campana de Gauss centrada en 0
    # QUÉ ES MALO: si está sesgado a la derecha, subestimas precios altos.
    # Si está bimodal, hay dos subpoblaciones (fumadores/no fumadores)
    # que el modelo no está separando bien
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0,                    color="tomato",  linestyle="--", linewidth=1.5, label="Cero")
    ax2.axvline(x=np.mean(residuals),   color="orange",  linestyle="-",  linewidth=1.5,
                label=f"Media: {np.mean(residuals):.0f}$")
    ax2.axvline(x=np.median(residuals), color="green",   linestyle="-",  linewidth=1.5,
                label=f"Mediana: {np.median(residuals):.0f}$")
    ax2.set_title("Distribución de Residuos", fontweight="bold")
    ax2.set_xlabel("Residuo ($)")
    ax2.set_ylabel("Frecuencia")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── GRÁFICA 3: Q-Q PLOT ───────────────────────────────────────────────
    # QUÉ BUSCAS: que los puntos sigan la línea roja
    # QUÉ ES MALO: si las colas se desvían mucho, tienes más errores
    # extremos de lo esperado. Normal en datasets con outliers como este.
    from scipy import stats
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax3.scatter(osm, osr, alpha=0.5, color="steelblue", s=20)
    ax3.plot(osm, slope * np.array(osm) + intercept, color="tomato", linewidth=1.5)
    ax3.set_title(f"Q-Q Plot  (R={r:.3f})", fontweight="bold")
    ax3.set_xlabel("Cuantiles Teóricos")
    ax3.set_ylabel("Cuantiles Observados")
    ax3.grid(True, alpha=0.3)

    # ── GRÁFICA 4: ERROR ABSOLUTO POR CUARTIL DE PRECIO ───────────────────
    # QUÉ BUSCAS: que los boxplots sean similares en todos los cuartiles
    # QUÉ ES MALO: si el Q4 (precios altos) tiene errores muchísimo mayores,
    # el modelo predice bien los seguros baratos pero falla con los caros.
    # En este dataset casi seguro verás eso por culpa de los fumadores.
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    bins       = np.percentile(y_true, [0, 25, 50, 75, 100])
    bin_labels = ["Q1\n(barato)", "Q2", "Q3", "Q4\n(caro)"]
    bin_errors = [
        abs_errors[(y_true >= bins[i]) & (y_true < bins[i + 1])]
        for i in range(len(bins) - 1)
    ]
    ax4.boxplot(bin_errors, labels=bin_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax4.set_title("Error Absoluto por Cuartil de Precio", fontweight="bold")
    ax4.set_xlabel("Cuartil del Precio Real")
    ax4.set_ylabel("Error Absoluto ($)")
    ax4.grid(True, alpha=0.3)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.show()