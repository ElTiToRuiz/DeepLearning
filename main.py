# IMPORTAMOS LO NECESARIO
from src.preprocess import load_and_preprocess_data
from src.model import ShallowNN, DeepNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.plots import plot_losses, plot_predictions
from src.optuna_tuning import tune_with_optuna
from src.final_model import FinalOptunaNN

# CARGAMOS Y PREPROCESAMOS LOS DATOS
X_train, X_test, y_train, y_test = load_and_preprocess_data()


# COMPROBAMOS LOS TAMAÑOS
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# NUMERO DE VARIABLES DE ENTRADA INPUT QUE RECIBE LA RED EN ESTE CASO 1
input_dim = X_train.shape[1]


# ENTRENAMOS EL SNN
print("Training Shallow Neural Network...\n")

# CREAMOS EL MODELO SNN
shallow_model = ShallowNN(input_dim)

# ENTRENAMOS Y GUARDAMOS EL LOSS
shallow_train_losses, shallow_test_losses = train_model(
    model=shallow_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=200,
    lr=0.01
)
shallow_results = evaluate_model(
    model=shallow_model,
    X_test=X_test,
    y_test=y_test,
    model_name="ShallowNN"
)

# ENTRENAMOS EL DNN
print("\nTraining Deep Neural Network...\n")

# CREAMOS EL MODELO SNN
deep_model = DeepNN(input_dim)

# ENTRENAMOS Y GUARDAMOS EL LOSS
deep_train_losses, deep_test_losses = train_model(
    model=deep_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=200,
    lr=0.01
)
deep_results = evaluate_model(
    model=deep_model,
    X_test=X_test,
    y_test=y_test,
    model_name="DeepNN"
)

# MOSTRAMOS LA ULTIMA LOSS DE TRAIN Y TEST DE CADA MODELO
print("Final results...\n")

print(f"ShallowNN - Final Train Loss: {shallow_train_losses[-1]:.4f}")
print(f"ShallowNN - Final Test Loss:  {shallow_test_losses[-1]:.4f}\n")

print(f"DeepNN    - Final Train Loss: {deep_train_losses[-1]:.4f}")
print(f"DeepNN    - Final Test Loss:  {deep_test_losses[-1]:.4f}")

# GRÁFICAS
# CURVAS DE ENTRENAMIENTO
plot_losses(shallow_train_losses, shallow_test_losses, model_name="ShallowNN")
plot_losses(deep_train_losses, deep_test_losses, model_name="DeepNN")

# PREDICCION VS VALOR REAL
plot_predictions(shallow_results["y_true"], shallow_results["y_pred"], model_name="ShallowNN")
plot_predictions(deep_results["y_true"], deep_results["y_pred"], model_name="DeepNN")

# OPTUNA 
print("\nRunning Optuna hyperparameter tuning...\n")

study = tune_with_optuna(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    input_dim=input_dim,
    n_trials=20)



# MODELO FINAL
print("\nTraining Final Optuna Model...\n")

final_model = FinalOptunaNN(input_dim)

final_train_losses, final_test_losses = train_model(
    model=final_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=295,
    lr=0.09661371934765023,
    weight_decay=5.374082557775886e-05
)

final_results = evaluate_model(
    model=final_model,
    X_test=X_test,
    y_test=y_test,
    model_name="FinalOptunaNN"
)

plot_losses(final_train_losses, final_test_losses, model_name="FinalOptunaNN")
plot_predictions(final_results["y_true"], final_results["y_pred"], model_name="FinalOptunaNN")


# COMPARACION FINAL
print("\nFinal comparison of all models")
print("-" * 40)
print(f"ShallowNN     RMSE: {shallow_results['rmse']:.4f} | R²: {shallow_results['r2']:.4f}")
print(f"DeepNN        RMSE: {deep_results['rmse']:.4f} | R²: {deep_results['r2']:.4f}")
print(f"FinalOptunaNN RMSE: {final_results['rmse']:.4f} | R²: {final_results['r2']:.4f}")