from src.config.config import set_seeds, EPOCHS, LEARNING_RATE, OPTUNA_TRIALS
from src.logger.logger import logger
from src.data.preprocess import load_and_preprocess_data
from src.models.shallow_nn import ShallowNN
from src.models.deep_nn import DeepNN
from src.models.optuna_nn import FinalOptunaNN
from src.training.train import train_model
from src.training.optuna_tuning import tune_with_optuna
from src.evaluation.evaluate import evaluate_model, compare_models
from src.evaluation.plots import plot_losses, plot_predictions, plot_residuals
from src.utils.utils import save_model


def run():
    # 1. Start by fixing the seeds
    set_seeds()

    # 2. Get the data ready
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_dim = X_train.shape[1]

    # 3. Train the simple baseline (Shallow NN)
    logger.info("--- Training ShallowNN ---")
    shallow_model = ShallowNN(input_dim)
    shallow_train_losses, shallow_test_losses = train_model(
        shallow_model, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    save_model(shallow_model, "shallow_nn")
    shallow_results = evaluate_model(shallow_model, X_test, y_test, "ShallowNN")

    plot_losses(shallow_train_losses, shallow_test_losses, "ShallowNN")
    plot_predictions(shallow_results["y_true"], shallow_results["y_pred"], "ShallowNN")
    plot_residuals(shallow_results, "ShallowNN")

    # 4. Try something bigger (Deep NN)
    logger.info("--- Training DeepNN ---")
    deep_model = DeepNN(input_dim)
    deep_train_losses, deep_test_losses = train_model(
        deep_model, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    save_model(deep_model, "deep_nn")
    deep_results = evaluate_model(deep_model, X_test, y_test, "DeepNN")

    plot_losses(deep_train_losses, deep_test_losses, "DeepNN")
    plot_predictions(deep_results["y_true"], deep_results["y_pred"], "DeepNN")
    plot_residuals(deep_results, "DeepNN")

    # 5. Quick comparison: Shallow vs Deep
    compare_models({"ShallowNN": shallow_results, "DeepNN": deep_results})

    # 6. Let Optuna find the best hyperparameters
    logger.info("--- Optuna Tuning ---")
    study = tune_with_optuna(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        input_dim=input_dim,
        n_trials=OPTUNA_TRIALS
    )

    # 7. Final model using those top results
    best = study.best_trial.params

    # Build the hidden dims list back from Optuna's trial
    hidden_dims = [
        best[f"hidden_dim_{i}"]
        for i in range(best["n_layers"])
    ]

    logger.info("--- Final Trial: Training best Optuna model ---")
    final_model = FinalOptunaNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=best["dropout_rate"]
    )
    final_train_losses, final_test_losses = train_model(
        final_model, X_train, y_train, X_test, y_test,
        epochs=best["epochs"],
        lr=best["lr"],
        weight_decay=best["weight_decay"]
    )
    save_model(final_model, "final_optuna_nn")
    final_results = evaluate_model(final_model, X_test, y_test, "FinalOptunaNN")

    plot_losses(final_train_losses, final_test_losses, "FinalOptunaNN")
    plot_predictions(final_results["y_true"], final_results["y_pred"], "FinalOptunaNN")
    plot_residuals(final_results, "FinalOptunaNN")

    # 8. Comparison of everything we tried
    compare_models({
        "ShallowNN":      shallow_results,
        "DeepNN":         deep_results,
        "FinalOptunaNN":  final_results,
    })

    logger.info("All done! Pipeline finished.")