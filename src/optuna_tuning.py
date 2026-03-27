import optuna
import torch
import torch.nn as nn
import torch.optim as optim


class OptunaNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()

        # CREAMOS LA RED
        layers = []

        # TAMAÑO
        current_dim = input_dim

        # Recorremos la lista de capas ocultas elegidas por Optuna
        for hidden_dim in hidden_dims:
            # CAPA LINEAL
            layers.append(nn.Linear(current_dim, hidden_dim))

            # RELU
            layers.append(nn.ReLU())

            # DROPOUT 
            layers.append(nn.Dropout(dropout_rate))

            # LA SALIDA DE ESTA SERA LA ENTRADA DE LA SIGUIENTE CAPA
            current_dim = hidden_dim

        # ULTIMA CAPA DE SALIDA
        layers.append(nn.Linear(current_dim, 1)) # 1 VALOR SOLO AL SER REGRESION

        # EJECUTAMOS LAS CAPAS EN ORDEN
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# TUNING
def tune_with_optuna(X_train, y_train, X_test, y_test, input_dim, n_trials=20):

    # DEFINIMOS QUE HACE OPTUNA EN CADA PRUEBA
    def objective(trial):

        # HIPERPARAMETROS QUE PROBAMOS

        # CAPAS OCULTAS ENTRE 1 Y 3
        n_layers = trial.suggest_int("n_layers", 1, 3)

        # NEURONAS POR CAPA
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 16, 128)
            hidden_dims.append(hidden_dim)

        # LEARNING RATE
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        # CON ESTO EVITAMOS OVERFITTING
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        # REGULARIZAMOS L2
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # NUMERO DE EPOCHS
        epochs = trial.suggest_int("epochs", 100, 300)

        # CREAMOS EL MODELO
        model = OptunaNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        # LOSS DE REGRESION
        criterion = nn.MSELoss()

        # ADAM
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # -------------------------
        # ENTRENAMOS
        for epoch in range(epochs):

            # MODO ENTRENAMIENTO
            model.train()

            # PREDICCIONES SOBRE TRAIN
            predictions = model(X_train)

            # CALCULAMOS LA LOSS
            train_loss = criterion(predictions, y_train)

            # LIMPIAMOS LOS GRADIENTES ACUMULADOS
            optimizer.zero_grad()

            # BACKPROPAGATION
            train_loss.backward()

            # ACTUALIZAMOS LOS PESOS
            optimizer.step()

        # EVALUAMOS
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)

        return test_loss.item()

    # MINIMIZAMOS LA LOSS
    study = optuna.create_study(direction="minimize")

    # HACEMOS LAS PRUEBAS
    study.optimize(objective, n_trials=n_trials)

    # MOSTRAMOS LOS MEJORES
    print("\nBest Optuna Trial")
    print("-" * 30)
    print("Best test loss:", study.best_trial.value)
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

    return study