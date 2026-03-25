import torch
import torch.nn as nn


# SNN

class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # PRIMERA CAPA QUE TRANSFORMA EL INPUT_DIM A 32 NEURONAS
        self.hidden = nn.Linear(input_dim, 32)

        # FUNCION DE ACTIVACION RELU PARA QUE LA RED APRENDA PATRONES COMPLEJOS
        self.relu = nn.ReLU()

        # CAPA DE SALIDA QUE ES 1 YA QUE ES UN PROBLEMA DE REGRESION
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # LA ENTRADA PASA POR LA CAPA OCULTA
        x = self.hidden(x)

        # APLICAMOS LA FUNCION DE ACTIVACION RELU
        x = self.relu(x)

        # GENERAMOS LA PREDICCION FINAL
        x = self.output(x)

        return x


# DNN

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # RED MAS PROFUNDA QUE SNN
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # 64 NEURONAS
            nn.ReLU(),                 # ACTIVACION

            nn.Linear(64, 32),         # 64 -> 32
            nn.ReLU(),

            nn.Linear(32, 16),         # 32 -> 16
            nn.ReLU(),

            nn.Linear(16, 1)           # SALIDA EN 1
        )

    def forward(self, x):
        # NN.SEQUENTIAL APLICA TODAS LAS CAPAS EN ORDEN
        return self.network(x)