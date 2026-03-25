import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    
    # FUNCION DE PERDIDA Y USAMOS MSELOSS YA QUE ES UN PROBLEMA DE REGRESSION
    criterion = nn.MSELoss()
    
    # ADAM ACTUALIZA LOS PESOS UTILIZANDO EL GRADIENTE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # LISTAS PARA GUARDAR LA EVOLUVION DEL LOSS
    train_losses = []
    test_losses = []

    # BUCLE PRINCIPAL
    for epoch in range(epochs):
        
   
        model.train()
        
        # FORWARD PASS
        predictions = model(X_train)
        
        # CALCULAMOS LA LOSS COMPARANDO CON MODELOS REALES LA PREDICCION
        train_loss = criterion(predictions, y_train)
        
        # PONEMOS A 0 LOS GRADIENTES ACUMULADOS YA QUE PYTORCH ACUMULA POR DEFECTO
        optimizer.zero_grad()
        
        # CALCULA LOS GRADIENTES DE LA LOSS EN BASE A LOS PESOS
        train_loss.backward()
        
        # EL OPTIMIZADOR USA LOS GRADIENTES PARA ACTUALIZAR LOS PARAMETROS
        optimizer.step()
        
        # TEST SIN LOS PESOS
        model.eval()
        
        # TORCH.NO_GRAD DESACTIVA EL CALCULO DE GRADIENTES
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        
        # GUARDAMOS LAS PERDIDAS
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # MOSTRAMOS PROGRESO, EN ESTE CASO CADA 20 EPOCH
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
    
    return train_losses, test_losses