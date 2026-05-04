import torch
import torch.nn as nn
import torch.optim as optim
 
from src.shared.logger import logger
 
 
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=200, lr=0.01, weight_decay=0.0):
    """
    Main training loop. Gets the model ready, runs epochs,
    and grabs the losses so we can plot them later.
    """
    # Stick with MSE for regression
    criterion = nn.MSELoss()
 
    # Opting for Adam because it's usually reliable
    # weight_decay adds some L2 to keep it from overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    train_losses = []
    test_losses  = []
 
    for epoch in range(epochs):
 
        # -- Step 1: Training Mode --
        model.train()
 
        predictions = model(X_train)
        train_loss  = criterion(predictions, y_train)
 
        optimizer.zero_grad()   # Get rid of old gradients
        train_loss.backward()   # Backprop time
        optimizer.step()        # Update the weights
 
        # -- Step 2: Evaluation Mode --
        model.eval()
        with torch.no_grad():   # No need to track grads here
            test_predictions = model(X_test)
            test_loss        = criterion(test_predictions, y_test)
 
        # Keep track of the progress
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
 
        # Print a status update every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f}"
            )
 
    return train_losses, test_losses