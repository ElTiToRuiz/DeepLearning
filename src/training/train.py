import torch
import torch.nn as nn
import torch.optim as optim
 
from src.logger.logger import logger
 
 
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=200, lr=0.01, weight_decay=0.0):
    """
    Standard training loop with Adam and MSELoss.
    Returns lists of train and test losses for plotting.
    """
    # MSELoss — standard loss function for regression
    criterion = nn.MSELoss()
 
    # Adam — adjusts lr using 1st and 2nd order moments
    # weight_decay adds L2 regularization if not 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    train_losses = []
    test_losses  = []
 
    for epoch in range(epochs):
 
        # ── TRAINING ──────────────────────────────────────────────────────
        model.train()
 
        predictions = model(X_train)
        train_loss  = criterion(predictions, y_train)
 
        optimizer.zero_grad()   # clear accumulated gradients
        train_loss.backward()   # compute gradients (backprop)
        optimizer.step()        # update weights
 
        # ── EVALUATION ON TEST ───────────────────────────────────────────
        model.eval()
        with torch.no_grad():   # deactivate gradients, just predict
            test_predictions = model(X_test)
            test_loss        = criterion(test_predictions, y_test)
 
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
 
        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f}"
            )
 
    return train_losses, test_losses