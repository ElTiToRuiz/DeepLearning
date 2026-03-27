import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
from src.config.config import DATA_PATH, TEST_SIZE, RANDOM_SEED
from src.logger.logger import logger
 
 
def load_and_preprocess_data(path=DATA_PATH):
    """
    Loads the CSV, applies one-hot encoding, scales the features,
    and returns PyTorch tensors ready for training.
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded | Shape: {df.shape}")
 
    # SEPARATE FEATURES AND TARGET
    X = df.drop("charges", axis=1)
    y = df["charges"]
 
    # ONE-HOT ENCODING — converts text to 0/1 columns
    # drop_first=True avoids perfect multicollinearity
    X = pd.get_dummies(X, drop_first=True)
    logger.debug(f"Features after get_dummies: {list(X.columns)}")
 
    # SPLIT 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
 
    # SCALING — mean=0, std=1
    # fit_transform only on train to avoid data leakage
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
 
    # CONVERT TO TENSORS
    X_train = torch.tensor(X_train,        dtype=torch.float32)
    X_test  = torch.tensor(X_test,         dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)
 
    logger.info(f"Preprocessing OK | X_train: {X_train.shape} | X_test: {X_test.shape}")
 
    return X_train, X_test, y_train, y_test