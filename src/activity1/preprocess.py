import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
from src.activity1.config import DATA_PATH, TEST_SIZE, RANDOM_SEED
from src.shared.logger import logger
 
 
def load_and_preprocess_data(path=DATA_PATH):
    """
    Load up the data, do some cleaning, scale everything, 
    and wrap it in PyTorch tensors so we're ready to train.
    """
    logger.info(f"Grabbing data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded! Shape: {df.shape}")
 
    # Grab the target column ('charges') and keep the rest as features (X)
    X = df.drop("charges", axis=1)
    y = df["charges"]
 
    # Switch categorical text strings into dummy columns (0/1)
    # drop_first=True stops columns from being redundant
    X = pd.get_dummies(X, drop_first=True)
    logger.debug(f"New column list: {list(X.columns)}")
 
    # Standard 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    # add validation

    # Scaling everything so we're centered at 0 with 1 std dev.
    # We only fit on train to avoid cheating (data leakage)!
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
 
    # Convert everything to float32 tensors for PyTorch
    X_train = torch.tensor(X_train,        dtype=torch.float32)
    X_test  = torch.tensor(X_test,         dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)
 
    logger.info("Preprocessing finished. Tensors are ready.")
 
    return X_train, X_test, y_train, y_test