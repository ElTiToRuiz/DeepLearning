# EN ESTE PY ES DONDE VAMOS A PREPARAR LOS DATOS PARA QUE LA RED NEURONAL PUEDA APRENDER
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch # type: ignore

from src.config import DATA_PATH, TEST_SIZE, RANDOM_SEED

# FUNCION DE PREPROCESADO
def load_and_preprocess_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # SEPARAMOS X E y

    X = df.drop("charges", axis=1) # VARIABLES DE ENTRADA
    y = df["charges"] # LO QUE QUEREMOS PREDECIR

    # CONVERTIMOS TEXTO A NUMEROS 0,1
    X = pd.get_dummies(X, drop_first=True)

    # DIVIDIMOS LOS DATOS EN TRAIN Y TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # AHORA VAMOS A ESCALAR LOS DATOS PARA MEDIA=0 Y STD=1
    # CON ESTO CONSEGUIMOS QUE LA RED ENTRENE MEJOR
    scaler = StandardScaler()
    
    # LO AJUSTAMOS SOLO CON TRAIN Y EVITAMOS DATA LEAKAGE
    X_train = scaler.fit_transform(X_train)
    
    # APLICAMOS AL TEST
    X_test = scaler.transform(X_test)

    #CONVERTIMOS A TENSORES YA QUE ES LO QUE NECESITA PYTORCH
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # DEVOLVEMOS LOS DATOS
    return X_train, X_test, y_train, y_test