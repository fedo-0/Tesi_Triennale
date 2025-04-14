import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetworkFlowDataset(Dataset):
    def __init__(self, df, numeric_columns, categorical_columns, target_column, scaler=None, encoders=None):
        """
        df: DataFrame contenente i dati.
        numeric_columns: lista dei nomi delle colonne numeriche.
        categorical_columns: lista delle colonne categoriali.
        target_column: nome della colonna target (es. "Label").
        scaler: scaler addestrato (se disponibile) per le feature numeriche.
        encoders: dizionario di encoder (se già addestrati) per le feature categoriali.
        """
        self.df = df.copy()
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

        # Gestione dei valori mancanti (se necessario)
        self.df = self.df.dropna(subset=numeric_columns + [target_column])

        # Normalizzazione delle feature numeriche
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])
        else:
            self.scaler = scaler
            self.df[numeric_columns] = self.scaler.transform(self.df[numeric_columns])

        # Codifica delle feature categoriali
        self.encoders = {}
        for col in categorical_columns:
            if encoders is None or col not in encoders:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
            else:
                self.df[col] = encoders[col].transform(self.df[col].astype(str))
                self.encoders[col] = encoders[col]

        # Target: assumiamo che "Label" sia la colonna target
        self.targets = torch.tensor(self.df[target_column].values, dtype=torch.long)
        self.numeric_data = torch.tensor(self.df[numeric_columns].values, dtype=torch.float32)

        # Elaborazione delle feature categoriali: salviamo ciascuna come tensore
        self.categorical_data = {}
        for col in categorical_columns:
            self.categorical_data[col] = torch.tensor(self.df[col].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        numeric_features = self.numeric_data[idx]  # tensor di dimensione [num_numeric]
        cat_features = {col: self.categorical_data[col][idx] for col in self.categorical_columns}
        target = self.targets[idx]
        return numeric_features, cat_features, target

def load_dataset(config_path="dataset.json", csv_path="resources/datasets/NF-UNSW-NB15-v3.csv"):
    """
    Carica il file di configurazione JSON e il file CSV del dataset reale.
    Restituisce:
      - df: il DataFrame con i dati
      - numeric_columns: lista di colonne numeriche
      - categorical_columns: lista di colonne categoriali
      - target_column: nome della colonna target
    """
    # Caricamento della configurazione dal file JSON
    with open(config_path, "r") as f:
        config = json.load(f)
    dataset_config = config["dataset"]
    numeric_columns = dataset_config["numeric_columns"]
    categorical_columns = dataset_config["categorical_columns"]
    target_column = dataset_config["target_column"]

    # Carica il dataset dal file CSV
    df = pd.read_csv(csv_path)
    return df, numeric_columns, categorical_columns, target_column

# Test standalone
if __name__ == "__main__":
    df, num_cols, cat_cols, target_col = load_dataset()
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    print("Target column:", target_col)
    print("Prime righe del dataset:\n", df.head())
