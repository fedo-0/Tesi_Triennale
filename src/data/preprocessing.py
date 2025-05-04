import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetworkFlowDataset(Dataset):
    def __init__(self, df, numeric_columns, categorical_columns, target_column, scaler=None, encoders=None, balance_classes=False):
        """
        df: DataFrame contenente i dati.
        numeric_columns: lista dei nomi delle colonne numeriche.
        categorical_columns: lista delle colonne categoriali.
        target_column: nome della colonna target (es. "Label").
        scaler: scaler addestrato (se disponibile) per le feature numeriche.
        encoders: dizionario di encoder (se già addestrati) per le feature categoriali.
        balance_classes: se True, bilancia le classi mediante undersampling della classe maggioritaria
        """
        self.df = df.copy()
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.cat_encoders = {}
        self.cat_dims = {}

        # Gestione dei valori mancanti (più robusta)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Invece di rimuovere righe, imputa con la mediana per colonne numeriche
        for col in numeric_columns:
            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)

        # Verifica della presenza di tutte le colonne necessarie
        missing_cols = set(numeric_columns + categorical_columns + [target_column]) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")

        # Normalizzazione delle feature numeriche
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])
        else:
            self.scaler = scaler
            self.df[numeric_columns] = self.scaler.transform(self.df[numeric_columns])

        # Codifica delle feature categoriali
        self.encoders = encoders or {}
        self.cat_dims = {}
        for col in categorical_columns:
            if col not in self.encoders:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
            else:
                # Gestisce nuove categorie usando un valore di default
                self.df[col] = self.df[col].astype(str)
                # Mappa usando un dizionario per gestire valori sconosciuti
                mapping_dict = {val: idx for idx, val in enumerate(self.encoders[col].classes_)}
                self.df[col] = self.df[col].map(lambda x: mapping_dict.get(x, -1))  # -1 per valori sconosciuti
                # Sostituisce -1 con un valore casuale o il più frequente
                if (self.df[col] == -1).any():
                    most_common = self.df[col][self.df[col] != -1].mode()[0]
                    self.df.loc[self.df[col] == -1, col] = most_common
            
            self.cat_dims[col] = len(self.encoders[col].classes_)

        # Bilanciamento delle classi (opzionale)
        if balance_classes:
            self._balance_classes()

        # Preparazione dei dati
        self.X_numeric = torch.tensor(self.df[numeric_columns].values, dtype=torch.float32)
        self.categorical_data = {col: torch.tensor(self.df[col].values, dtype=torch.long) 
                                for col in categorical_columns}
        self.y = torch.tensor(self.df[target_column].values, dtype=torch.long)

    def _balance_classes(self):
        """Bilancia le classi tramite undersampling della classe maggioritaria"""
        class_counts = self.df[self.target_column].value_counts()
        min_class_count = class_counts.min()
        
        balanced_df = pd.DataFrame()
        for class_val, count in class_counts.items():
            class_df = self.df[self.df[self.target_column] == class_val]
            if count > min_class_count:
                class_df = class_df.sample(min_class_count, random_state=42)
            balanced_df = pd.concat([balanced_df, class_df])
        
        self.df = balanced_df.reset_index(drop=True)
        print(f"Dataset bilanciato: {self.df[self.target_column].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Restituisce le feature in maniera più efficiente"""
        numeric_features = self.X_numeric[idx]
        cat_features = {col: self.categorical_data[col][idx] for col in self.categorical_columns}
        y = self.y[idx]
        return numeric_features, cat_features, y
    
    def get_categorical_info(self):
        """Restituisce un dizionario con informazioni sulle feature categoriali"""
        return {col: (self.cat_dims[col], 
                min(16, self.cat_dims[col] // 2) if self.cat_dims[col] > 100 else 
                min(8, self.cat_dims[col] // 2 if self.cat_dims[col] > 2 else 2))
                for col in self.categorical_columns}

def load_dataset(config_path="config/dataset.json", csv_path="resources/datasets/NF-UNSW-NB15-v3.csv"):
    """
    Carica il file di configurazione JSON e il file CSV del dataset reale.
    Restituisce:
      - df: il DataFrame con i dati
      - numeric_columns: lista di colonne numeriche
      - categorical_columns: lista di colonne categoriali
      - target_column: nome della colonna target
    """
    # Caricamento della configurazione dal file JSON
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        dataset_config = config["dataset"]
        numeric_columns = dataset_config["numeric_columns"]
        categorical_columns = dataset_config["categorical_columns"]
        target_column = dataset_config["target_column"]
    except Exception as e:
        raise ValueError(f"Errore nel caricamento della configurazione: {e}")

    # Carica il dataset dal file CSV
    try:
        df = pd.read_csv(csv_path)
        
        # Verifica presenza colonne
        missing_cols = set(numeric_columns + categorical_columns + [target_column]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
            
        # Conversione tipi di dati
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rapida analisi dati
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution: {df[target_column].value_counts().to_dict()}")
        print(f"Missing values: {df[numeric_columns + categorical_columns + [target_column]].isna().sum().sum()}")
        
        return df, numeric_columns, categorical_columns, target_column
    except Exception as e:
        raise ValueError(f"Errore nel caricamento del dataset: {e}")

# Test standalone
if __name__ == "__main__":
    df, num_cols, cat_cols, target_col = load_dataset()
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    print("Target column:", target_col)
    print("Prime righe del dataset:\n", df.head())
