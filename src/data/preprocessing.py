import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

# carica la lista delle feature divise per tipo
def load_dataset_config(config_path="config/dataset.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

def frequency_encoding(df, categorical_columns):
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            # Calcola le frequenze per ogni valore nella colonna
            freq_map = df_encoded[col].value_counts().to_dict()
            # Sostituisce i valori con le loro frequenze
            df_encoded[col] = df_encoded[col].map(freq_map)
            print(f"Frequency encoding applicato alla colonna: {col}")
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    
    return df_encoded

def log1p_transform(df, numeric_columns):
    df_transformed = df.copy()
    
    for col in numeric_columns:
        if col in df_transformed.columns:
            # Applica log1p solo a valori non negativi, tutti i negativi vengono trasormati in 0
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
            print(f"Trasformazione log1p applicata alla colonna: {col}")
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    
    return df_transformed

def minmax_scaling(df, numeric_columns):
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    
    # Seleziona solo le colonne numeriche presenti nel dataset
    numeric_cols_present = [col for col in numeric_columns if col in df_scaled.columns]
    
    if numeric_cols_present:
        # Applica MinMax scaling
        df_scaled[numeric_cols_present] = scaler.fit_transform(df_scaled[numeric_cols_present])
        print(f"MinMax scaling applicato a {len(numeric_cols_present)} colonne numeriche")
    
    return df_scaled, scaler

def preprocess_dataset(dataset_path, config_path, output_path):
    # Carica la configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    target_column = config['target_column']
    
    print(f"Configurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Colonna target: {target_column}")
    
    # Carica il dataset
    print(f"\nCaricamento dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset originale: {df.shape[0]} righe, {df.shape[1]} colonne")

    """ 
    # Rimuovi righe con valori NaN
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    print(f"Rimosse {removed_rows} righe contenenti valori NaN")
    print(f"Dataset dopo rimozione NaN: {df.shape[0]} righe, {df.shape[1]} colonne")
    """
######## START ##########

    # Gestione valori infiniti e NaN
    initial_rows = len(df)

    # Identifica colonne numeriche presenti nel dataset
    numeric_cols_present = [col for col in numeric_columns if col in df.columns]

    # Sostituisci valori infiniti con NaN nelle colonne numeriche
    if numeric_cols_present:
        inf_count = 0
        for col in numeric_cols_present:
            col_inf_count = np.isinf(df[col]).sum()
            inf_count += col_inf_count
            if col_inf_count > 0:
                print(f"⚠️  Trovati {col_inf_count} valori infiniti in colonna: {col}")
        
        df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], np.nan)
        print(f"Sostituiti {inf_count} valori infiniti con NaN")

    # Rimuovi righe con valori NaN (include sia NaN originali che ex-infiniti)
    df = df.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    print(f"Rimosse {removed_rows} righe contenenti valori NaN o infiniti")
    print(f"Dataset dopo pulizia: {df.shape[0]} righe, {df.shape[1]} colonne")

######## END ##########

    if len(df) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    
    # Verifica che la colonna target sia presente
    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset!")
    
    # Separa features e target
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"\nInizio preprocessing delle features...")
    
    # 1. Frequency encoding per colonne categoriche
    print("\n--- FREQUENCY ENCODING ---")
    X_encoded = frequency_encoding(X, categorical_columns)
    
    # 2. Trasformazione log1p per colonne numeriche
    print("\n--- TRASFORMAZIONE LOG1P ---")
    X_log_transformed = log1p_transform(X_encoded, numeric_columns)
    
    # 3. MinMax scaling per colonne numeriche
    print("\n--- MINMAX SCALING ---")
    X_scaled, scaler = minmax_scaling(X_log_transformed, numeric_columns)
    
    # Ricombina features e target
    df_preprocessed = X_scaled.copy()
    df_preprocessed[target_column] = y
    
    # Crea la directory di output se non esiste
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salva il dataset preprocessato
    print(f"\nSalvataggio dataset preprocessato in: {output_path}")
    df_preprocessed.to_csv(output_path, index=False)
    
    print(f"Dataset preprocessato salvato con successo!")
    print(f"Dimensioni finali: {df_preprocessed.shape[0]} righe, {df_preprocessed.shape[1]} colonne")
    
    # Stampa statistiche finali
    print(f"\n--- STATISTICHE FINALI ---")
    print(f"Righe processate: {len(df_preprocessed)}")
    print(f"Colonne totali: {len(df_preprocessed.columns)}")
    print(f"Distribuzione target:")
    print(df_preprocessed[target_column].value_counts().sort_index())
    
    return df_preprocessed, scaler

if __name__ == "__main__":
    preprocess_dataset()