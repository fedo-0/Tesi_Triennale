import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

# carica la lista delle feature divise per tipo
def load_dataset_config(config_path="config/dataset.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

def frequency_encoding_fit(df_train, categorical_columns, max_levels=32):
    """
    Calcola la mappatura frequency encoding basandosi solo sul training set
    """
    freq_mappings = {}
    
    for col in categorical_columns:
        if col in df_train.columns:
            # Calcola la frequenza dei valori nella colonna del training set
            value_counts = df_train[col].value_counts()
            
            # Prende solo i primi max_levels valori più frequenti
            top_values = value_counts.nlargest(max_levels).index.tolist()
            
            # Crea la mappatura: il valore più frequente -> 1, secondo -> 2, ..., max_levels -> max_levels
            freq_map = {val: i + 1 for i, val in enumerate(top_values)}
            freq_mappings[col] = freq_map
            
            print(f"Frequency encoding mappatura creata per colonna: {col} ({len(freq_map)} valori)")
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    
    return freq_mappings

def frequency_encoding_transform(df, freq_mappings):
    """
    Applica la mappatura frequency encoding a un dataset
    """
    df_encoded = df.copy()
    
    for col, freq_map in freq_mappings.items():
        if col in df_encoded.columns:
            # Mappa i valori presenti in freq_map, tutti gli altri diventano 0
            df_encoded[col] = df_encoded[col].map(lambda x: freq_map.get(x, 0))
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    
    return df_encoded

def log1p_transform(df, numeric_columns):
    df_transformed = df.copy()
    
    for col in numeric_columns:
        if col in df_transformed.columns:
            # Applica log1p solo a valori non negativi, tutti i negativi vengono trasformati in 0
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
        else:
            print(f"Attenzione: colonna {col} non trovata nel dataset")
    
    return df_transformed

def minmax_scaling_fit(df_train, numeric_columns):
    """
    Fit dello scaler solo sul training set
    """
    scaler = MinMaxScaler()
    
    # Seleziona solo le colonne numeriche presenti nel dataset
    numeric_cols_present = [col for col in numeric_columns if col in df_train.columns]
    
    if numeric_cols_present:
        # Fit dello scaler solo sui dati di training
        scaler.fit(df_train[numeric_cols_present])
        print(f"MinMax scaler fitted su {len(numeric_cols_present)} colonne numeriche del training set")
    
    return scaler, numeric_cols_present

def minmax_scaling_transform(df, scaler, numeric_cols_present):
    """
    Applica il transform usando lo scaler già fittato
    """
    df_scaled = df.copy()
    
    if numeric_cols_present:
        # Applica il transform
        df_scaled[numeric_cols_present] = scaler.transform(df_scaled[numeric_cols_present])
    
    return df_scaled

def preprocess_dataset(dataset_path, config_path, output_dir):
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

    if len(df) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    
    # Verifica che la colonna target sia presente
    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset!")
    
    # Separa features e target
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"\n--- SUDDIVISIONE DATASET (TEMPORALE) ---")
    # Split sequenziale per preservare l'ordine temporale con proporzioni 80-10-10
    total_rows = len(X)
    
    # Calcola gli indici di divisione per ottenere esattamente 80-10-10
    train_size = int(total_rows * 0.8)
    val_size = int(total_rows * 0.1)
    # Il test set prende il resto per assicurarsi di non perdere righe
    
    train_end = train_size
    val_end = train_end + val_size
    # test_end = total_rows (implicitamente)
    
    # Split sequenziale mantenendo l'ordine temporale
    X_train = X.iloc[:train_end].reset_index(drop=True).copy()
    X_val = X.iloc[train_end:val_end].reset_index(drop=True).copy()
    X_test = X.iloc[val_end:].reset_index(drop=True).copy()
    
    y_train = y.iloc[:train_end].reset_index(drop=True).copy()
    y_val = y.iloc[train_end:val_end].reset_index(drop=True).copy()
    y_test = y.iloc[val_end:].reset_index(drop=True).copy()
    
    print(f"Training set: {X_train.shape[0]} righe ({X_train.shape[0]/total_rows*100:.1f}%) - Righe: 0 a {train_end-1}")
    print(f"Validation set: {X_val.shape[0]} righe ({X_val.shape[0]/total_rows*100:.1f}%) - Righe: {train_end} a {val_end-1}")
    print(f"Test set: {X_test.shape[0]} righe ({X_test.shape[0]/total_rows*100:.1f}%) - Righe: {val_end} a {total_rows-1}")
    
    # Verifica che la somma sia corretta
    total_check = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"Verifica totale righe: {total_check} (dovrebbe essere {total_rows})")
    
    print(f"\nInizio preprocessing delle features...")
    
    # 1. Frequency encoding - FIT solo sul training set
    print("\n--- FREQUENCY ENCODING ---")
    freq_mappings = frequency_encoding_fit(X_train, categorical_columns)
    
    # Applica frequency encoding a tutti i set usando la mappatura del training
    X_train_encoded = frequency_encoding_transform(X_train, freq_mappings)
    X_val_encoded = frequency_encoding_transform(X_val, freq_mappings)
    X_test_encoded = frequency_encoding_transform(X_test, freq_mappings)
    print("Frequency encoding applicato a train, validation e test set")
    
    # 2. Trasformazione log1p per colonne numeriche
    print("\n--- TRASFORMAZIONE LOG1P ---")
    X_train_log = log1p_transform(X_train_encoded, numeric_columns)
    X_val_log = log1p_transform(X_val_encoded, numeric_columns)
    X_test_log = log1p_transform(X_test_encoded, numeric_columns)
    print("Trasformazione log1p applicata a train, validation e test set")
    
    # 3. MinMax scaling - FIT solo sul training set
    print("\n--- MINMAX SCALING ---")
    scaler, numeric_cols_present = minmax_scaling_fit(X_train_log, numeric_columns)
    
    # Applica scaling a tutti i set usando lo scaler fittato sul training
    X_train_scaled = minmax_scaling_transform(X_train_log, scaler, numeric_cols_present)
    X_val_scaled = minmax_scaling_transform(X_val_log, scaler, numeric_cols_present)
    X_test_scaled = minmax_scaling_transform(X_test_log, scaler, numeric_cols_present)
    print("MinMax scaling applicato a train, validation e test set")
    
    # Ricombina features e target per ogni set
    df_train = X_train_scaled.copy()
    df_train[target_column] = y_train.reset_index(drop=True)
    
    df_val = X_val_scaled.copy()
    df_val[target_column] = y_val.reset_index(drop=True)
    
    df_test = X_test_scaled.copy()
    df_test[target_column] = y_test.reset_index(drop=True)
    
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Definisci i percorsi di output
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    # Salva i dataset preprocessati
    print(f"\nSalvataggio dataset preprocessati in: {output_dir}")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Dataset salvati con successo!")
    print(f"- Training: {train_path} ({df_train.shape[0]} righe, {df_train.shape[1]} colonne)")
    print(f"- Validation: {val_path} ({df_val.shape[0]} righe, {df_val.shape[1]} colonne)")
    print(f"- Test: {test_path} ({df_test.shape[0]} righe, {df_test.shape[1]} colonne)")
    
    # Stampa statistiche finali
    print(f"\n--- STATISTICHE FINALI ---")
    print(f"Distribuzione target nel training set:")
    print(df_train[target_column].value_counts().sort_index())
    print(f"\nDistribuzione target nel validation set:")
    print(df_val[target_column].value_counts().sort_index())
    print(f"\nDistribuzione target nel test set:")
    print(df_test[target_column].value_counts().sort_index())
    
    return df_train, df_val, df_test, scaler, freq_mappings

if __name__ == "__main__":

    preprocess_dataset(
        dataset_path="resources/datasets/NF-UNSW-NB15-v3.csv",
        config_path="config/dataset.json",
        output_dir="resources/datasets"
    )
