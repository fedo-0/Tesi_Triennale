import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset_config(config_path="config/dataset.json"):
    """Carica la configurazione del dataset"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

def frequency_encoding_fit(df_train, categorical_columns, max_levels=32):
    """Calcola la mappatura frequency encoding basandosi solo sul training set"""
    freq_mappings = {}
    
    for col in categorical_columns:
        if col in df_train.columns:
            value_counts = df_train[col].value_counts()
            top_values = value_counts.nlargest(max_levels).index.tolist()
            freq_map = {val: i + 1 for i, val in enumerate(top_values)}
            freq_mappings[col] = freq_map
    
    return freq_mappings

def frequency_encoding_transform(df, freq_mappings):
    """Applica la mappatura frequency encoding a un dataset"""
    df_encoded = df.copy()
    
    for col, freq_map in freq_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(lambda x: freq_map.get(x, 0))
    
    return df_encoded

def standard_scaling_fit(df_train, numeric_columns):
    """Fit dello StandardScaler solo sul training set"""
    scaler = StandardScaler()
    numeric_cols_present = [col for col in numeric_columns if col in df_train.columns]
    
    if numeric_cols_present:
        scaler.fit(df_train[numeric_cols_present])
    
    return scaler, numeric_cols_present

def standard_scaling_transform(df, scaler, numeric_cols_present):
    """Applica il transform usando lo StandardScaler già fittato"""
    df_scaled = df.copy()
    
    if numeric_cols_present:
        df_scaled[numeric_cols_present] = scaler.transform(df_scaled[numeric_cols_present])
    
    return df_scaled

def create_multiclass_encoding(df_train, attack_col='Attack', label_col='Label'):
    """Crea l'encoding per le classi multiclass basandosi solo sul training set"""
    print(f"\n--- CREAZIONE ENCODING MULTICLASS ---")
    
    unique_classes = df_train[attack_col].unique()
    n_classes = len(unique_classes)
    
    print(f"Classi trovate nel training set: {n_classes}")
    
    # Ordina le classi per frequenza
    class_counts = df_train[attack_col].value_counts()
    sorted_classes = class_counts.index.tolist()
    
    # Crea LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted_classes)
    
    # Mostra mapping
    print(f"\nMapping classi (ordinate per frequenza):")
    class_mapping = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_mapping[class_name] = i
        class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
        frequency = class_counts[class_name]
        print(f"  {class_name} → {i} ({class_type}) - {frequency:,} campioni")
    
    return label_encoder, class_mapping

def apply_multiclass_encoding(df, label_encoder, attack_col='Attack'):
    """Applica l'encoding multiclass a un dataset"""
    df_encoded = df.copy()
    
    known_classes = set(label_encoder.classes_)
    df_classes = set(df[attack_col].unique())
    unknown_classes = df_classes - known_classes
    
    if unknown_classes:
        print(f"⚠️  Classi non viste nel training: {unknown_classes}")
        most_frequent_class = label_encoder.classes_[0]
        df_encoded[attack_col] = df_encoded[attack_col].apply(
            lambda x: most_frequent_class if x in unknown_classes else x
        )
    
    df_encoded['multiclass_target'] = label_encoder.transform(df_encoded[attack_col])
    
    return df_encoded

def analyze_processed_distribution(df, label_encoder, set_name="Dataset"):
    """Analizza la distribuzione delle classi nel dataset processato"""
    print(f"\n--- ANALISI DISTRIBUZIONE {set_name.upper()} ---")
    
    total_samples = len(df)
    value_counts = df['multiclass_target'].value_counts().sort_index()
    
    print(f"Totale campioni: {total_samples:,}")
    print(f"Classi trovate: {len(value_counts)}")
    
    print(f"\nDistribuzione classi processate:")
    for class_idx, count in value_counts.items():
        class_name = label_encoder.classes_[class_idx]
        percentage = (count / total_samples) * 100
        class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
        print(f"  {class_name} ({class_idx}): {count:,} ({percentage:.2f}%) - {class_type}")
    
    return value_counts

def preprocess_dataset_multiclass(clean_split_dir, config_path, output_dir,
                                          label_col='Label', attack_col='Attack'):
    """
    Preprocessing MLP per dataset multiclasse partendo da split già puliti
    Input: train.csv, val.csv, test.csv già puliti da split.py
    Output: dataset processati pronti per MLP con encoding e scaling
    """
    
    print("PREPROCESSING MULTICLASSE PER MLP")
    print("=" * 50)
    print("Input: Dataset già puliti e divisi da split.py")
    print("Output: Dataset processati per architettura MLP")
    
    # Carica configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    ip_columns = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    categorical_columns = [col for col in categorical_columns if col not in ip_columns]
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    
    # === FASE 1: CARICAMENTO DATASET PULITI ===
    print(f"\n=== FASE 1: CARICAMENTO DATASET PULITI ===")
    
    train_path = os.path.join(clean_split_dir, "train.csv")
    val_path = os.path.join(clean_split_dir, "val.csv")
    test_path = os.path.join(clean_split_dir, "test.csv")
    
    # Verifica esistenza file
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} non trovato: {path}")
    
    # Carica dataset
    print("Caricamento dataset...")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    print(f"- Training: {train_data.shape[0]:,} campioni, {train_data.shape[1]} colonne")
    print(f"- Validation: {val_data.shape[0]:,} campioni, {val_data.shape[1]} colonne")
    print(f"- Test: {test_data.shape[0]:,} campioni, {test_data.shape[1]} colonne")
    
    # Verifica colonne target
    for df_name, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if label_col not in df.columns:
            raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset {df_name}!")
        if attack_col not in df.columns:
            raise ValueError(f"Colonna attack '{attack_col}' non trovata nel dataset {df_name}!")
    
    # Identifica features disponibili
    expected_features = numeric_columns + categorical_columns
    feature_columns = [col for col in expected_features if col in train_data.columns]
    
    print(f"\nFeatures utilizzate: {len(feature_columns)} di {len(expected_features)} configurate")
    
    # === FASE 2: CREAZIONE ENCODING MULTICLASS ===
    print(f"\n=== FASE 2: CREAZIONE ENCODING MULTICLASS ===")
    
    # Crea encoding basato solo su training set
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    # Applica encoding a tutti i set
    print(f"\nApplicazione encoding...")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)
    
    # === FASE 3: SEPARAZIONE FEATURES E TARGET ===
    print(f"\n=== FASE 3: SEPARAZIONE FEATURES E TARGET ===")
    
    X_train = train_data_encoded[feature_columns].copy()
    X_val = val_data_encoded[feature_columns].copy()
    X_test = test_data_encoded[feature_columns].copy()
    
    y_train = train_data_encoded['multiclass_target'].copy()
    y_val = val_data_encoded['multiclass_target'].copy()
    y_test = test_data_encoded['multiclass_target'].copy()
    
    print(f"Features estratte: {X_train.shape[1]} colonne")
    print(f"Target estratti: multiclass_target ({len(label_encoder.classes_)} classi)")
    
    # === FASE 4: PREPROCESSING FEATURES PER MLP ===
    print(f"\n=== FASE 4: PREPROCESSING FEATURES PER MLP ===")
    
    # 1. Frequency encoding per variabili categoriche
    print("Applicazione frequency encoding...")
    freq_mappings = frequency_encoding_fit(X_train, categorical_columns)
    
    X_train_encoded = frequency_encoding_transform(X_train, freq_mappings)
    X_val_encoded = frequency_encoding_transform(X_val, freq_mappings)
    X_test_encoded = frequency_encoding_transform(X_test, freq_mappings)
    
    print(f"- Colonne categoriche processate: {len([col for col in categorical_columns if col in freq_mappings])}")
    
    # 2. Standard scaling per tutte le features (numeriche + categoriche encoded)
    print("Applicazione Standard Scaling...")
    scaler, numeric_cols_present = standard_scaling_fit(X_train_encoded, numeric_columns)
    
    # Scala anche le colonne categoriche encoded per MLP
    categorical_cols_present = [col for col in categorical_columns if col in X_train_encoded.columns]
    all_cols_to_scale = numeric_cols_present + categorical_cols_present
    
    if all_cols_to_scale:
        # Re-fit scaler su tutte le colonne (numeriche + categoriche encoded)
        scaler_full = StandardScaler()
        scaler_full.fit(X_train_encoded[all_cols_to_scale])
        
        X_train_scaled = X_train_encoded.copy()
        X_val_scaled = X_val_encoded.copy()
        X_test_scaled = X_test_encoded.copy()
        
        X_train_scaled[all_cols_to_scale] = scaler_full.transform(X_train_encoded[all_cols_to_scale])
        X_val_scaled[all_cols_to_scale] = scaler_full.transform(X_val_encoded[all_cols_to_scale])
        X_test_scaled[all_cols_to_scale] = scaler_full.transform(X_test_encoded[all_cols_to_scale])
        
        print(f"- Colonne scalate: {len(all_cols_to_scale)} (numeriche + categoriche)")
    else:
        X_train_scaled = X_train_encoded.copy()
        X_val_scaled = X_val_encoded.copy()
        X_test_scaled = X_test_encoded.copy()
        scaler_full = None
        print("- Nessuna colonna da scalare trovata")
    
    # === FASE 5: RICOMBINAZIONE E SALVATAGGIO ===
    print(f"\n=== FASE 5: RICOMBINAZIONE E SALVATAGGIO ===")
    
    # Ricombina features processate e target
    df_train_final = X_train_scaled.copy()
    df_train_final['multiclass_target'] = y_train.values
    
    df_val_final = X_val_scaled.copy()
    df_val_final['multiclass_target'] = y_val.values
    
    df_test_final = X_test_scaled.copy()
    df_test_final['multiclass_target'] = y_test.values
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva dataset processati per MLP
    train_mlp_path = os.path.join(output_dir, "train_multiclass.csv")
    val_mlp_path = os.path.join(output_dir, "val_multiclass.csv")
    test_mlp_path = os.path.join(output_dir, "test_multiclass.csv")
    
    df_train_final.to_csv(train_mlp_path, index=False)
    df_val_final.to_csv(val_mlp_path, index=False)
    df_test_final.to_csv(test_mlp_path, index=False)
    
    print(f"Dataset MLP salvati:")
    print(f"- Training: {train_mlp_path} ({df_train_final.shape[0]:,} righe)")
    print(f"- Validation: {val_mlp_path} ({df_val_final.shape[0]:,} righe)")
    print(f"- Test: {test_mlp_path} ({df_test_final.shape[0]:,} righe)")
    
    # === FASE 6: SALVATAGGIO METADATI E OGGETTI ===
    print(f"\n=== FASE 6: SALVATAGGIO METADATI E OGGETTI ===")
    
    # Salva metadati MLP
    mlp_metadata = {
        'architecture': 'MLP',
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'n_classes': len(label_encoder.classes_),
        'feature_columns': feature_columns,
        'preprocessing_applied': {
            'frequency_encoding': True,
            'standard_scaling': True,
            'categorical_columns_scaled': True
        },
        'columns_info': {
            'label_col': label_col,
            'attack_col': attack_col,
            'numeric_columns': numeric_cols_present,
            'categorical_columns': categorical_cols_present,
            'total_features': len(feature_columns)
        },
        'dataset_shapes': {
            'train': df_train_final.shape,
            'val': df_val_final.shape,
            'test': df_test_final.shape
        },
        'input_source': clean_split_dir,
        'preprocessing_version': 'mlp_multiclass_v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Salva metadati
    metadata_path = os.path.join(output_dir, "mlp_multiclass_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(mlp_metadata, f, indent=2)
    
    # Salva mappings
    mappings_path = os.path.join(output_dir, "mlp_multiclass_mappings.json")
    mappings_data = {
        'freq_mappings': freq_mappings,
        'class_mapping': class_mapping,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'scaled_columns': all_cols_to_scale
    }
    with open(mappings_path, 'w') as f:
        json.dump(mappings_data, f, indent=2)
    
    # Salva oggetti di preprocessing
    if scaler_full is not None:
        scaler_path = os.path.join(output_dir, "mlp_multiclass_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_full, f)
        print(f"- Scaler salvato: {scaler_path}")
    
    encoder_path = os.path.join(output_dir, "mlp_multiclass_label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"- Metadati: {metadata_path}")
    print(f"- Mappings: {mappings_path}")
    print(f"- Label encoder: {encoder_path}")
    
    # === STATISTICHE FINALI ===
    print(f"\n=== STATISTICHE FINALI MLP ===")
    
    # Analizza distribuzione per ogni set
    for set_name, dataset in [("Training", df_train_final), ("Validation", df_val_final), ("Test", df_test_final)]:
        analyze_processed_distribution(dataset, label_encoder, set_name)
    
    print(f"\n🎯 RIEPILOGO PREPROCESSING MLP:")
    print(f"✅ Dataset caricati da: {clean_split_dir}")
    print(f"✅ Encoding multiclasse: {len(label_encoder.classes_)} classi")
    print(f"✅ Frequency encoding: {len([col for col in categorical_columns if col in freq_mappings])} colonne categoriche")
    print(f"✅ Standard scaling: {len(all_cols_to_scale)} colonne totali")
    print(f"✅ Dataset MLP salvati in: {output_dir}")
    print(f"✅ Pronti per training architettura MLP")
    
    return df_train_final, df_val_final, df_test_final, scaler_full, freq_mappings, label_encoder, class_mapping


if __name__ == "__main__":
    df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
        clean_split_dir="resources/datasets",
        config_path="config/dataset.json",
        output_dir="resources/datasets",
        label_col='Label',
        attack_col='Attack'
    )