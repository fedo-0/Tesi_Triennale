import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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

def log1p_transform(df, numeric_columns):
    """Applica trasformazione log1p alle colonne numeriche"""
    df_transformed = df.copy()
    
    for col in numeric_columns:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
    
    return df_transformed

def minmax_scaling_fit(df_train, numeric_columns):
    """Fit dello scaler solo sul training set"""
    scaler = MinMaxScaler()
    numeric_cols_present = [col for col in numeric_columns if col in df_train.columns]
    
    if numeric_cols_present:
        scaler.fit(df_train[numeric_cols_present])
    
    return scaler, numeric_cols_present

def minmax_scaling_transform(df, scaler, numeric_cols_present):
    """Applica il transform usando lo scaler già fittato"""
    df_scaled = df.copy()
    
    if numeric_cols_present:
        df_scaled[numeric_cols_present] = scaler.transform(df_scaled[numeric_cols_present])
    
    return df_scaled

def analyze_multiclass_distribution(df, label_col='Label', attack_col='Attack'):
    """Analizza la distribuzione delle classi multiclass"""
    print(f"\n--- ANALISI DISTRIBUZIONE MULTICLASS ---")
    
    attack_counts = df[attack_col].value_counts()
    total_samples = len(df)
    
    print(f"Totale campioni: {total_samples:,}")
    print(f"Classi uniche trovate: {len(attack_counts)}")
    
    print(f"\nDistribuzione classi:")
    for class_name, count in attack_counts.items():
        percentage = (count / total_samples) * 100
        class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
        print(f"  {class_name}: {count:,} ({percentage:.2f}%) - {class_type}")
    
    return attack_counts

def filter_rare_classes(df, attack_col='Attack', min_samples=5000):
    """
    Rimuove classi con troppo pochi campioni
    """
    print(f"\n--- FILTRAGGIO CLASSI RARE (min_samples={min_samples:,}) ---")
    
    initial_samples = len(df)
    initial_classes = df[attack_col].nunique()
    
    # Analizza distribuzione
    class_counts = df[attack_col].value_counts()
    
    # Identifica classi rare
    rare_classes = class_counts[class_counts < min_samples].index.tolist()
    keep_classes = class_counts[class_counts >= min_samples].index.tolist()
    
    print(f"Classi originali: {initial_classes}")
    print(f"Campioni originali: {initial_samples:,}")
    
    if rare_classes:
        print(f"\nClassi rimosse (< {min_samples:,} campioni):")
        removed_samples = 0
        for class_name in rare_classes:
            count = class_counts[class_name]
            removed_samples += count
            print(f"  ❌ {class_name}: {count:,} campioni")
        
        print(f"\nClassi mantenute (≥ {min_samples:,} campioni):")
        for class_name in keep_classes:
            count = class_counts[class_name]
            print(f"  ✅ {class_name}: {count:,} campioni")
        
        # Filtra il dataset
        df_filtered = df[df[attack_col].isin(keep_classes)].copy()
        df_filtered.reset_index(drop=True, inplace=True)
        
        final_samples = len(df_filtered)
        final_classes = df_filtered[attack_col].nunique()
        
        print(f"\n📊 Risultato filtraggio:")
        print(f"  Classi: {initial_classes} → {final_classes} (-{initial_classes - final_classes})")
        print(f"  Campioni: {initial_samples:,} → {final_samples:,} (-{removed_samples:,})")
        print(f"  Riduzione: {(removed_samples/initial_samples)*100:.2f}%")
        
        return df_filtered, rare_classes
    else:
        print(f"✅ Nessuna classe da rimuovere (tutte hanno ≥ {min_samples:,} campioni)")
        return df.copy(), []

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

def create_micro_windows_indices(df, label_col='Label', attack_col='Attack', 
                                min_window_size=10, max_window_size=30):
    """
    Crea micro-finestre MA salva solo gli INDICI per risparmiare memoria
    """
    print(f"\n--- CREAZIONE MICRO-FINESTRE ({min_window_size}-{max_window_size} pacchetti) ---")
    
    # Identifica attacchi con logica binaria
    if df[label_col].dtype == 'object':
        attack_mask = df[label_col] != 'Benign'
    else:
        attack_mask = df[label_col] != 0
    
    micro_windows = []
    current_pos = 0
    window_id = 0
    
    total_len = len(df)
    progress_interval = max(1, total_len // 100)  # Mostra progresso ogni 1%
    
    while current_pos < total_len:
        remaining_packets = total_len - current_pos
        max_possible_size = min(max_window_size, remaining_packets)
        
        window_size = min_window_size
        
        # Determina dimensione finestra
        lookahead_end = min(current_pos + max_possible_size, total_len)
        lookahead_attack_ratio = attack_mask.iloc[current_pos:lookahead_end].mean()
        
        if lookahead_attack_ratio == 0:
            window_size = max_possible_size
        elif lookahead_attack_ratio < 0.1:
            window_size = min(max_window_size, max_possible_size)
        else:
            window_size = min(min_window_size + 5, max_possible_size)
        
        end_pos = min(current_pos + window_size, total_len)
        
        # Calcola statistiche senza copiare i dati
        window_attack_mask = attack_mask.iloc[current_pos:end_pos]
        attack_count = window_attack_mask.sum()
        attack_ratio = attack_count / (end_pos - current_pos)
        
        # SOLO INDICI - nessuna copia dei dati
        window_info = {
            'id': window_id,
            'start_pos': current_pos,
            'end_pos': end_pos,
            'size': end_pos - current_pos,
            'attack_count': attack_count,
            'attack_ratio': attack_ratio,
            'benign_count': (end_pos - current_pos) - attack_count,
        }
        
        micro_windows.append(window_info)
        current_pos = end_pos
        window_id += 1
        
        # Mostra progresso
        if window_id % progress_interval == 0 or current_pos >= total_len:
            progress = (current_pos / total_len) * 100
            print(f"  Progresso: {progress:.1f}% ({window_id:,} finestre)")
    
    print(f"Totale micro-finestre create: {len(micro_windows)}")
    
    # Statistiche finali
    sizes = [w['size'] for w in micro_windows]
    attack_ratios = [w['attack_ratio'] for w in micro_windows]
    
    print(f"\nStatistiche Micro-Finestre:")
    print(f"  Dimensione media: {np.mean(sizes):.1f} pacchetti")
    print(f"  Range dimensioni: {min(sizes)}-{max(sizes)} pacchetti")
    print(f"  Finestre con attacchi: {sum(1 for r in attack_ratios if r > 0)} ({sum(1 for r in attack_ratios if r > 0)/len(attack_ratios)*100:.1f}%)")
    
    return micro_windows

def stratified_window_assignment(micro_windows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Assegna le micro-finestre ai set usando logica binaria"""
    print(f"\n--- ASSEGNAZIONE STRATIFICATA MICRO-FINESTRE ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}) ---")
    
    # Categorizza finestre per contenuto binario
    pure_benign = [w for w in micro_windows if w['attack_ratio'] == 0]
    low_attack = [w for w in micro_windows if 0 < w['attack_ratio'] <= 0.2]
    medium_attack = [w for w in micro_windows if 0.2 < w['attack_ratio'] <= 0.6]
    high_attack = [w for w in micro_windows if w['attack_ratio'] > 0.6]
    
    categories = {
        'pure_benign': pure_benign,
        'low_attack': low_attack, 
        'medium_attack': medium_attack,
        'high_attack': high_attack
    }
    
    print("Distribuzione categorie micro-finestre:")
    for name, windows in categories.items():
        print(f"  {name}: {len(windows)} finestre ({len(windows)/len(micro_windows)*100:.1f}%)")
    
    # Distribuzione round-robin
    train_windows, val_windows, test_windows = [], [], []
    
    for category_name, windows in categories.items():
        if len(windows) == 0:
            continue
        
        for i, window in enumerate(windows):
            cycle_pos = i % 20  # Pattern 70-15-15
            
            if cycle_pos < 14:  # 70% train
                train_windows.append(window)
            elif cycle_pos < 17:  # 15% val
                val_windows.append(window)
            else:  # 15% test
                test_windows.append(window)
    
    # Ordina per posizione temporale
    train_windows.sort(key=lambda x: x['start_pos'])
    val_windows.sort(key=lambda x: x['start_pos'])
    test_windows.sort(key=lambda x: x['start_pos'])
    
    print(f"\nTotale finestre assegnate:")
    print(f"  Train: {len(train_windows)} finestre")
    print(f"  Validation: {len(val_windows)} finestre")
    print(f"  Test: {len(test_windows)} finestre")
    
    return train_windows, val_windows, test_windows

def extract_datasets_from_indices(df, train_windows, val_windows, test_windows):
    """
    Estrae i dataset usando gli indici - MEMORY EFFICIENT
    """
    print(f"\n--- ESTRAZIONE DATASET DA INDICI ---")
    
    # Estrai tutti gli indici
    train_indices = []
    val_indices = []
    test_indices = []
    
    for w in train_windows:
        train_indices.extend(range(w['start_pos'], w['end_pos']))
    for w in val_windows:
        val_indices.extend(range(w['start_pos'], w['end_pos']))
    for w in test_windows:
        test_indices.extend(range(w['start_pos'], w['end_pos']))
    
    # Subset del DataFrame originale (molto efficiente)
    train_data = df.iloc[train_indices].copy()
    val_data = df.iloc[val_indices].copy()
    test_data = df.iloc[test_indices].copy()
    
    # Reset degli indici
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    print(f"Dataset estratti:")
    print(f"  Train: {len(train_data):,} pacchetti")
    print(f"  Validation: {len(val_data):,} pacchetti")
    print(f"  Test: {len(test_data):,} pacchetti")
    
    return train_data, val_data, test_data

def micro_window_split_multiclass(df, label_col='Label', attack_col='Attack',
                                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                                 min_window_size=10, max_window_size=30):
    """Split con micro-finestre per multiclass - MEMORY OPTIMIZED"""
    print("MICRO-WINDOW STRATIFIED TEMPORAL SPLIT - MULTICLASS")
    print("=" * 70)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Crea micro-finestre (solo indici)
    micro_windows = create_micro_windows_indices(df, label_col, attack_col, min_window_size, max_window_size)
    
    if len(micro_windows) < 3:
        raise ValueError("Troppo poche micro-finestre per creare train/val/test set")
    
    # Assegnazione stratificata
    train_windows, val_windows, test_windows = stratified_window_assignment(
        micro_windows, train_ratio, val_ratio, test_ratio
    )
    
    # Estrazione dataset usando indici
    train_data, val_data, test_data = extract_datasets_from_indices(
        df, train_windows, val_windows, test_windows
    )
    
    return train_data, val_data, test_data

def preprocess_dataset_multiclass(dataset_path, config_path, output_dir, 
                                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                                 min_window_size=10, max_window_size=30,
                                 label_col='Label', attack_col='Attack',
                                 min_samples_per_class=5000):
    """
    Funzione principale per preprocessing multiclasse con micro-finestre - MEMORY OPTIMIZED
    """
    
    print("PREPROCESSING MULTICLASSE CON MICRO-FINESTRE TEMPORALI")
    print("=" * 70)
    print("Versione: Memory-optimized per dataset di milioni di righe")
    
    # Carica configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Split ratio: {train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}")
    
    # Carica dataset
    print(f"\nCaricamento dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset originale: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    # Gestione valori infiniti e NaN
    initial_rows = len(df)
    numeric_cols_present = [col for col in numeric_columns if col in df.columns]

    if numeric_cols_present:
        inf_count = 0
        for col in numeric_cols_present:
            col_inf_count = np.isinf(df[col]).sum()
            inf_count += col_inf_count
        
        if inf_count > 0:
            print(f"Sostituiti {inf_count} valori infiniti con NaN")
            df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Rimosse {removed_rows:,} righe contenenti valori NaN o infiniti")
    print(f"Dataset dopo pulizia: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    if len(df) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    
    # Verifica colonne target
    if label_col not in df.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset!")
    if attack_col not in df.columns:
        raise ValueError(f"Colonna attack '{attack_col}' non trovata nel dataset!")
    
    # Usa solo le features dal config
    expected_features = numeric_columns + categorical_columns
    feature_columns = [col for col in expected_features if col in df.columns]
    
    print(f"\nFeatures utilizzate: {len(feature_columns)} di {len(expected_features)} configurate")

    # Rimuovo tutte le classi con meno dei sample minimi stabiliti
    df_filtered, removed_classes = filter_rare_classes(df, attack_col, min_samples=min_samples_per_class)

    if removed_classes:
        print(f"\n⚠️  Dataset filtrato: rimosse {len(removed_classes)} classi rare")
        # Aggiorna il DataFrame di lavoro
        df = df_filtered
        
    analyze_multiclass_distribution(df, label_col, attack_col)

    # Split con micro-finestre (MEMORY OPTIMIZED)
    train_data, val_data, test_data = micro_window_split_multiclass(
        df, label_col, attack_col, train_ratio, val_ratio, test_ratio, 
        min_window_size, max_window_size
    )
    
    # Libera memoria del dataset originale
    del df
    
    # Crea encoding multiclass basato solo su training set
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    # Applica encoding multiclass a tutti i set
    print(f"\n--- APPLICAZIONE ENCODING MULTICLASS ---")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)
    
    # Libera memoria dei dataset non encoded
    del train_data, val_data, test_data
    
    # Separa features e target
    X_train = train_data_encoded[feature_columns].copy()
    X_val = val_data_encoded[feature_columns].copy()
    X_test = test_data_encoded[feature_columns].copy()
    
    y_train = train_data_encoded['multiclass_target'].copy()
    y_val = val_data_encoded['multiclass_target'].copy()
    y_test = test_data_encoded['multiclass_target'].copy()
    
    # Libera memoria dei dataset encoded
    del train_data_encoded, val_data_encoded, test_data_encoded
    
    print(f"\nVerifica totale pacchetti: {len(X_train) + len(X_val) + len(X_test):,}")
    
    # Preprocessing features
    print(f"\n--- PREPROCESSING FEATURES ---")
    
    # 1. Frequency encoding
    print("Applicazione frequency encoding...")
    freq_mappings = frequency_encoding_fit(X_train, categorical_columns)
    X_train_encoded = frequency_encoding_transform(X_train, freq_mappings)
    X_val_encoded = frequency_encoding_transform(X_val, freq_mappings)
    X_test_encoded = frequency_encoding_transform(X_test, freq_mappings)
    
    del X_train, X_val, X_test
    
    # 2. Trasformazione log1p
    print("Applicazione trasformazione log1p...")
    X_train_log = log1p_transform(X_train_encoded, numeric_columns)
    X_val_log = log1p_transform(X_val_encoded, numeric_columns)
    X_test_log = log1p_transform(X_test_encoded, numeric_columns)
    
    del X_train_encoded, X_val_encoded, X_test_encoded
    
    # 3. MinMax scaling
    print("Applicazione MinMax scaling...")
    scaler, numeric_cols_present = minmax_scaling_fit(X_train_log, numeric_columns)
    X_train_scaled = minmax_scaling_transform(X_train_log, scaler, numeric_cols_present)
    X_val_scaled = minmax_scaling_transform(X_val_log, scaler, numeric_cols_present)
    X_test_scaled = minmax_scaling_transform(X_test_log, scaler, numeric_cols_present)
    
    del X_train_log, X_val_log, X_test_log
    
    # Ricombina features e target
    df_train = X_train_scaled.copy()
    df_train['multiclass_target'] = y_train.reset_index(drop=True)
    
    df_val = X_val_scaled.copy()
    df_val['multiclass_target'] = y_val.reset_index(drop=True)
    
    df_test = X_test_scaled.copy()
    df_test['multiclass_target'] = y_test.reset_index(drop=True)
    
    del X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    # Crea directory output e salva
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_multiclass.csv")
    val_path = os.path.join(output_dir, "val_multiclass.csv")
    test_path = os.path.join(output_dir, "test_multiclass.csv")
    
    print(f"\n--- SALVATAGGIO DATASET ---")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    # Salva metadati
    metadata = {
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'n_classes': len(label_encoder.classes_),
        'feature_columns': feature_columns,
        'label_col': label_col,
        'attack_col': attack_col,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'min_window_size': min_window_size,
        'max_window_size': max_window_size,
        'removed_rare_classes': removed_classes,
        'min_samples_threshold': min_samples_per_class,
        'preprocessing_version': 'multiclass_v2.0_memory_optimized'
    }
    
    metadata_path = os.path.join(output_dir, "multiclass_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    mappings_path = os.path.join(output_dir, "multiclass_mappings.json")
    mappings_data = {
        'freq_mappings': freq_mappings,
        'class_mapping': class_mapping,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns
    }
    with open(mappings_path, 'w') as f:
        json.dump(mappings_data, f, indent=2)
    
    # Salva scaler
    import pickle
    scaler_path = os.path.join(output_dir, "multiclass_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Dataset salvati:")
    print(f"- Training: {train_path} ({df_train.shape[0]:,} righe)")
    print(f"- Validation: {val_path} ({df_val.shape[0]:,} righe)")
    print(f"- Test: {test_path} ({df_test.shape[0]:,} righe)")
    
    # Statistiche finali essenziali
    print(f"\n--- STATISTICHE FINALI ---")
    
    for set_name, dataset in [("Training", df_train), ("Validation", df_val), ("Test", df_test)]:
        total = len(dataset)
        value_counts = dataset['multiclass_target'].value_counts().sort_index()
        print(f"\n{set_name} Set: {total:,} pacchetti")
        
        for class_idx, count in value_counts.items():
            class_name = label_encoder.classes_[class_idx]
            percentage = (count / total) * 100
            print(f"  {class_name} ({class_idx}): {count:,} ({percentage:.2f}%)")
    
    return df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping


if __name__ == "__main__":
    df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
        dataset_path="resources/datasets/dataset_ton_v3.csv",
        config_path="config/dataset.json",
        output_dir="resources/datasets",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_window_size=10,
        max_window_size=30,
        label_col='Label',
        attack_col='Attack',
        min_samples_per_class=5000
    )