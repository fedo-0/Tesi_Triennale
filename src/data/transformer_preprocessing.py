import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def load_dataset_config(config_path="config/dataset.json"):
    """Carica la configurazione del dataset"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

def create_embedding_mappings(df_train, categorical_columns, min_freq=10, max_vocab_size=10000):
    """
    Crea mappings per embedding delle variabili categoriche per Transformer
    """
    embedding_mappings = {}
    vocab_stats = {}
    
    for col in categorical_columns:
        if col in df_train.columns:
            # Analizza frequenze
            value_counts = df_train[col].value_counts()
            
            # Filtra valori troppo rari
            frequent_values = value_counts[value_counts >= min_freq]
            
            # Limita dimensione vocabolario se necessario
            if len(frequent_values) > max_vocab_size:
                frequent_values = frequent_values.nlargest(max_vocab_size)
            
            # Crea mapping: valore -> indice
            # 0 riservato per valori sconosciuti/rari (UNK token)
            vocab_to_idx = {'<UNK>': 0}
            for idx, (value, count) in enumerate(frequent_values.items(), 1):
                vocab_to_idx[value] = idx
            
            embedding_mappings[col] = vocab_to_idx
            
            vocab_stats[col] = {
                'vocab_size': len(vocab_to_idx),
                'original_unique_values': int(df_train[col].nunique()),
                'frequent_values_kept': len(frequent_values),
                'min_frequency_threshold': min_freq,
                'coverage': float(frequent_values.sum() / len(df_train))
            }
            
            print(f"  {col}: {vocab_stats[col]['vocab_size']} tokens "
                  f"(coverage: {vocab_stats[col]['coverage']:.2%})")
    
    return embedding_mappings, vocab_stats

def apply_embedding_mappings(df, embedding_mappings):
    """Applica i mappings per embedding alle variabili categoriche"""
    df_mapped = df.copy()
    
    for col, vocab_mapping in embedding_mappings.items():
        if col in df_mapped.columns:
            # Mappa valori conosciuti, usa 0 (<UNK>) per valori sconosciuti
            df_mapped[col] = df_mapped[col].map(vocab_mapping).fillna(0).astype(int)
    
    return df_mapped

def normalize_numeric_features(df_train, df_val, df_test, numeric_columns):
    """
    Normalizzazione Min-Max per features numeriche in Transformer
    """
    numeric_cols_present = [col for col in numeric_columns if col in df_train.columns]
    
    if not numeric_cols_present:
        return df_train.copy(), df_val.copy(), df_test.copy(), {}, numeric_cols_present
    
    # Calcola min e max dal training set
    normalization_params = {}
    
    df_train_norm = df_train.copy()
    df_val_norm = df_val.copy()
    df_test_norm = df_test.copy()
    
    for col in numeric_cols_present:
        min_val = df_train[col].min()
        max_val = df_train[col].max()
        
        # Evita divisione per zero
        if max_val == min_val:
            print(f"Colonna {col} ha valore costante: {min_val}")
            normalization_params[col] = {'min': float(min_val), 'max': float(max_val), 'range': 1.0}
            continue
        
        normalization_params[col] = {
            'min': float(min_val), 
            'max': float(max_val),
            'range': float(max_val - min_val)
        }
        
        # Applica Min-Max scaling: (x - min) / (max - min)
        df_train_norm[col] = (df_train[col] - min_val) / (max_val - min_val)
        df_val_norm[col] = (df_val[col] - min_val) / (max_val - min_val)
        df_test_norm[col] = (df_test[col] - min_val) / (max_val - min_val)
        
        # Clamp valori fuori range per val/test
        df_val_norm[col] = df_val_norm[col].clip(0, 1)
        df_test_norm[col] = df_test_norm[col].clip(0, 1)
    
    return df_train_norm, df_val_norm, df_test_norm, normalization_params, numeric_cols_present

def create_temporal_sequences(df, feature_columns, target_col, sequence_length, stride=1):
    """
    Crea sequenze temporali per il Transformer
    
    Args:
        df: DataFrame con dati ordinati temporalmente
        feature_columns: Lista delle colonne features
        target_col: Nome colonna target
        sequence_length: Lunghezza delle sequenze (es. 32, 64, 128)
        stride: Passo tra sequenze consecutive (1 = overlap massimo)
        
    Returns:
        sequences: Array (num_sequences, sequence_length, num_features)
        targets: Array (num_sequences,) - target dell'ultimo elemento di ogni sequenza
        sequence_info: Dict con informazioni sulle sequenze create
    """
    
    print(f"\n--- CREAZIONE SEQUENZE TEMPORALI ---")
    print(f"Parametri:")
    print(f"  - Lunghezza sequenza: {sequence_length}")
    print(f"  - Stride: {stride}")
    print(f"  - Features per timestep: {len(feature_columns)}")
    
    total_samples = len(df)
    
    # Calcola numero di sequenze possibili
    num_sequences = (total_samples - sequence_length) // stride + 1
    
    if num_sequences <= 0:
        raise ValueError(f"Dataset troppo piccolo per creare sequenze. "
                        f"Samples: {total_samples}, Sequence length: {sequence_length}")
    
    # Prepara arrays
    sequences = np.zeros((num_sequences, sequence_length, len(feature_columns)), dtype=np.float32)
    targets = np.zeros(num_sequences, dtype=np.int32)
    
    # Estrai features e target
    features_data = df[feature_columns].values
    targets_data = df[target_col].values
    
    # Crea sequenze
    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        
        # Sequenza: (sequence_length, num_features)
        sequences[i] = features_data[start_idx:end_idx]
        
        # Target: etichetta dell'ultimo elemento della sequenza
        targets[i] = targets_data[end_idx - 1]
    
    sequence_info = {
        'total_samples': total_samples,
        'num_sequences': num_sequences,
        'sequence_length': sequence_length,
        'num_features': len(feature_columns),
        'stride': stride,
        'coverage': f"{num_sequences * stride + sequence_length - 1}/{total_samples}",
        'utilization': (num_sequences * stride + sequence_length - 1) / total_samples
    }
    
    print(f"Sequenze create:")
    print(f"  - Totali: {num_sequences:,}")
    print(f"  - Shape: ({num_sequences}, {sequence_length}, {len(feature_columns)})")
    print(f"  - Copertura dataset: {sequence_info['coverage']} ({sequence_info['utilization']:.1%})")
    
    return sequences, targets, sequence_info

def save_temporal_sequences(sequences, targets, sequence_info, output_path, set_name):
    """
    Salva le sequenze temporali in formato efficiente
    Usa NPZ per arrays grandi (più efficiente di CSV per dati 3D)
    """
    
    # Crea il file NPZ con sequences e targets
    npz_path = output_path.replace('.csv', '.npz')
    
    np.savez_compressed(npz_path, 
                       sequences=sequences,
                       targets=targets,
                       **sequence_info)
    
    print(f"- {set_name}: {npz_path} ({sequences.shape[0]:,} sequenze)")
    
    # Crea anche un CSV con informazioni di base (per debug/ispezione)
    summary_df = pd.DataFrame({
        'sequence_id': range(len(targets)),
        'target': targets,
        'sequence_start_idx': [i * sequence_info['stride'] for i in range(len(targets))]
    })
    
    summary_csv_path = output_path.replace('.csv', '_sequences_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    return npz_path, summary_csv_path

def analyze_temporal_distribution(targets, label_encoder, set_name="Dataset"):
    """Analizza la distribuzione delle classi nelle sequenze"""
    print(f"\n--- DISTRIBUZIONE SEQUENZE {set_name.upper()} ---")
    
    unique, counts = np.unique(targets, return_counts=True)
    total_sequences = len(targets)
    
    print(f"Sequenze totali: {total_sequences:,}")
    
    for class_idx, count in zip(unique, counts):
        class_name = label_encoder.classes_[class_idx]
        percentage = (count / total_sequences) * 100
        class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
        print(f"  {class_name} ({class_idx}): {count:,} sequenze ({percentage:.2f}%) - {class_type}")

def create_multiclass_encoding(df_train, attack_col='Attack', label_col='Label'):
    """Crea l'encoding per le classi multiclass"""
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
        print(f"  {class_name} -> {i} ({class_type}) - {frequency:,} campioni")
    
    return label_encoder, class_mapping

def apply_multiclass_encoding(df, label_encoder, attack_col='Attack'):
    """Applica l'encoding multiclass a un dataset"""
    df_encoded = df.copy()
    
    known_classes = set(label_encoder.classes_)
    df_classes = set(df[attack_col].unique())
    unknown_classes = df_classes - known_classes
    
    if unknown_classes:
        print(f"Classi non viste nel training: {unknown_classes}")
        most_frequent_class = label_encoder.classes_[0]
        df_encoded[attack_col] = df_encoded[attack_col].apply(
            lambda x: most_frequent_class if x in unknown_classes else x
        )
    
    df_encoded['multiclass_target'] = label_encoder.transform(df_encoded[attack_col])
    
    return df_encoded

def create_feature_groups(feature_columns, numeric_columns, categorical_columns):

    numeric_features = [col for col in numeric_columns if col in feature_columns]
    categorical_features = [col for col in categorical_columns if col in feature_columns]
    
    feature_groups = {
        'numeric': {
            'columns': numeric_features,
            'count': len(numeric_features),
            'type': 'continuous'
        },
        'categorical': {
            'columns': categorical_features,
            'count': len(categorical_features),
            'type': 'embedded'
        }
    }
    
    print(f"Feature groups creati:")
    print(f"  - Numeriche: {len(numeric_features)}")
    print(f"  - Categoriche: {len(categorical_features)}")
    
    return feature_groups

def preprocess_dataset_transformer(clean_split_dir, config_path, output_dir,
                                 label_col='Label', attack_col='Attack',
                                 sequence_length=64, sequence_stride=1,
                                 min_freq_categorical=10, max_vocab_size=10000):
    
    print("PREPROCESSING TEMPORALE PER TRANSFORMER")
    print("=" * 55)
    
    # Carica configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Lunghezza sequenze: {sequence_length}")
    print(f"- Stride sequenze: {sequence_stride}")
    print(f"- Min freq per vocabolari: {min_freq_categorical}")
    print(f"- Max dimensione vocabolario: {max_vocab_size}")
    
    # === FASE 1: CARICAMENTO DATASET ORDINATI TEMPORALMENTE ===
    print(f"\n=== FASE 1: CARICAMENTO DATASET ORDINATI ===")
    
    train_path = os.path.join(clean_split_dir, "train.csv")
    val_path = os.path.join(clean_split_dir, "val.csv")
    test_path = os.path.join(clean_split_dir, "test.csv")
    
    # Verifica esistenza file
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} non trovato: {path}")
    
    # Carica dataset (MANTENGONO L'ORDINE TEMPORALE)
    print("Caricamento dataset ordinati temporalmente...")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    print(f"- Training: {train_data.shape[0]:,} campioni temporali")
    print(f"- Validation: {val_data.shape[0]:,} campioni temporali")
    print(f"- Test: {test_data.shape[0]:,} campioni temporali")
    
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
    print(f"- Numeriche: {len([col for col in numeric_columns if col in feature_columns])}")
    print(f"- Categoriche: {len([col for col in categorical_columns if col in feature_columns])}")

    # Crea gruppi di features
    feature_groups = create_feature_groups(feature_columns, numeric_columns, categorical_columns)
    
    # === FASE 2: CREAZIONE ENCODING MULTICLASS ===
    print(f"\n=== FASE 2: CREAZIONE ENCODING MULTICLASS ===")
    
    # Crea encoding basato solo su training set
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    # Applica encoding a tutti i set MANTENENDO L'ORDINE
    print(f"\nApplicazione encoding (preservando ordine temporale)...")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)

    # === FASE 3: PREPROCESSING FEATURES ===
    print(f"\n=== FASE 3: PREPROCESSING FEATURES ===")
    
    # Separa features dai dati encoded
    X_train = train_data_encoded[feature_columns].copy()
    X_val = val_data_encoded[feature_columns].copy()
    X_test = test_data_encoded[feature_columns].copy()
    
    # 1. Creazione mappings per embedding (variabili categoriche)
    print("\nCreazione mappings per embedding categorici...")
    embedding_mappings, vocab_stats = create_embedding_mappings(
        X_train, categorical_columns, min_freq=min_freq_categorical, max_vocab_size=max_vocab_size
    )
    
    # Applica mappings PRESERVANDO L'ORDINE
    X_train_embedded = apply_embedding_mappings(X_train, embedding_mappings)
    X_val_embedded = apply_embedding_mappings(X_val, embedding_mappings)
    X_test_embedded = apply_embedding_mappings(X_test, embedding_mappings)
    
    print(f"- Vocabolari creati: {len(embedding_mappings)}")
    
    # 2. Normalizzazione features numeriche PRESERVANDO L'ORDINE
    print("\nNormalizzazione Min-Max per features numeriche...")
    X_train_processed, X_val_processed, X_test_processed, normalization_params, numeric_cols_present = normalize_numeric_features(
        X_train_embedded, X_val_embedded, X_test_embedded, numeric_columns
    )
    
    print(f"- Features numeriche normalizzate: {len(numeric_cols_present)}")
    
    # === FASE 4: CREAZIONE SEQUENZE TEMPORALI ===
    print(f"\n=== FASE 4: CREAZIONE SEQUENZE TEMPORALI ===")
    
    # Crea sequenze per training set
    print("Training sequences:")
    train_sequences, train_seq_targets, train_seq_info = create_temporal_sequences(
        pd.concat([X_train_processed, train_data_encoded[['multiclass_target']]], axis=1),
        feature_columns, 'multiclass_target', sequence_length, sequence_stride
    )
    
    # Crea sequenze per validation set  
    print("\nValidation sequences:")
    val_sequences, val_seq_targets, val_seq_info = create_temporal_sequences(
        pd.concat([X_val_processed, val_data_encoded[['multiclass_target']]], axis=1),
        feature_columns, 'multiclass_target', sequence_length, sequence_stride
    )
    
    # Crea sequenze per test set
    print("\nTest sequences:")
    test_sequences, test_seq_targets, test_seq_info = create_temporal_sequences(
        pd.concat([X_test_processed, test_data_encoded[['multiclass_target']]], axis=1),
        feature_columns, 'multiclass_target', sequence_length, sequence_stride
    )
    
    # === FASE 5: SALVATAGGIO SEQUENZE TEMPORALI ===
    print(f"\n=== FASE 5: SALVATAGGIO SEQUENZE TEMPORALI ===")
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva sequenze in formato NPZ (efficiente per arrays 3D)
    train_npz, train_summary = save_temporal_sequences(
        train_sequences, train_seq_targets, train_seq_info,
        os.path.join(output_dir, "train_transformer.csv"), "Training"
    )
    
    val_npz, val_summary = save_temporal_sequences(
        val_sequences, val_seq_targets, val_seq_info,
        os.path.join(output_dir, "val_transformer.csv"), "Validation"
    )
    
    test_npz, test_summary = save_temporal_sequences(
        test_sequences, test_seq_targets, test_seq_info,
        os.path.join(output_dir, "test_transformer.csv"), "Test"
    )
    
    print("Dataset Transformer temporali salvati:")
    print(f"- Training NPZ: {train_npz}")
    print(f"- Validation NPZ: {val_npz}")
    print(f"- Test NPZ: {test_npz}")
    
    # === FASE 6: SALVATAGGIO METADATI ===
    print(f"\n=== FASE 6: SALVATAGGIO METADATI ===")
    
    # Metadati completi per Transformer temporale
    transformer_metadata = {
        'architecture': 'Temporal_Transformer',
        'temporal_config': {
            'sequence_length': sequence_length,
            'sequence_stride': sequence_stride,
            'feature_dim': len(feature_columns)
        },
        'dataset_info': {
            'train_sequences': int(train_seq_info['num_sequences']),
            'val_sequences': int(val_seq_info['num_sequences']),
            'test_sequences': int(test_seq_info['num_sequences']),
            'total_sequences': int(train_seq_info['num_sequences'] + val_seq_info['num_sequences'] + test_seq_info['num_sequences'])
        },
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'n_classes': len(label_encoder.classes_),
        'feature_columns': feature_columns,
        'feature_groups': feature_groups,
        'preprocessing_applied': {
            'temporal_sequences': True,
            'embedding_mappings': True,
            'min_max_normalization': True,
            'order_preservation': True
        },
        'embedding_config': {
            'min_frequency_threshold': min_freq_categorical,
            'max_vocab_size': max_vocab_size,
            'vocab_stats': vocab_stats
        },
        'normalization_config': {
            'method': 'min_max',
            'numeric_columns': numeric_cols_present,
            'params': normalization_params
        },
        'file_paths': {
            'train_npz': train_npz,
            'val_npz': val_npz,
            'test_npz': test_npz,
            'train_summary': train_summary,
            'val_summary': val_summary,
            'test_summary': test_summary
        },
        'input_source': clean_split_dir,
        'preprocessing_version': 'temporal_transformer_v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Salva metadati
    metadata_path = os.path.join(output_dir, "transformer_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(transformer_metadata, f, indent=2)
    
    # Salva mappings e parametri
    mappings_path = os.path.join(output_dir, "transformer_mappings.json")
    mappings_data = {
        'embedding_mappings': embedding_mappings,
        'class_mapping': class_mapping,
        'normalization_params': normalization_params,
        'vocab_stats': vocab_stats,
        'sequence_info': {
            'train': train_seq_info,
            'val': val_seq_info,
            'test': test_seq_info
        }
    }
    with open(mappings_path, 'w') as f:
        json.dump(mappings_data, f, indent=2)
    
    # Salva label encoder
    encoder_path = os.path.join(output_dir, "transformer_label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"- Metadati: {metadata_path}")
    print(f"- Mappings: {mappings_path}")
    print(f"- Label encoder: {encoder_path}")
    
    # === ANALISI FINALE ===
    print(f"\n=== ANALISI DISTRIBUZIONE TEMPORALE ===")
    
    # Analizza distribuzione delle sequenze
    analyze_temporal_distribution(train_seq_targets, label_encoder, "Training")
    analyze_temporal_distribution(val_seq_targets, label_encoder, "Validation")  
    analyze_temporal_distribution(test_seq_targets, label_encoder, "Test")
    
    print(f"\nRIEPILOGO PREPROCESSING TEMPORALE:")
    print(f"Dataset caricati da: {clean_split_dir}")
    print(f"Encoding multiclasse: {len(label_encoder.classes_)} classi")
    print(f"Sequenze temporali create:")
    print(f"  - Training: {train_seq_info['num_sequences']:,} sequenze")
    print(f"  - Validation: {val_seq_info['num_sequences']:,} sequenze")
    print(f"  - Test: {test_seq_info['num_sequences']:,} sequenze")
    print(f"  - Lunghezza: {sequence_length} timesteps")
    print(f"  - Features per timestep: {len(feature_columns)}")
    print(f"Embedding mappings: {len(embedding_mappings)} vocabolari categorici")
    print(f"Min-Max normalization: {len(numeric_cols_present)} colonne numeriche")
    print(f"Dataset temporali salvati in: {output_dir}")
    print(f"Pronti per training Transformer temporale")
    
    return {
        'sequences': {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences
        },
        'targets': {
            'train': train_seq_targets,
            'val': val_seq_targets,
            'test': test_seq_targets
        },
        'metadata': transformer_metadata,
        'mappings': mappings_data,
        'label_encoder': label_encoder
    }


if __name__ == "__main__":
    result = preprocess_dataset_transformer(
        clean_split_dir="resources/datasets",
        config_path="config/dataset.json",
        output_dir="resources/datasets",
        label_col='Label',
        attack_col='Attack',
        sequence_length=64,  # Lunghezza sequenze temporali
        sequence_stride=1,   # Overlap massimo tra sequenze
        min_freq_categorical=10,
        max_vocab_size=10000
    )