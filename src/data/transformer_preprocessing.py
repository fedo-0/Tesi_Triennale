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
                                 sequence_length=8,
                                 min_freq_categorical=10, max_vocab_size=10000):
    
    print("PREPROCESSING TEMPORALE PER TRANSFORMER")
    print("=" * 55)
    
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Lunghezza sequenze: {sequence_length}")
    
    # === FASE 1: CARICAMENTO DATASET ORDINATI TEMPORALMENTE ===
    print(f"\n=== FASE 1: CARICAMENTO DATASET ORDINATI ===")
    
    train_path = os.path.join(clean_split_dir, "train.csv")
    val_path = os.path.join(clean_split_dir, "val.csv")
    test_path = os.path.join(clean_split_dir, "test.csv")
    
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} non trovato: {path}")
    

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    print(f"- Training: {train_data.shape[0]:,} campioni temporali")
    print(f"- Validation: {val_data.shape[0]:,} campioni temporali")
    print(f"- Test: {test_data.shape[0]:,} campioni temporali")
    

    for df_name, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if label_col not in df.columns:
            raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset {df_name}!")
        if attack_col not in df.columns:
            raise ValueError(f"Colonna attack '{attack_col}' non trovata nel dataset {df_name}!")
    

    expected_features = numeric_columns + categorical_columns
    feature_columns = [col for col in expected_features if col in train_data.columns]
    
    print(f"\nFeatures utilizzate: {len(feature_columns)} di {len(expected_features)} configurate")
    print(f"- Numeriche: {len([col for col in numeric_columns if col in feature_columns])}")
    print(f"- Categoriche: {len([col for col in categorical_columns if col in feature_columns])}")

    feature_groups = create_feature_groups(feature_columns, numeric_columns, categorical_columns)
    
    # === FASE 2: CREAZIONE ENCODING MULTICLASS ===
    print(f"\n=== FASE 2: CREAZIONE ENCODING MULTICLASS ===")
    
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
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

    X_train_embedded = apply_embedding_mappings(X_train, embedding_mappings)
    X_val_embedded = apply_embedding_mappings(X_val, embedding_mappings)
    X_test_embedded = apply_embedding_mappings(X_test, embedding_mappings)
    
    print(f"- Vocabolari creati: {len(embedding_mappings)}")
    
    # 2. Normalizzazione features numeriche
    print("\nNormalizzazione Min-Max per features numeriche...")
    X_train_processed, X_val_processed, X_test_processed, normalization_params, numeric_cols_present = normalize_numeric_features(
        X_train_embedded, X_val_embedded, X_test_embedded, numeric_columns
    )
    
    print(f"- Features numeriche normalizzate: {len(numeric_cols_present)}")

    # === FASE 4: SALVATAGGIO DATASET PROCESSATI (SENZA SEQUENZE) ===
    print(f"\n=== FASE 4: SALVATAGGIO DATASET PROCESSATI ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_processed_path = os.path.join(output_dir, "train_transformer_processed.csv")
    val_processed_path = os.path.join(output_dir, "val_transformer_processed.csv")
    test_processed_path = os.path.join(output_dir, "test_transformer_processed.csv")
    
    train_final = pd.concat([X_train_processed, train_data_encoded[['multiclass_target']]], axis=1)
    val_final = pd.concat([X_val_processed, val_data_encoded[['multiclass_target']]], axis=1)
    test_final = pd.concat([X_test_processed, test_data_encoded[['multiclass_target']]], axis=1)
    
    train_final.to_csv(train_processed_path, index=False)
    val_final.to_csv(val_processed_path, index=False)
    test_final.to_csv(test_processed_path, index=False)
    
    print(f"Dataset processati salvati:")
    print(f"- Training: {train_processed_path} ({len(train_final):,} campioni)")
    print(f"- Validation: {val_processed_path} ({len(val_final):,} campioni)")
    print(f"- Test: {test_processed_path} ({len(test_final):,} campioni)")
    
    # === FASE 5: SALVATAGGIO METADATI ===
    print(f"\n=== FASE 5: SALVATAGGIO METADATI ===")
    
    # Metadati per Transformer con sampler
    transformer_metadata = {
        'architecture': 'Temporal_Transformer_with_Sampler',
        'temporal_config': {
            'sequence_length': sequence_length,
            'feature_dim': len(feature_columns),
            'uses_random_sampler': True,
            'sampler_type': 'RandomSlidingWindowSampler'
        },
        'dataset_info': {
            'train_samples': int(len(train_final)),
            'val_samples': int(len(val_final)),
            'test_samples': int(len(test_final)),
            'max_possible_sequences_train': max(0, len(train_final) - sequence_length + 1),
            'max_possible_sequences_val': max(0, len(val_final) - sequence_length + 1),
            'max_possible_sequences_test': max(0, len(test_final) - sequence_length + 1)
        },
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'n_classes': len(label_encoder.classes_),
        'feature_columns': feature_columns,
        'feature_groups': feature_groups,
        'preprocessing_applied': {
            'temporal_sequences': False,
            'embedding_mappings': True,
            'min_max_normalization': True,
            'order_preservation': True,
            'dynamic_sampling': True
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
            'train_csv': train_processed_path,
            'val_csv': val_processed_path,
            'test_csv': test_processed_path
        },
        'input_source': clean_split_dir,
        'preprocessing_version': 'temporal_transformer_sampler_v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, "transformer_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(transformer_metadata, f, indent=2)

    mappings_data = {
        'embedding_mappings': embedding_mappings,
        'class_mapping': class_mapping,
        'normalization_params': normalization_params,
        'vocab_stats': vocab_stats,
        'dataset_stats': {
            'train_shape': list(X_train_processed.shape),
            'val_shape': list(X_val_processed.shape),
            'test_shape': list(X_test_processed.shape)
        }
    }
    mapping_path = "resources/datasets/transformer_mappings.json"
    with open(mapping_path, 'w') as f:
        json.dump(mappings_data, f, indent=2)

    # Label encoder (invariato)
    encoder_path = os.path.join(output_dir, "transformer_label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"- Metadati: {metadata_path}")
    print(f"- Mappings: {mapping_path}")
    print(f"- Label encoder: {encoder_path}")
    
    return {
        'datasets': {
            'train': train_final,
            'val': val_final,
            'test': test_final
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
        sequence_length=8,
        min_freq_categorical=10,
        max_vocab_size=10000
    )