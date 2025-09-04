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


"""
def create_temporal_sequences(df, feature_columns, target_col, sequence_length=64, stride=1):

    Crea sequenze temporali basate sui flussi di rete per il Transformer
    Le sequenze raggruppano campioni consecutivi dello stesso mittente o destinatario
    Se il criterio non si applica, crea sequenze consecutive di 32 campioni
    
    Args:
        df: DataFrame con dati ordinati temporalmente
        feature_columns: Lista delle colonne features
        target_col: Nome colonna target
        sequence_length: Lunghezza massima delle sequenze (default 64)
        stride: Non utilizzato per flussi, usato per sequenze consecutive (default 1)
        
    Returns:
        sequences: Array (num_sequences, actual_seq_len, num_features)
        targets: Array (num_sequences,) - target dell'ultimo elemento di ogni sequenza
        sequence_info: Dict con informazioni sulle sequenze create
    
    
    print(f"\n--- CREAZIONE SEQUENZE BASATE SU FLUSSI DI RETE ---")
    print(f"Parametri:")
    print(f"  - Lunghezza massima sequenza: {sequence_length}")
    print(f"  - Features per timestep: {len(feature_columns)}")
    print(f"  - Criterio: stesso IPV4_SRC_ADDR o IPV4_DST_ADDR")
    print(f"  - Fallback: sequenze consecutive da 32 campioni")
    
    total_samples = len(df)
    
    # Estrai features e target
    features_data = df[feature_columns].values
    targets_data = df[target_col].values
    src_ips = df['IPV4_SRC_ADDR'].values
    dst_ips = df['IPV4_DST_ADDR'].values
    
    sequences_list = []
    targets_list = []
    sequence_lengths = []
    
    i = 0
    flows_processed = 0
    single_packet_flows = 0
    consecutive_sequences = 0
    fallback_sequence_length = sequence_length
    
    while i < total_samples:
        current_src = src_ips[i]
        current_dst = dst_ips[i]
        
        sequence_indices = [i]
        j = i + 1
        
        while j < total_samples and len(sequence_indices) < sequence_length:
            next_src = src_ips[j]
            next_dst = dst_ips[j]
            
            if (next_src == current_src or next_dst == current_dst or 
                next_src == current_dst or next_dst == current_src):
                sequence_indices.append(j)
                j += 1
            else:
                break
        
        if len(sequence_indices) > 15:
            sequence_features = features_data[sequence_indices]
            sequence_target = targets_data[sequence_indices[-1]]
            
            sequences_list.append(sequence_features)
            targets_list.append(sequence_target)
            sequence_lengths.append(len(sequence_indices))
            
            flows_processed += 1
            i = j if j > i + 1 else i + 1
            
        else:
            # Sequenza mono-pacchetto: crea sequenza consecutiva di 32 campioni
            single_packet_flows += 1
            
            # Verifica che ci siano abbastanza campioni per una sequenza da 32
            if i + fallback_sequence_length <= total_samples:
                consecutive_indices = list(range(i, i + fallback_sequence_length))
                sequence_features = features_data[consecutive_indices]
                sequence_target = targets_data[consecutive_indices[-1]]
                
                sequences_list.append(sequence_features)
                targets_list.append(sequence_target)
                sequence_lengths.append(fallback_sequence_length)
                
                consecutive_sequences += 1
                i += fallback_sequence_length  # Avanza di 32 posizioni
            else:
                # Non ci sono abbastanza campioni rimanenti, salta
                i += 1
    
    num_sequences = len(sequences_list)
    
    if num_sequences == 0:
        raise ValueError("Nessuna sequenza creata! Verifica i dati e i criteri di matching.")
    
    # Determina lunghezza massima effettiva
    max_actual_length = max(sequence_lengths)
    print(f"Lunghezza massima effettiva: {max_actual_length}")
    
    # Crea array padded con zeri
    sequences = np.zeros((num_sequences, max_actual_length, len(feature_columns)), dtype=np.float32)
    targets = np.zeros(num_sequences, dtype=np.int32)
    
    # Riempie gli array con padding
    for idx, (seq, target) in enumerate(zip(sequences_list, targets_list)):
        actual_length = len(seq)
        sequences[idx, :actual_length, :] = seq
        targets[idx] = target
    
    # Statistiche delle lunghezze
    length_stats = {
        'min': min(sequence_lengths),
        'max': max(sequence_lengths),
        'mean': np.mean(sequence_lengths),
        'median': np.median(sequence_lengths)
    }
    
    sequence_info = {
        'total_samples': total_samples,
        'num_sequences': num_sequences,
        'sequence_length': max_actual_length,
        'stride': stride,
        'num_features': len(feature_columns),
        'flows_processed': flows_processed,
        'single_packet_flows': single_packet_flows,
        'consecutive_sequences': consecutive_sequences,
        'fallback_sequence_length': fallback_sequence_length,
        'coverage': f"{sum(sequence_lengths)}/{total_samples}",
        'utilization': sum(sequence_lengths) / total_samples,
        'length_stats': length_stats,
        'sequence_lengths': sequence_lengths
    }
    
    print(f"Sequenze create:")
    print(f"  - Totali: {num_sequences:,}")
    print(f"  - Shape: ({num_sequences}, {max_actual_length}, {len(feature_columns)})")
    print(f"  - Flussi multi-pacchetto: {flows_processed:,}")
    print(f"  - Flussi mono-pacchetto → sequenze consecutive: {consecutive_sequences:,}")
    print(f"  - Lunghezza media: {length_stats['mean']:.1f}")
    print(f"  - Lunghezza mediana: {length_stats['median']:.1f}")
    print(f"  - Copertura dataset: {sequence_info['coverage']} ({sequence_info['utilization']:.1%})")
    
    return sequences, targets, sequence_info
"""

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
    ip_columns = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    categorical_columns = [col for col in categorical_columns if col not in ip_columns]
    
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
    ip_columns = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    
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
    
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} non trovato: {path}")
    
    print("Caricamento dataset ordinati temporalmente...")
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
    feature_columns_without_ip = [col for col in feature_columns if col not in ip_columns]
    
    print(f"\nFeatures utilizzate: {len(feature_columns)} di {len(expected_features)} configurate")
    print(f"- Numeriche: {len([col for col in numeric_columns if col in feature_columns])}")
    print(f"- Categoriche: {len([col for col in categorical_columns if col in feature_columns])}")

    # Crea gruppi di features
    feature_groups = create_feature_groups(feature_columns, numeric_columns, categorical_columns)
    
    # === FASE 2: CREAZIONE ENCODING MULTICLASS ===
    print(f"\n=== FASE 2: CREAZIONE ENCODING MULTICLASS ===")
    
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    print(f"\nApplicazione encoding...")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)

    # === FASE 3: PREPROCESSING FEATURES ===
    print(f"\n=== FASE 3: PREPROCESSING FEATURES ===")
    
    # Separa features dai dati encoded (SENZA colonne IP)
    X_train = train_data_encoded[feature_columns_without_ip].copy()
    X_val = val_data_encoded[feature_columns_without_ip].copy()
    X_test = test_data_encoded[feature_columns_without_ip].copy()
    
    # Rimuovi colonne IP dalle configurazioni
    categorical_columns_without_ip = [col for col in categorical_columns if col not in ip_columns]
    numeric_columns_without_ip = [col for col in numeric_columns if col not in ip_columns]
    
    # 1. Creazione mappings per embedding
    print("\nCreazione mappings per embedding categorici...")
    embedding_mappings, vocab_stats = create_embedding_mappings(
        X_train, categorical_columns_without_ip, min_freq=min_freq_categorical, max_vocab_size=max_vocab_size
    )
    
    # 2. Applica mappings (TRASFORMA I VALORI CATEGORICI)
    X_train_embedded = apply_embedding_mappings(X_train, embedding_mappings)
    X_val_embedded = apply_embedding_mappings(X_val, embedding_mappings)
    X_test_embedded = apply_embedding_mappings(X_test, embedding_mappings)
    
    # 3. Normalizzazione features numeriche
    print("\nNormalizzazione Min-Max per features numeriche...")
    X_train_processed, X_val_processed, X_test_processed, normalization_params, numeric_cols_present = normalize_numeric_features(
        X_train_embedded, X_val_embedded, X_test_embedded, numeric_columns_without_ip
    )
    
    # === FASE 4: CREAZIONE SEQUENZE CON DATI PROCESSATI ===
    print(f"\n=== FASE 4: CREAZIONE SEQUENZE TEMPORALI ===")
    
    # Ricombina dati processati con colonne IP e target per sequenze
    def create_data_for_sequences(processed_data, encoded_data):
        sequence_data = processed_data.copy()
        sequence_data['multiclass_target'] = encoded_data['multiclass_target']
        # Aggiungi colonne IP NON processate
        if 'IPV4_SRC_ADDR' in encoded_data.columns:
            sequence_data['IPV4_SRC_ADDR'] = encoded_data['IPV4_SRC_ADDR']
        if 'IPV4_DST_ADDR' in encoded_data.columns:
            sequence_data['IPV4_DST_ADDR'] = encoded_data['IPV4_DST_ADDR']
        return sequence_data
    
    # Crea sequenze usando dati PROCESSATI
    print("Training sequences:")
    train_sequence_data = create_data_for_sequences(X_train_processed, train_data_encoded)
    train_sequences, train_seq_targets, train_seq_info = create_temporal_sequences(
        train_sequence_data,
        feature_columns_without_ip,
        'multiclass_target',
        sequence_length,
        sequence_stride
    )
    
    print("\nValidation sequences:")
    val_sequence_data = create_data_for_sequences(X_val_processed, val_data_encoded)
    val_sequences, val_seq_targets, val_seq_info = create_temporal_sequences(
        val_sequence_data,
        feature_columns_without_ip,
        'multiclass_target',
        sequence_length,
        sequence_stride
    )
    
    print("\nTest sequences:")
    test_sequence_data = create_data_for_sequences(X_test_processed, test_data_encoded)
    test_sequences, test_seq_targets, test_seq_info = create_temporal_sequences(
        test_sequence_data,
        feature_columns_without_ip,
        'multiclass_target',
        sequence_length,
        sequence_stride
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
        'feature_columns': feature_columns_without_ip,
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
"""


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
    
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} non trovato: {path}")
    
    print("Caricamento dataset ordinati temporalmente...")
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
    
    # Rimuovi colonne IP dalle feature_columns per il modello
    ip_columns = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    feature_columns_without_ip = [col for col in feature_columns if col not in ip_columns]
    
    print(f"\nFeatures utilizzate: {len(feature_columns_without_ip)} di {len(expected_features)} configurate (IP escluse)")
    print(f"- Numeriche: {len([col for col in numeric_columns if col in feature_columns_without_ip])}")
    print(f"- Categoriche: {len([col for col in categorical_columns if col in feature_columns_without_ip])}")

    # Crea gruppi di features (SENZA colonne IP)
    feature_groups = create_feature_groups(feature_columns_without_ip, numeric_columns, categorical_columns)
    
    # === FASE 2: CREAZIONE ENCODING MULTICLASS ===
    print(f"\n=== FASE 2: CREAZIONE ENCODING MULTICLASS ===")
    
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    print(f"\nApplicazione encoding...")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)

    # === FASE 3: CREAZIONE SEQUENZE TEMPORALI (PRIMA DEL PROCESSING) ===
    print(f"\n=== FASE 3: CREAZIONE SEQUENZE TEMPORALI ===")
    
    # Crea sequenze usando i dati encoded ma NON processati (con colonne IP)
    print("Training sequences:")
    train_sequences, train_seq_targets, train_seq_info = create_temporal_sequences(
        train_data_encoded,
        feature_columns_without_ip,  # Solo features senza IP per le sequenze
        'multiclass_target',
        sequence_length,
        sequence_stride
    )
    
    print("\nValidation sequences:")
    val_sequences, val_seq_targets, val_seq_info = create_temporal_sequences(
        val_data_encoded,
        feature_columns_without_ip,
        'multiclass_target',
        sequence_length,
        sequence_stride
    )
    
    print("\nTest sequences:")
    test_sequences, test_seq_targets, test_seq_info = create_temporal_sequences(
        test_data_encoded,
        feature_columns_without_ip,
        'multiclass_target',
        sequence_length,
        sequence_stride
    )

    # === FASE 4: PREPROCESSING FEATURES (DOPO CREAZIONE SEQUENZE) ===
    print(f"\n=== FASE 4: PREPROCESSING FEATURES ===")
    
    # Separa features dai dati encoded (SENZA colonne IP)
    X_train = train_data_encoded[feature_columns_without_ip].copy()
    X_val = val_data_encoded[feature_columns_without_ip].copy()
    X_test = test_data_encoded[feature_columns_without_ip].copy()
    
    # Rimuovi colonne IP anche dalle configurazioni per il processing
    categorical_columns_without_ip = [col for col in categorical_columns if col not in ip_columns]
    numeric_columns_without_ip = [col for col in numeric_columns if col not in ip_columns]
    
    # 1. Creazione mappings per embedding (variabili categoriche SENZA IP)
    print("\nCreazione mappings per embedding categorici...")
    embedding_mappings, vocab_stats = create_embedding_mappings(
        X_train, categorical_columns_without_ip, min_freq=min_freq_categorical, max_vocab_size=max_vocab_size
    )
    
    # Applica mappings
    X_train_embedded = apply_embedding_mappings(X_train, embedding_mappings)
    X_val_embedded = apply_embedding_mappings(X_val, embedding_mappings)
    X_test_embedded = apply_embedding_mappings(X_test, embedding_mappings)
    
    print(f"- Vocabolari creati: {len(embedding_mappings)}")
    
    # 2. Normalizzazione features numeriche (SENZA IP)
    print("\nNormalizzazione Min-Max per features numeriche...")
    X_train_processed, X_val_processed, X_test_processed, normalization_params, numeric_cols_present = normalize_numeric_features(
        X_train_embedded, X_val_embedded, X_test_embedded, numeric_columns_without_ip
    )
    
    print(f"- Features numeriche normalizzate: {len(numeric_cols_present)}")
    
    # === FASE 5: SALVATAGGIO SEQUENZE TEMPORALI ===
    print(f"\n=== FASE 5: SALVATAGGIO SEQUENZE TEMPORALI ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    transformer_metadata = {
        'architecture': 'Temporal_Transformer',
        'temporal_config': {
            'sequence_length': sequence_length,
            'sequence_stride': sequence_stride,
            'feature_dim': len(feature_columns_without_ip)  # Senza IP
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
        'feature_columns': feature_columns_without_ip,  # Senza IP
        'feature_groups': feature_groups,
        'preprocessing_applied': {
            'temporal_sequences': True,
            'embedding_mappings': True,
            'min_max_normalization': True,
            'order_preservation': True,
            'ip_columns_excluded': True  # Documentato
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
    
    metadata_path = os.path.join(output_dir, "transformer_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(transformer_metadata, f, indent=2)
    
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
    
    encoder_path = os.path.join(output_dir, "transformer_label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"- Metadati: {metadata_path}")
    print(f"- Mappings: {mappings_path}")
    print(f"- Label encoder: {encoder_path}")
    
    # === ANALISI FINALE ===
    print(f"\n=== ANALISI DISTRIBUZIONE TEMPORALE ===")
    
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
    print(f"  - Features per timestep: {len(feature_columns_without_ip)} (IP escluse)")
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
"""
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