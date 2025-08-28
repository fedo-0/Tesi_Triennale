import pandas as pd
import numpy as np
import json
import os
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

def undersample_benign_class(df, attack_col='Attack', undersample_ratio=0.5):
    """
    Undersampling parziale della classe Benign mantenendo l'ordine temporale
    """
    print(f"\n--- UNDERSAMPLING PARZIALE CLASSE BENIGN ({undersample_ratio*100:.0f}% riduzione) ---")
    
    initial_samples = len(df)
    
    # Identifica classe benigna
    benign_mask = df[attack_col].str.lower().isin(['benign', 'normal'])
    attack_mask = ~benign_mask
    
    benign_count = benign_mask.sum()
    attack_count = attack_mask.sum()
    
    print(f"Distribuzione originale:")
    print(f"  Benign: {benign_count:,} ({benign_count/initial_samples*100:.2f}%)")
    print(f"  Attack: {attack_count:,} ({attack_count/initial_samples*100:.2f}%)")
    
    if benign_count == 0:
        print("⚠️  Nessun campione benigno trovato, nessun undersampling applicato")
        return df.copy(), 0
    
    # Calcola quanti campioni benign mantenere
    benign_to_keep = int(benign_count * (1 - undersample_ratio))
    benign_to_remove = benign_count - benign_to_keep
    
    print(f"\nUndersampling Benign:")
    print(f"  Da mantenere: {benign_to_keep:,}")
    print(f"  Da rimuovere: {benign_to_remove:,}")
    
    # Trova indici dei campioni benigni
    benign_indices = df[benign_mask].index.tolist()
    
    # Undersampling UNIFORME mantenendo ordine temporale
    # Prendi ogni N-esimo campione benigno per distribuzione uniforme nel tempo
    step = benign_count / benign_to_keep
    indices_to_keep = []
    
    for i in range(benign_to_keep):
        idx_position = int(i * step)
        if idx_position < len(benign_indices):
            indices_to_keep.append(benign_indices[idx_position])
    
    # Crea il dataset sottocampionato
    # Mantieni tutti gli attacchi + sottocampione benigni selezionati
    attack_indices = df[attack_mask].index.tolist()
    final_indices = sorted(attack_indices + indices_to_keep)  # Mantieni ordine temporale
    
    df_undersampled = df.loc[final_indices].copy()
    df_undersampled.reset_index(drop=True, inplace=True)
    
    # Statistiche finali
    final_samples = len(df_undersampled)
    final_benign = (df_undersampled[attack_col].str.lower().isin(['benign', 'normal'])).sum()
    final_attack = final_samples - final_benign
    
    print(f"\n📊 Risultato undersampling:")
    print(f"  Campioni totali: {initial_samples:,} → {final_samples:,} (-{initial_samples - final_samples:,})")
    print(f"  Benign: {benign_count:,} → {final_benign:,} (-{benign_count - final_benign:,})")
    print(f"  Attack: {attack_count:,} → {final_attack:,} (invariato)")
    print(f"  Riduzione totale: {((initial_samples - final_samples)/initial_samples)*100:.2f}%")
    print(f"  Nuova distribuzione Benign: {final_benign/final_samples*100:.2f}%")
    print(f"  Nuova distribuzione Attack: {final_attack/final_samples*100:.2f}%")
    
    return df_undersampled, benign_to_remove

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

def create_transformer_sequences(df, label_col='Label', attack_col='Attack', 
                                min_window_size=10, max_window_size=30, 
                                feature_columns=None, target_col='multiclass_target'):
    """
    Crea sequenze per Transformer invece di dataset flat
    Ogni micro-finestra diventa una sequenza di pacchetti
    """
    print(f"\n--- CREAZIONE SEQUENZE TRANSFORMER ({min_window_size}-{max_window_size} pacchetti) ---")
    
    # Identifica attacchi con logica binaria (come nel preprocessing normale)
    if df[label_col].dtype == 'object':
        attack_mask = df[label_col] != 'Benign'
    else:
        attack_mask = df[label_col] != 0
    
    sequences = []
    labels = []
    current_pos = 0
    window_id = 0
    
    total_len = len(df)
    progress_interval = max(1, total_len // 100)
    
    while current_pos < total_len:
        remaining_packets = total_len - current_pos
        max_possible_size = min(max_window_size, remaining_packets)
        
        # Determina dimensione finestra (logica identica al preprocessing normale)
        window_size = min_window_size
        lookahead_end = min(current_pos + max_possible_size, total_len)
        lookahead_attack_ratio = attack_mask.iloc[current_pos:lookahead_end].mean()
        
        if lookahead_attack_ratio == 0:
            window_size = max_possible_size
        elif lookahead_attack_ratio < 0.1:
            window_size = min(max_window_size, max_possible_size)
        else:
            window_size = min(min_window_size + 5, max_possible_size)
        
        end_pos = min(current_pos + window_size, total_len)
        
        # Estrai sequenza di pacchetti (features)
        window_data = df.iloc[current_pos:end_pos]
        sequence_features = window_data[feature_columns].values  # Shape: (seq_len, n_features)
        
        # Label della finestra: classe più frequente nella finestra
        window_targets = window_data[target_col].values
        window_label = np.bincount(window_targets).argmax()  # Classe più frequente
        
        sequences.append(sequence_features)
        labels.append(window_label)
        
        current_pos = end_pos
        window_id += 1
        
        if window_id % progress_interval == 0 or current_pos >= total_len:
            progress = (current_pos / total_len) * 100
            print(f"  Progresso: {progress:.1f}% ({window_id:,} sequenze)")
    
    print(f"Totale sequenze create: {len(sequences)}")
    
    # Statistiche sequenze
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Lunghezza sequenze: min={min(seq_lengths)}, max={max(seq_lengths)}, media={np.mean(seq_lengths):.1f}")
    
    return sequences, labels, seq_lengths

def pad_sequences(sequences, max_length=None, pad_value=0.0):
    """
    Padding delle sequenze per uniformare la lunghezza
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    print(f"Padding sequenze a lunghezza massima: {max_length}")
    
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        seq_len = len(seq)
        if seq_len > max_length:
            # Tronca se troppo lunga
            padded_seq = seq[:max_length]
            mask = np.ones(max_length, dtype=np.float32)
        else:
            # Padding se troppo corta
            padding_needed = max_length - seq_len
            padding = np.full((padding_needed, seq.shape[1]), pad_value, dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
            
            # Attention mask: 1 per token reali, 0 per padding
            mask = np.concatenate([np.ones(seq_len), np.zeros(padding_needed)]).astype(np.float32)
        
        padded_sequences.append(padded_seq)
        attention_masks.append(mask)
    
    return np.array(padded_sequences), np.array(attention_masks)

def transformer_window_split(sequences, labels, attention_masks, 
                            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split delle sequenze mantenendo ordine temporale
    """
    print(f"\n--- SPLIT SEQUENZE TRANSFORMER ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}) ---")
    
    n_sequences = len(sequences)
    
    # Split temporale (mantieni ordine)
    train_end = int(n_sequences * train_ratio)
    val_end = int(n_sequences * (train_ratio + val_ratio))
    
    train_sequences = sequences[:train_end]
    train_labels = labels[:train_end]
    train_masks = attention_masks[:train_end]
    
    val_sequences = sequences[train_end:val_end]
    val_labels = labels[train_end:val_end]
    val_masks = attention_masks[train_end:val_end]
    
    test_sequences = sequences[val_end:]
    test_labels = labels[val_end:]
    test_masks = attention_masks[val_end:]
    
    print(f"Split completato:")
    print(f"  Train: {len(train_sequences):,} sequenze")
    print(f"  Val: {len(val_sequences):,} sequenze")
    print(f"  Test: {len(test_sequences):,} sequenze")
    
    return (train_sequences, train_labels, train_masks), \
           (val_sequences, val_labels, val_masks), \
           (test_sequences, test_labels, test_masks)

def preprocess_dataset_transformer(dataset_path, config_path, output_dir,
                                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                                 min_window_size=10, max_window_size=30,
                                 label_col='Label', attack_col='Attack',
                                 min_samples_per_class=10000,
                                 benign_undersample_ratio=0.5):
    """
    Preprocessing specifico per Transformer - produce sequenze invece di campioni singoli
    """
    
    print("PREPROCESSING TRANSFORMER CON MICRO-FINESTRE SEQUENZIALI")
    print("=" * 70)
    
    # Fase 1-4: Identiche al preprocessing normale
    # (caricamento, pulizia, undersampling, filtraggio classi)
    # ... [usa le stesse funzioni del preprocessing normale] ...
    
    # SIMULAZIONE delle fasi 1-4 (sostituisci con il tuo preprocessing reale)
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    df = pd.read_csv(dataset_path)
    # ... pulizia, undersampling, filtraggio come nel preprocessing normale ...
    
    expected_features = numeric_columns + categorical_columns
    feature_columns = [col for col in expected_features if col in df.columns]
    
    # Fase 5: NUOVO - Crea encoding e preprocessing normale dei pacchetti
    # (necessario per avere features numeriche uniformi)
    print(f"\n--- PREPROCESSING FEATURES PER TRANSFORMER ---")
    
    # Frequency encoding
    freq_mappings = frequency_encoding_fit(df[feature_columns], categorical_columns)
    df_encoded = frequency_encoding_transform(df[feature_columns], freq_mappings)
    
    # Standard scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)
    
    # Ricombina con label columns
    df_final = df_scaled.copy()
    df_final[label_col] = df[label_col].values
    df_final[attack_col] = df[attack_col].values
    
    # Crea encoding multiclass
    label_encoder, class_mapping = create_multiclass_encoding(df_final, attack_col, label_col)
    df_final = apply_multiclass_encoding(df_final, label_encoder, attack_col)
    
    # Fase 6: NUOVO - Crea sequenze per Transformer
    sequences, labels, seq_lengths = create_transformer_sequences(
        df_final, label_col, attack_col, min_window_size, max_window_size, 
        feature_columns, 'multiclass_target'
    )
    
    # Fase 7: Padding sequenze
    max_seq_length = min(max(seq_lengths), 50)  # Limite ragionevole
    padded_sequences, attention_masks = pad_sequences(sequences, max_seq_length)
    
    # Fase 8: Split temporale
    train_data, val_data, test_data = transformer_window_split(
        padded_sequences, labels, attention_masks, train_ratio, val_ratio, test_ratio
    )
    
    # Fase 9: Salvataggio
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva dati transformer
    np.save(os.path.join(output_dir, "train_sequences.npy"), train_data[0])
    np.save(os.path.join(output_dir, "train_labels.npy"), train_data[1])
    np.save(os.path.join(output_dir, "train_masks.npy"), train_data[2])
    
    np.save(os.path.join(output_dir, "val_sequences.npy"), val_data[0])
    np.save(os.path.join(output_dir, "val_labels.npy"), val_data[1])
    np.save(os.path.join(output_dir, "val_masks.npy"), val_data[2])
    
    np.save(os.path.join(output_dir, "test_sequences.npy"), test_data[0])
    np.save(os.path.join(output_dir, "test_labels.npy"), test_data[1])
    np.save(os.path.join(output_dir, "test_masks.npy"), test_data[2])
    
    # Salva metadati
    metadata = {
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'class_mapping': class_mapping,
        'n_classes': len(label_encoder.classes_),
        'feature_columns': feature_columns,
        'n_features': len(feature_columns),
        'max_seq_length': max_seq_length,
        'min_window_size': min_window_size,
        'max_window_size': max_window_size,
        'model_type': 'transformer_multiclass'
    }
    
    with open(os.path.join(output_dir, "transformer_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n--- SALVATAGGIO COMPLETATO ---")
    print(f"Sequenze salvate: train={train_data[0].shape}, val={val_data[0].shape}, test={test_data[0].shape}")
    print(f"Shape sequenze: (n_sequences, max_seq_length, n_features) = {train_data[0].shape}")
    
    return train_data, val_data, test_data, scaler, label_encoder, metadata

# Le altre funzioni del preprocessing normale (frequency_encoding_fit, etc.)
# vanno copiate dal tuo preprocessing esistente...