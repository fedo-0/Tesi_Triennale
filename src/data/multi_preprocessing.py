import torch
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import defaultdict, Counter
from imblearn.over_sampling import SMOTE

def load_dataset_config(config_path="config/dataset.json"):
    """Carica la configurazione del dataset"""
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
    """Applica trasformazione log1p alle colonne numeriche"""
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

def analyze_multiclass_distribution(df, label_col='Label', attack_col='Attack'):
    """
    Analizza la distribuzione delle classi multiclass
    """
    print(f"\n--- ANALISI DISTRIBUZIONE MULTICLASS ---")
    
    # Conta occorrenze per tipo di attacco
    attack_counts = df[attack_col].value_counts()
    total_samples = len(df)
    
    print(f"Totale campioni: {total_samples:,}")
    print(f"Classi uniche trovate: {len(attack_counts)}")
    
    print(f"\nDistribuzione classi:")
    for class_name, count in attack_counts.items():
        percentage = (count / total_samples) * 100
        class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
        print(f"  {class_name}: {count:,} ({percentage:.2f}%) - {class_type}")
    
    # Raggruppa in binario per micro-finestre (come nel classificatore binario)
    binary_attacks = df[label_col].value_counts()
    print(f"\nDistribuzione binaria (per logica micro-finestre):")
    for label, count in binary_attacks.items():
        label_name = "Benigno" if label == 0 else "Attacco"
        percentage = (count / total_samples) * 100
        print(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
    
    return attack_counts

def create_multiclass_encoding(df_train, attack_col='Attack', label_col='Label'):
    """
    Crea l'encoding per le classi multiclass basandosi solo sul training set
    """
    print(f"\n--- CREAZIONE ENCODING MULTICLASS ---")
    
    # Estrai tutte le classi uniche dal training set
    unique_classes = df_train[attack_col].unique()
    n_classes = len(unique_classes)
    
    print(f"Classi trovate nel training set: {n_classes}")
    
    # Ordina le classi per frequenza (più frequenti per prime)
    class_counts = df_train[attack_col].value_counts()
    sorted_classes = class_counts.index.tolist()
    
    # Crea LabelEncoder con ordine determinato
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
    """
    Applica l'encoding multiclass a un dataset
    """
    df_encoded = df.copy()
    
    # Gestisci classi non viste nel training (se presenti in val/test)
    known_classes = set(label_encoder.classes_)
    df_classes = set(df[attack_col].unique())
    unknown_classes = df_classes - known_classes
    
    if unknown_classes:
        print(f"⚠️  Classi non viste nel training: {unknown_classes}")
        print("   Verranno mappate alla classe più frequente del training")
        # Mappa classi sconosciute alla classe più frequente (prima nell'encoder)
        most_frequent_class = label_encoder.classes_[0]
        df_encoded[attack_col] = df_encoded[attack_col].apply(
            lambda x: most_frequent_class if x in unknown_classes else x
        )
    
    # Applica encoding
    df_encoded['multiclass_target'] = label_encoder.transform(df_encoded[attack_col])
    
    return df_encoded

def analyze_packet_distribution_multiclass(df, label_col='Label', attack_col='Attack'):
    """
    Analizza la distribuzione dei pacchetti per il multiclass
    (Mantiene logica binaria per micro-finestre + analisi multiclass)
    """
    print(f"\n--- ANALISI DISTRIBUZIONE PACCHETTI (MULTICLASS) ---")
    
    # Analisi multiclass dettagliata
    multiclass_dist = analyze_multiclass_distribution(df, label_col, attack_col)
    
    # Analisi binaria per micro-finestre (LOGICA IDENTICA AL BINARIO)
    if df[label_col].dtype == 'object':
        attack_mask = df[label_col] != 'Benign'
    else:
        attack_mask = df[label_col] != 0
    
    total_packets = len(df)
    attack_packets = attack_mask.sum()
    benign_packets = total_packets - attack_packets
    
    print(f"\nStatistiche binarie (per logica micro-finestre):")
    print(f"Pacchetti totali: {total_packets:,}")
    print(f"Pacchetti di attacco: {attack_packets:,} ({attack_packets/total_packets*100:.2f}%)")
    print(f"Pacchetti benigni: {benign_packets:,} ({benign_packets/total_packets*100:.2f}%)")
    
    # Trova dove finiscono gli attacchi (LOGICA IDENTICA AL BINARIO)
    if attack_packets > 0:
        attack_positions = df[attack_mask].index.tolist()
        first_attack = attack_positions[0]
        last_attack = attack_positions[-1]
        attack_span = (last_attack + 1) / total_packets
        
        print(f"Primo attacco: posizione {first_attack:,}")
        print(f"Ultimo attacco: posizione {last_attack:,}")
        print(f"Span degli attacchi: {attack_span*100:.1f}% del dataset")
        
        return {
            'total_packets': total_packets,
            'attack_packets': attack_packets,
            'last_attack_position': last_attack,
            'attack_span': attack_span,
            'multiclass_distribution': multiclass_dist
        }
    else:
        print("Nessun attacco trovato nel dataset!")
        return {
            'total_packets': total_packets,
            'attack_packets': 0,
            'last_attack_position': 0,
            'attack_span': 0,
            'multiclass_distribution': multiclass_dist
        }

def create_micro_windows_multiclass(df, label_col='Label', attack_col='Attack', 
                                   min_window_size=10, max_window_size=30):
    """
    Crea micro-finestre per multiclass
    (Usa LOGICA BINARIA per categorizzazione, aggiunge info multiclass)
    """
    print(f"\n--- CREAZIONE MICRO-FINESTRE MULTICLASS ({min_window_size}-{max_window_size} pacchetti) ---")
    print("Strategia: Categorizzazione binaria per bilanciamento + Info multiclass per analisi")
    
    # Identifica attacchi con LOGICA BINARIA (IDENTICA AL BINARIO)
    if df[label_col].dtype == 'object':
        attack_mask = df[label_col] != 'Benign'
    else:
        attack_mask = df[label_col] != 0
    
    micro_windows = []
    current_pos = 0
    window_id = 0
    
    while current_pos < len(df):
        # Logica di dimensionamento (IDENTICA AL BINARIO)
        remaining_packets = len(df) - current_pos
        max_possible_size = min(max_window_size, remaining_packets)
        
        window_size = min_window_size
        
        # Determina dimensione basandosi su presenza binaria di attacchi
        lookahead_window = df.iloc[current_pos:current_pos + max_possible_size]
        lookahead_attack_ratio = attack_mask.iloc[current_pos:current_pos + max_possible_size].mean()
        
        if lookahead_attack_ratio == 0:  # Solo benigni
            window_size = max_possible_size
        elif lookahead_attack_ratio < 0.1:  # Pochi attacchi
            window_size = min(max_window_size, max_possible_size)
        else:  # Zona con attacchi, finestre più piccole
            window_size = min(min_window_size + 5, max_possible_size)
        
        # Estrai micro-finestra
        end_pos = min(current_pos + window_size, len(df))
        micro_window_data = df.iloc[current_pos:end_pos].copy()
        
        # Calcola statistiche binarie (IDENTICHE AL BINARIO)
        window_attack_mask = attack_mask.iloc[current_pos:end_pos]
        attack_count = window_attack_mask.sum()
        attack_ratio = attack_count / len(micro_window_data)
        
        # NUOVO: Aggiungi statistiche multiclass dettagliate
        window_multiclass_dist = micro_window_data[attack_col].value_counts()
        unique_attack_types = window_multiclass_dist[window_multiclass_dist.index != 'Benign'].to_dict() if 'Benign' in window_multiclass_dist.index else window_multiclass_dist.to_dict()
        
        window_info = {
            'id': window_id,
            'data': micro_window_data,
            'start_pos': current_pos,
            'end_pos': end_pos,
            'size': len(micro_window_data),
            # Statistiche binarie (per categorizzazione)
            'attack_count': attack_count,
            'attack_ratio': attack_ratio,
            'benign_count': len(micro_window_data) - attack_count,
            # Statistiche multiclass (per analisi)
            'multiclass_distribution': window_multiclass_dist,
            'attack_types': unique_attack_types,
            'n_attack_types': len(unique_attack_types)
        }
        
        micro_windows.append(window_info)
        current_pos = end_pos
        window_id += 1
        
        if window_id % 1000 == 0:
            print(f"  Processate {window_id} micro-finestre...")
    
    print(f"Totale micro-finestre create: {len(micro_windows)}")
    
    # Analizza le micro-finestre (BINARIO + MULTICLASS)
    sizes = [w['size'] for w in micro_windows]
    attack_ratios = [w['attack_ratio'] for w in micro_windows]
    attack_types_per_window = [w['n_attack_types'] for w in micro_windows]
    
    print(f"\nStatistiche Micro-Finestre:")
    print(f"  Dimensione media: {np.mean(sizes):.1f} pacchetti")
    print(f"  Range dimensioni: {min(sizes)}-{max(sizes)} pacchetti")
    print(f"  Ratio attacchi medio: {np.mean(attack_ratios):.3f}")
    print(f"  Finestre con attacchi: {sum(1 for r in attack_ratios if r > 0)} ({sum(1 for r in attack_ratios if r > 0)/len(attack_ratios)*100:.1f}%)")
    print(f"  Finestre solo benigne: {sum(1 for r in attack_ratios if r == 0)} ({sum(1 for r in attack_ratios if r == 0)/len(attack_ratios)*100:.1f}%)")
    print(f"  Tipi di attacco medio per finestra: {np.mean(attack_types_per_window):.1f}")
    
    return micro_windows

def stratified_window_assignment_multiclass(micro_windows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Assegna le micro-finestre ai set usando LOGICA BINARIA
    (IDENTICA al classificatore binario)
    """
    print(f"\n--- ASSEGNAZIONE STRATIFICATA MICRO-FINESTRE ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}) ---")
    print("Assegnazione basata su categorizzazione binaria (attacco/benigno)")
    
    # Categorizza finestre per contenuto BINARIO (IDENTICA AL BINARIO)
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
    
    print("Distribuzione categorie micro-finestre (categorizzazione binaria):")
    for name, windows in categories.items():
        print(f"  {name}: {len(windows)} finestre ({len(windows)/len(micro_windows)*100:.1f}%)")
        
        # NUOVO: Analisi multiclass per categoria
        if len(windows) > 0:
            all_types = {}
            for w in windows:
                for attack_type, count in w['attack_types'].items():
                    all_types[attack_type] = all_types.get(attack_type, 0) + count
            if all_types:
                print(f"    Tipi attacchi: {list(all_types.keys())}")
    
    # Distribuzione round-robin (IDENTICA AL BINARIO)
    train_windows, val_windows, test_windows = [], [], []
    
    for category_name, windows in categories.items():
        if len(windows) == 0:
            continue
        
        for i, window in enumerate(windows):
            cycle_pos = i % 20  # Pattern 70-15-15: ogni 20 finestre → 14 train, 3 val, 3 test
            
            if cycle_pos < 14:  # 70% train
                train_windows.append(window)
            elif cycle_pos < 17:  # 15% val
                val_windows.append(window)
            else:  # 15% test
                test_windows.append(window)
        
        # Statistiche per categoria
        train_cat = [w for w in windows if w in train_windows]
        val_cat = [w for w in windows if w in val_windows]
        test_cat = [w for w in windows if w in test_windows]
        
        print(f"    Assegnazione: Train={len(train_cat)}, Val={len(val_cat)}, Test={len(test_cat)}")
    
    # Ordina per posizione temporale (IDENTICA AL BINARIO)
    train_windows.sort(key=lambda x: x['start_pos'])
    val_windows.sort(key=lambda x: x['start_pos'])
    test_windows.sort(key=lambda x: x['start_pos'])
    
    print(f"\nTotale finestre assegnate:")
    print(f"  Train: {len(train_windows)} finestre")
    print(f"  Validation: {len(val_windows)} finestre")
    print(f"  Test: {len(test_windows)} finestre")
    
    return train_windows, val_windows, test_windows

def reconstruct_datasets_multiclass(train_windows, val_windows, test_windows):
    """
    Ricostruisce i dataset finali (IDENTICA AL BINARIO)
    """
    print(f"\n--- RICOSTRUZIONE DATASET FINALI ---")
    
    train_data = pd.concat([w['data'] for w in train_windows], ignore_index=True)
    val_data = pd.concat([w['data'] for w in val_windows], ignore_index=True)
    test_data = pd.concat([w['data'] for w in test_windows], ignore_index=True)
    
    print(f"Dataset ricostruiti:")
    print(f"  Train: {len(train_data)} pacchetti")
    print(f"  Validation: {len(val_data)} pacchetti")
    print(f"  Test: {len(test_data)} pacchetti")
    
    return train_data, val_data, test_data

def micro_window_split_multiclass(df, label_col='Label', attack_col='Attack',
                                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                                 min_window_size=10, max_window_size=30):
    """
    Split con micro-finestre per multiclass
    (Combina logica binaria per bilanciamento + output multiclass)
    """
    print("MICRO-WINDOW STRATIFIED TEMPORAL SPLIT - MULTICLASS")
    print("=" * 70)
    print("Strategia: Categorizzazione binaria per micro-finestre + Output multiclass finale")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Analisi distribuzione (INCLUDE MULTICLASS)
    packet_stats = analyze_packet_distribution_multiclass(df, label_col, attack_col)
    
    # Crea micro-finestre con logica binaria + info multiclass
    micro_windows = create_micro_windows_multiclass(df, label_col, attack_col, min_window_size, max_window_size)
    
    if len(micro_windows) < 3:
        raise ValueError("Troppo poche micro-finestre per creare train/val/test set")
    
    # Assegnazione stratificata binaria
    train_windows, val_windows, test_windows = stratified_window_assignment_multiclass(
        micro_windows, train_ratio, val_ratio, test_ratio
    )
    
    # Ricostruzione dataset
    train_data, val_data, test_data = reconstruct_datasets_multiclass(train_windows, val_windows, test_windows)
    
    return train_data, val_data, test_data

# ===== FUNZIONE PRINCIPALE =====

def preprocess_dataset_multiclass(dataset_path, config_path, output_dir, 
                                 train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                                 min_window_size=10, max_window_size=30,
                                 label_col='Label', attack_col='Attack'):
    """
    Funzione principale per preprocessing multiclasse con micro-finestre
    """
    
    print("PREPROCESSING MULTICLASSE CON MICRO-FINESTRE TEMPORALI")
    print("CON SMOTE!!!")
    print("=" * 70)
    print("Versione: Autonoma e completa per classificazione multiclasse")
    print(f"Strategia: Bilanciamento binario + Output multiclasse")
    
    # Carica configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Colonna label binaria: {label_col}")
    print(f"- Colonna attack multiclass: {attack_col}")
    print(f"- Rapporti split: {train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}")
    
    # Carica dataset
    print(f"\nCaricamento dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset originale: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    # Gestione valori infiniti e NaN (IDENTICA AL BINARIO)
    initial_rows = len(df)
    numeric_cols_present = [col for col in numeric_columns if col in df.columns]

    if numeric_cols_present:
        inf_count = 0
        for col in numeric_cols_present:
            col_inf_count = np.isinf(df[col]).sum()
            inf_count += col_inf_count
            if col_inf_count > 0:
                print(f"⚠️  Trovati {col_inf_count} valori infiniti in colonna: {col}")
        
        df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], np.nan)
        print(f"Sostituiti {inf_count} valori infiniti con NaN")

    df = df.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    print(f"Rimosse {removed_rows:,} righe contenenti valori NaN o infiniti")
    print(f"Dataset dopo pulizia: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    if len(df) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    
    # Verifica colonne target
    if label_col not in df.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset!")
    if attack_col not in df.columns:
        raise ValueError(f"Colonna attack '{attack_col}' non trovata nel dataset!")
    
    """
    # Split con micro-finestre (logica binaria per bilanciamento)
    feature_columns = [col for col in df.columns if col not in [label_col, attack_col]]
    
    train_data, val_data, test_data = micro_window_split_multiclass(
        df, label_col, attack_col, train_ratio, val_ratio, test_ratio, 
        min_window_size, max_window_size
    )
    """

    # CORREZIONE: Usa SOLO le features dal config, non tutte le colonne CSV
    expected_features = numeric_columns + categorical_columns
    print(f"\n--- CORREZIONE FEATURE COLUMNS ---")
    print(f"Features da CSV (tutte): {len([col for col in df.columns if col not in [label_col, attack_col]])}")
    print(f"Features da config: {len(expected_features)}")

    # Verifica che le features dal config esistano nel dataset
    missing_features = set(expected_features) - set(df.columns)
    extra_features_in_csv = set(df.columns) - set(expected_features) - {label_col, attack_col}

    if missing_features:
        print(f"⚠️  Features mancanti nel CSV: {missing_features}")
    if extra_features_in_csv:
        print(f"📊 Features extra nel CSV (ignorate): {list(extra_features_in_csv)[:5]}... (total: {len(extra_features_in_csv)})")

    # USA SOLO LE FEATURES DAL CONFIG
    feature_columns = [col for col in expected_features if col in df.columns]

    print(f"Features finali usate: {len(feature_columns)} (dovrebbero essere 48)")
    print(f"✅ Match con config: {len(feature_columns) == len(expected_features)}")

    train_data, val_data, test_data = micro_window_split_multiclass(
        df, label_col, attack_col, train_ratio, val_ratio, test_ratio, 
        min_window_size, max_window_size
    )
    
    # Crea encoding multiclass basato solo su training set
    label_encoder, class_mapping = create_multiclass_encoding(train_data, attack_col, label_col)
    
    # Applica encoding multiclass a tutti i set
    print(f"\n--- APPLICAZIONE ENCODING MULTICLASS ---")
    train_data_encoded = apply_multiclass_encoding(train_data, label_encoder, attack_col)
    val_data_encoded = apply_multiclass_encoding(val_data, label_encoder, attack_col)
    test_data_encoded = apply_multiclass_encoding(test_data, label_encoder, attack_col)
    print("Encoding multiclass applicato a tutti i set")
    
    # Separa features e target
    X_train = train_data_encoded[feature_columns].copy()
    X_val = val_data_encoded[feature_columns].copy()
    X_test = test_data_encoded[feature_columns].copy()

    # Target multiclass (invece di binario)
    y_train = train_data_encoded['multiclass_target'].copy()
    y_val = val_data_encoded['multiclass_target'].copy()
    y_test = test_data_encoded['multiclass_target'].copy()

    print(f"\nVerifica totale pacchetti: {len(train_data) + len(val_data) + len(test_data):,} / {len(df):,}")

    # *** NUOVO: APPLICA SMOTE AL TRAINING SET ***
    print(f"\n--- APPLICAZIONE SMOTE ---")

    # Mostra distribuzione prima di SMOTE
    original_dist = Counter(y_train)
    print(f"Distribuzione PRIMA di SMOTE:")
    for class_id, count in sorted(original_dist.items()):
        percentage = count / len(y_train) * 100
        print(f"  Classe {class_id}: {count:,} ({percentage:.2f}%)")

    # Calcola strategia SMOTE (porta classi piccole al 20% della maggiore)
    max_samples = max(original_dist.values())
    target_min = int(max_samples * 0.2)  # 20% della classe maggiore

    sampling_strategy = {}
    for class_id, count in original_dist.items():
        if count < target_min:
            sampling_strategy[class_id] = target_min

    print(f"\nStrategia SMOTE (target: {target_min:,} sample minimi):")
    for class_id, target in sampling_strategy.items():
        original = original_dist[class_id]
        multiplier = target / original
        print(f"  Classe {class_id}: {original:,} → {target:,} (x{multiplier:.1f})")

    # Applica SMOTE
    try:
        min_class_size = min(original_dist.values())
        k_neighbors = min(5, max(1, min_class_size - 1))
        
        print(f"\nApplicando SMOTE (k_neighbors={k_neighbors})...")
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=k_neighbors
        )
        
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Mostra risultato
        smote_dist = Counter(y_train_smote)
        print(f"\nDistribuzione DOPO SMOTE:")
        for class_id, count in sorted(smote_dist.items()):
            percentage = count / len(y_train_smote) * 100
            original = original_dist[class_id]
            multiplier = count / original
            print(f"  Classe {class_id}: {count:,} ({percentage:.2f}%) - x{multiplier:.1f}")
        
        print(f"\n✅ SMOTE completato!")
        print(f"Dataset training: {len(y_train):,} → {len(y_train_smote):,} (+{len(y_train_smote) - len(y_train):,})")
        
        # Usa dati SMOTE per il resto del preprocessing
        X_train = X_train_smote.copy()
        y_train = y_train_smote.copy()
        
    except Exception as e:
        print(f"❌ Errore SMOTE: {e}")
        print("Continuando senza SMOTE...")
    
    # Preprocessing features (IDENTICO AL BINARIO)
    print(f"\n--- PREPROCESSING FEATURES ---")
    
    # 1. Frequency encoding
    print("Applicazione frequency encoding...")
    freq_mappings = frequency_encoding_fit(X_train, categorical_columns)
    X_train_encoded = frequency_encoding_transform(X_train, freq_mappings)
    X_val_encoded = frequency_encoding_transform(X_val, freq_mappings)
    X_test_encoded = frequency_encoding_transform(X_test, freq_mappings)
    
    # 2. Trasformazione log1p
    print("Applicazione trasformazione log1p...")
    X_train_log = log1p_transform(X_train_encoded, numeric_columns)
    X_val_log = log1p_transform(X_val_encoded, numeric_columns)
    X_test_log = log1p_transform(X_test_encoded, numeric_columns)
    
    # 3. MinMax scaling
    print("Applicazione MinMax scaling...")
    scaler, numeric_cols_present = minmax_scaling_fit(X_train_log, numeric_columns)
    X_train_scaled = minmax_scaling_transform(X_train_log, scaler, numeric_cols_present)
    X_val_scaled = minmax_scaling_transform(X_val_log, scaler, numeric_cols_present)
    X_test_scaled = minmax_scaling_transform(X_test_log, scaler, numeric_cols_present)
    
    # Ricombina features e target multiclass
    df_train = X_train_scaled.copy()
    df_train['multiclass_target'] = y_train.reset_index(drop=True)
    
    df_val = X_val_scaled.copy()
    df_val['multiclass_target'] = y_val.reset_index(drop=True)
    
    df_test = X_test_scaled.copy()
    df_test['multiclass_target'] = y_test.reset_index(drop=True)
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva dataset
    train_path = os.path.join(output_dir, "train_multiclass_smote.csv")
    val_path = os.path.join(output_dir, "val_multiclass_smote.csv")
    test_path = os.path.join(output_dir, "test_multiclass_smote.csv")
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    # Salva metadati multiclass
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
        'preprocessing_version': 'multiclass_v1.0'
    }
    
    metadata_path = os.path.join(output_dir, "multiclass_metadata_smote.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salva anche i mapping e scaler per riuso
    mappings_path = os.path.join(output_dir, "multiclass_mappings_smote.json")
    mappings_data = {
        'freq_mappings': freq_mappings,
        'class_mapping': class_mapping,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns
    }
    with open(mappings_path, 'w') as f:
        json.dump(mappings_data, f, indent=2)
    
    # Salva scaler (formato pickle)
    import pickle
    scaler_path = os.path.join(output_dir, "multiclass_scaler_smote.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n--- SALVATAGGIO COMPLETATO ---")
    print(f"Dataset salvati:")
    print(f"- Training: {train_path} ({df_train.shape[0]:,} righe)")
    print(f"- Validation: {val_path} ({df_val.shape[0]:,} righe)")
    print(f"- Test: {test_path} ({df_test.shape[0]:,} righe)")
    print(f"\nMetadati salvati:")
    print(f"- Configurazione: {metadata_path}")
    print(f"- Mappature: {mappings_path}")
    print(f"- Scaler: {scaler_path}")
    
    # Statistiche finali dettagliate
    print(f"\n--- STATISTICHE FINALI MULTICLASSE ---")
    
    total_original = len(df_train) + len(df_val) + len(df_test)
    
    for set_name, dataset in [("Training", df_train), ("Validation", df_val), ("Test", df_test)]:
        total = len(dataset)
        print(f"\n{set_name} Set: {total:,} pacchetti ({total/total_original*100:.1f}% del totale)")
        
        # Distribuzione classi multiclass
        value_counts = dataset['multiclass_target'].value_counts().sort_index()
        print(f"  Distribuzione classi:")
        for class_idx, count in value_counts.items():
            class_name = label_encoder.classes_[class_idx]
            percentage = (count / total) * 100
            class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
            print(f"    {class_name} ({class_idx}): {count:,} ({percentage:.2f}%) - {class_type}")
        
        # Verifica bilanciamento binario (per confronto)
        benign_classes = [i for i, name in enumerate(label_encoder.classes_) 
                         if name.lower() in ['benign', 'normal']]
        attack_classes = [i for i, name in enumerate(label_encoder.classes_) 
                         if name.lower() not in ['benign', 'normal']]
        
        benign_count = dataset[dataset['multiclass_target'].isin(benign_classes)].shape[0]
        attack_count = dataset[dataset['multiclass_target'].isin(attack_classes)].shape[0]
        
        print(f"  Bilanciamento binario equivalente:")
        print(f"    Benigni: {benign_count:,} ({benign_count/total*100:.2f}%)")
        print(f"    Attacchi: {attack_count:,} ({attack_count/total*100:.2f}%)")
    
    # Confronto con distribuzione originale
    print(f"\n--- CONFRONTO CON DISTRIBUZIONE ORIGINALE ---")
    original_dist = df[attack_col].value_counts(normalize=True).sort_index()
    combined_dist = pd.concat([df_train, df_val, df_test])['multiclass_target'].value_counts(normalize=True).sort_index()
    
    print(f"Preservazione distribuzione:")
    for class_idx in combined_dist.index:
        class_name = label_encoder.classes_[class_idx]
        original_pct = original_dist.get(class_name, 0) * 100
        final_pct = combined_dist[class_idx] * 100
        diff = abs(original_pct - final_pct)
        status = "✅" if diff < 2.0 else "⚠️" if diff < 5.0 else "❌"
        print(f"  {class_name}: {original_pct:.2f}% → {final_pct:.2f}% (Δ{diff:.2f}%) {status}")
    
    # Verifica completezza
    print(f"\nVerifica completezza:")
    print(f"  Righe originali: {len(df):,}")
    print(f"  Righe finali: {total_original:,}")
    print(f"  Differenza: {len(df) - total_original:,} ({'✅ Nessuna perdita' if len(df) == total_original else '⚠️ Perdita dati'})")
    
    return df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping

# ===== FUNZIONI UTILITY =====

def load_multiclass_metadata(output_dir):
    """
    Carica i metadati del preprocessing multiclasse
    """
    metadata_path = os.path.join(output_dir, "multiclass_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    mappings_path = os.path.join(output_dir, "multiclass_mappings.json")
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    import pickle
    scaler_path = os.path.join(output_dir, "multiclass_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Ricrea label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
    
    return metadata, mappings, scaler, label_encoder

def analyze_multiclass_results(output_dir):
    """
    Analizza i risultati del preprocessing multiclasse
    """
    print("ANALISI RISULTATI PREPROCESSING MULTICLASSE")
    print("=" * 50)
    
    # Carica metadati
    metadata, mappings, scaler, label_encoder = load_multiclass_metadata(output_dir)
    
    # Carica dataset
    train_df = pd.read_csv(os.path.join(output_dir, "train_multiclass.csv"))
    val_df = pd.read_csv(os.path.join(output_dir, "val_multiclass.csv"))
    test_df = pd.read_csv(os.path.join(output_dir, "test_multiclass.csv"))
    
    print(f"Configurazione:")
    print(f"  Classi totali: {metadata['n_classes']}")
    print(f"  Features: {len(metadata['feature_columns'])}")
    print(f"  Split ratio: {metadata['train_ratio']}-{metadata['val_ratio']}-{metadata['test_ratio']}")
    
    print(f"\nMapping classi:")
    for class_name, class_idx in metadata['class_mapping'].items():
        train_count = (train_df['multiclass_target'] == class_idx).sum()
        val_count = (val_df['multiclass_target'] == class_idx).sum()
        test_count = (test_df['multiclass_target'] == class_idx).sum()
        total_count = train_count + val_count + test_count
        
        print(f"  {class_name} ({class_idx}): {total_count:,} campioni")
        print(f"    Train: {train_count:,}, Val: {val_count:,}, Test: {test_count:,}")
    
    return metadata, mappings, scaler, label_encoder

def verify_multiclass_consistency(output_dir):
    """
    Verifica la consistenza del preprocessing multiclasse
    """
    print("VERIFICA CONSISTENZA PREPROCESSING MULTICLASSE")
    print("=" * 50)
    
    try:
        # Carica tutto
        metadata, mappings, scaler, label_encoder = load_multiclass_metadata(output_dir)
        train_df = pd.read_csv(os.path.join(output_dir, "train_multiclass.csv"))
        val_df = pd.read_csv(os.path.join(output_dir, "val_multiclass.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test_multiclass.csv"))
        
        checks_passed = 0
        total_checks = 6
        
        # Check 1: Numero classi
        expected_classes = metadata['n_classes']
        actual_classes = len(np.unique(np.concatenate([
            train_df['multiclass_target'].values,
            val_df['multiclass_target'].values,
            test_df['multiclass_target'].values
        ])))
        
        if actual_classes == expected_classes:
            print("✅ Check classi: PASS")
            checks_passed += 1
        else:
            print(f"❌ Check classi: FAIL (Expected: {expected_classes}, Found: {actual_classes})")
        
        # Check 2: Range valori target
        all_targets = np.concatenate([
            train_df['multiclass_target'].values,
            val_df['multiclass_target'].values,
            test_df['multiclass_target'].values
        ])
        
        if all_targets.min() >= 0 and all_targets.max() < expected_classes:
            print("✅ Check range target: PASS")
            checks_passed += 1
        else:
            print(f"❌ Check range target: FAIL (Range: {all_targets.min()}-{all_targets.max()})")
        
        # Check 3: Nessun valore NaN
        has_nan = any([
            train_df.isnull().any().any(),
            val_df.isnull().any().any(),
            test_df.isnull().any().any()
        ])
        
        if not has_nan:
            print("✅ Check valori NaN: PASS")
            checks_passed += 1
        else:
            print("❌ Check valori NaN: FAIL")
        
        # Check 4: Dimensioni features
        expected_features = len(metadata['feature_columns'])
        actual_features = train_df.shape[1] - 1  # -1 per multiclass_target
        
        if actual_features == expected_features:
            print("✅ Check dimensioni features: PASS")
            checks_passed += 1
        else:
            print(f"❌ Check dimensioni features: FAIL (Expected: {expected_features}, Found: {actual_features})")
        
        # Check 5: Bilanciamento tra set
        total_samples = len(train_df) + len(val_df) + len(test_df)
        train_ratio = len(train_df) / total_samples
        val_ratio = len(val_df) / total_samples
        test_ratio = len(test_df) / total_samples
        
        expected_train = metadata['train_ratio']
        expected_val = metadata['val_ratio']
        expected_test = metadata['test_ratio']
        
        train_diff = abs(train_ratio - expected_train)
        val_diff = abs(val_ratio - expected_val)
        test_diff = abs(test_ratio - expected_test)
        
        if all(diff < 0.02 for diff in [train_diff, val_diff, test_diff]):  # Tolleranza 2%
            print("✅ Check bilanciamento set: PASS")
            checks_passed += 1
        else:
            print(f"❌ Check bilanciamento set: FAIL")
            print(f"   Train: {train_ratio:.3f} vs {expected_train:.3f} (Δ{train_diff:.3f})")
            print(f"   Val: {val_ratio:.3f} vs {expected_val:.3f} (Δ{val_diff:.3f})")
            print(f"   Test: {test_ratio:.3f} vs {expected_test:.3f} (Δ{test_diff:.3f})")
        
        # Check 6: Presenza di tutte le classi in training
        train_classes = set(train_df['multiclass_target'].unique())
        expected_train_classes = set(range(expected_classes))
        
        if train_classes == expected_train_classes:
            print("✅ Check classi in training: PASS")
            checks_passed += 1
        else:
            missing_classes = expected_train_classes - train_classes
            print(f"❌ Check classi in training: FAIL (Classi mancanti: {missing_classes})")
        
        # Risultato finale
        print(f"\nRisultato: {checks_passed}/{total_checks} check superati")
        
        if checks_passed == total_checks:
            print("🎉 PREPROCESSING MULTICLASSE COMPLETAMENTE CONSISTENTE!")
            return True
        elif checks_passed >= total_checks - 1:
            print("⚠️  Preprocessing multiclasse quasi consistente (problemi minori)")
            return True
        else:
            print("❌ Preprocessing multiclasse con problemi significativi")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante verifica: {str(e)}")
        return False


if __name__ == "__main__":
    df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
            dataset_path="resources/datasets/NF-UNSW-NB15-v3.csv",
            config_path="config/dataset.json",
            output_dir="resources/datasets",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            min_window_size=10,
            max_window_size=30,
            label_col='Label',
            attack_col='Attack'
        )

        
        