import pandas as pd
import numpy as np
import json
import os
from data.windowing import sliding_window_split_multiclass
from data.undersampling import uniform_undersampling_to_minority

def load_dataset_config(config_path="config/dataset.json"):
    """Carica la configurazione del dataset"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['dataset']

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

def clean_and_split_dataset(dataset_path, config_path, output_dir,
                           train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                           window_size=10,
                           label_col='Label', attack_col='Attack',
                           min_samples_per_class=10000):
    """
    Pipeline principale per pulizia e divisione del dataset in train/val/test
    Salva i 3 set puliti ma NON processati per permettere processing specifici per architettura
    """
    
    print("PIPELINE DI PULIZIA E DIVISIONE DATASET")
    print("=" * 60)
    
    # Carica configurazione
    config = load_dataset_config(config_path)
    numeric_columns = config['numeric_columns']
    categorical_columns = config['categorical_columns']
    
    print(f"\nConfigurazione caricata:")
    print(f"- Colonne numeriche: {len(numeric_columns)}")
    print(f"- Colonne categoriche: {len(categorical_columns)}")
    print(f"- Split ratio: {train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}")
    print(f"- Dimensione finestra: {window_size}")
    print(f"- Soglia minima campioni per classe: {min_samples_per_class:,}")
    
    # Carica dataset
    print(f"\nCaricamento dataset da: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset originale: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    # === FASE 1: PULIZIA VALORI PROBLEMATICI ===
    print(f"\n=== FASE 1: PULIZIA VALORI PROBLEMATICI ===")
    
    initial_rows = len(df)
    numeric_cols_present = [col for col in numeric_columns if col in df.columns]

    # Gestione valori infiniti
    if numeric_cols_present:
        inf_count = 0
        for col in numeric_cols_present:
            col_inf_count = np.isinf(df[col]).sum()
            inf_count += col_inf_count
        
        if inf_count > 0:
            print(f"Sostituiti {inf_count:,} valori infiniti con NaN")
            df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], np.nan)
        else:
            print("✅ Nessun valore infinito trovato")

    # Rimozione righe con NaN
    df_clean = df.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        print(f"Rimosse {removed_rows:,} righe contenenti valori NaN")
        print(f"Riduzione: {(removed_rows/initial_rows)*100:.2f}%")
    else:
        print("✅ Nessuna riga con NaN da rimuovere")
    
    print(f"Dataset dopo pulizia: {df_clean.shape[0]:,} righe, {df_clean.shape[1]} colonne")

    if len(df_clean) == 0:
        raise ValueError("Il dataset risulta vuoto dopo la rimozione dei valori NaN!")
    
    # Verifica colonne target
    if label_col not in df_clean.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset!")
    if attack_col not in df_clean.columns:
        raise ValueError(f"Colonna attack '{attack_col}' non trovata nel dataset!")
    
    # Verifica features configurate
    expected_features = numeric_columns + categorical_columns
    feature_columns = [col for col in expected_features if col in df_clean.columns]
    missing_features = [col for col in expected_features if col not in df_clean.columns]
    
    print(f"\nFeatures verificate:")
    print(f"- Trovate: {len(feature_columns)} di {len(expected_features)} configurate")
    if missing_features:
        print(f"- Mancanti: {missing_features}")

    # === FASE 2: FILTRAGGIO CLASSI RARE ===
    print(f"\n=== FASE 2: FILTRAGGIO CLASSI RARE ===")
    
    # Analisi distribuzione iniziale
    analyze_multiclass_distribution(df_clean, label_col, attack_col)
    
    # Filtra classi rare
    df_filtered, removed_classes = filter_rare_classes(df_clean, attack_col, min_samples=min_samples_per_class)
    
    if removed_classes:
        print(f"\n⚠️  Dataset filtrato: rimosse {len(removed_classes)} classi rare")
        df_clean = df_filtered
        
        # Analisi distribuzione dopo filtraggio
        analyze_multiclass_distribution(df_clean, label_col, attack_col)
    
    """
    # === FASE 3: UNDERSAMPLING UNIFORME ===
    print(f"\n=== FASE 3: UNDERSAMPLING UNIFORME ===")
    
    df_undersampled, total_removed_samples = uniform_undersampling_to_minority(df_clean, attack_col)

    if total_removed_samples > 0:
        print(f"\n⚠️  Applicato undersampling uniforme: rimosse {total_removed_samples:,} campioni")
        df_clean = df_undersampled
        
        # Analisi distribuzione finale
        analyze_multiclass_distribution(df_clean, label_col, attack_col)
    else:
        print("✅ Nessun undersampling necessario")
    """

    # === FASE 4: DIVISIONE CON MICRO-FINESTRE ===
    print(f"\n=== FASE 4: DIVISIONE CON MICRO-FINESTRE ===")
    
    # Split con micro-finestre temporali
    train_data, val_data, test_data = sliding_window_split_multiclass(
        df_clean, window_size, train_ratio, val_ratio, test_ratio
    )
    
    # Verifica split
    total_after_split = len(train_data) + len(val_data) + len(test_data)
    print(f"\nVerifica split:")
    print(f"- Dataset pre-split: {len(df_clean):,} campioni")
    print(f"- Dataset post-split: {total_after_split:,} campioni")
    print(f"- Training: {len(train_data):,} ({len(train_data)/total_after_split*100:.1f}%)")
    print(f"- Validation: {len(val_data):,} ({len(val_data)/total_after_split*100:.1f}%)")
    print(f"- Test: {len(test_data):,} ({len(test_data)/total_after_split*100:.1f}%)")
    
    if total_after_split != len(df_clean):
        print(f"⚠️  Differenza: {len(df_clean) - total_after_split:,} campioni")
    
    # === FASE 5: SALVATAGGIO ===
    print(f"\n=== FASE 5: SALVATAGGIO ===")
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths dei file di output
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    # Salva i dataset
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Dataset salvati:")
    print(f"- Training: {train_path} ({train_data.shape[0]:,} righe)")
    print(f"- Validation: {val_path} ({val_data.shape[0]:,} righe)")
    print(f"- Test: {test_path} ({test_data.shape[0]:,} righe)")
    
    class_distribution_dict = {}
    for class_name, count in df_clean[attack_col].value_counts().items():
        class_distribution_dict[str(class_name)] = int(count)
    
    # Salva metadati dello split
    split_metadata = {
        'original_dataset_path': dataset_path,
        'original_samples': int(initial_rows),
        'cleaned_samples': int(len(df_clean)),
        'final_samples': {
            'train': int(len(train_data)),
            'val': int(len(val_data)),
            'test': int(len(test_data)),
            'total': int(total_after_split)
        },
        'split_ratios': {
            'train_ratio': float(train_ratio),
            'val_ratio': float(val_ratio),
            'test_ratio': float(test_ratio)
        },
        'split_parameters': {
            'window_size': int(window_size),
            'min_samples_per_class': int(min_samples_per_class)
        },
        'cleaning_stats': {
            'removed_nan_rows': int(removed_rows),
            'removed_rare_classes': removed_classes,
        #    'undersampled_samples': int(total_removed_samples)
        },
        'columns_info': {
            'label_col': label_col,
            'attack_col': attack_col,
            'feature_columns': feature_columns,
            'numeric_columns': [col for col in numeric_columns if col in feature_columns],
            'categorical_columns': [col for col in categorical_columns if col in feature_columns],
            'missing_features': missing_features
        },
        'class_distribution': {
            'final_classes': class_distribution_dict,
            'n_classes': int(df_clean[attack_col].nunique())
        },
        'split_version': 'clean_split_v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Salva metadati
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"- Metadati: {metadata_path}")
    
    # === STATISTICHE FINALI ===
    print(f"\n=== STATISTICHE FINALI ===")
    
    # Distribuzione classi per ogni set
    for set_name, dataset in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        total = len(dataset)
        class_counts = dataset[attack_col].value_counts().sort_index()
        print(f"\n{set_name} Set: {total:,} campioni")
        
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            class_type = "Benigno" if class_name.lower() in ['benign', 'normal'] else "Attacco"
            print(f"  {class_name}: {count:,} ({percentage:.2f}%) - {class_type}")
    
    # Riepilogo generale
    print(f"\n🎯 RIEPILOGO OPERAZIONI:")
    print(f"✅ Dataset caricato: {initial_rows:,} campioni originali")
    print(f"✅ Pulizia completata: -{removed_rows:,} righe con NaN/inf")
    if removed_classes:
        print(f"✅ Classi rare filtrate: -{len(removed_classes)} classi")
    #if total_removed_samples > 0:
    #    print(f"✅ Undersampling applicato: -{total_removed_samples:,} campioni")
    print(f"✅ Split temporale completato: {total_after_split:,} campioni finali")
    print(f"✅ Dataset pronti per processing architettura-specifico")
    
    print(f"\n📁 File generati in {output_dir}:")
    print(f"   - train.csv ({len(train_data):,} campioni)")
    print(f"   - val.csv ({len(val_data):,} campioni)")  
    print(f"   - test.csv ({len(test_data):,} campioni)")
    print(f"   - split_metadata.json (metadati operazione)")
    
    return train_data, val_data, test_data, split_metadata


if __name__ == "__main__":
    train_data, val_data, test_data, metadata = clean_and_split_dataset(
        dataset_path="resources/datasets/dataset_ton_v3.csv",
        config_path="config/dataset.json",
        output_dir="resources/datasets",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        window_size=10,
        label_col='Label',
        attack_col='Attack',
        min_samples_per_class=10000
    )