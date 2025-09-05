import numpy as np

def create_sliding_windows_indices(df, window_size=10):
    """
    Crea sliding windows statiche di dimensione fissa
    """
    print(f"\n--- CREAZIONE SLIDING WINDOWS STATICHE (dimensione: {window_size} pacchetti) ---")
    
    sliding_windows = []
    window_id = 0
    total_len = len(df)
    
    # Progresso
    progress_interval = max(1, (total_len // window_size) // 100)  # Mostra progresso ogni 1%
    
    # Crea finestre di dimensione fissa senza sovrapposizione
    for start_pos in range(0, total_len, window_size):
        end_pos = min(start_pos + window_size, total_len)
        
        # Skip finestre troppo piccole (meno del 50% della dimensione target)
        if (end_pos - start_pos) < (window_size * 0.5):
            print(f"  Saltata ultima finestra troppo piccola: {end_pos - start_pos} pacchetti")
            break
        
        # SOLO INDICI - nessuna copia dei dati
        window_info = {
            'id': window_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'size': end_pos - start_pos
        }
        
        sliding_windows.append(window_info)
        window_id += 1
        
        # Mostra progresso
        if window_id % progress_interval == 0 or end_pos >= total_len:
            progress = (end_pos / total_len) * 100
            print(f"  Progresso: {progress:.1f}% ({window_id:,} finestre)")
    
    print(f"Totale sliding windows create: {len(sliding_windows):,}")
    
    # Statistiche finali
    if sliding_windows:
        sizes = [w['size'] for w in sliding_windows]
        total_packets_covered = sum(sizes)
        coverage_percentage = (total_packets_covered / total_len) * 100
        
        print(f"\nStatistiche Sliding Windows:")
        print(f"  Dimensione target: {window_size} pacchetti")
        print(f"  Dimensione media effettiva: {np.mean(sizes):.1f} pacchetti")
        print(f"  Range dimensioni: {min(sizes)}-{max(sizes)} pacchetti")
        print(f"  Pacchetti coperti: {total_packets_covered:,} di {total_len:,} ({coverage_percentage:.2f}%)")
        print(f"  Pacchetti scartati: {total_len - total_packets_covered:,}")
    
    return sliding_windows

def round_robin_window_assignment(sliding_windows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Assegna le sliding windows ai set usando round-robin per mantenere ordine temporale
    """
    print(f"\n--- ASSEGNAZIONE ROUND-ROBIN SLIDING WINDOWS ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f}) ---")
    
    total_windows = len(sliding_windows)
    
    # Calcola il pattern di assegnazione basato sui ratio
    # Per 70-15-15: ogni 20 finestre -> 14 train, 3 val, 3 test
    cycle_length = 20
    train_per_cycle = int(cycle_length * train_ratio)  # 14
    val_per_cycle = int(cycle_length * val_ratio)      # 3
    test_per_cycle = cycle_length - train_per_cycle - val_per_cycle  # 3
    
    print(f"Pattern round-robin ogni {cycle_length} finestre:")
    print(f"  Train: {train_per_cycle} finestre")
    print(f"  Validation: {val_per_cycle} finestre") 
    print(f"  Test: {test_per_cycle} finestre")
    
    train_windows, val_windows, test_windows = [], [], []
    
    # Assegnazione round-robin mantenendo ordine temporale
    for i, window in enumerate(sliding_windows):
        cycle_pos = i % cycle_length
        
        if cycle_pos < train_per_cycle:  # Prime 14 posizioni -> train
            train_windows.append(window)
        elif cycle_pos < train_per_cycle + val_per_cycle:  # Successive 3 -> val
            val_windows.append(window)
        else:  # Ultime 3 -> test
            test_windows.append(window)
    
    # Le finestre sono già in ordine temporale per costruzione

    print(f"\nTotale finestre assegnate:")
    print(f"  Train: {len(train_windows):,} finestre ({len(train_windows)/total_windows*100:.1f}%)")
    print(f"  Validation: {len(val_windows):,} finestre ({len(val_windows)/total_windows*100:.1f}%)")
    print(f"  Test: {len(test_windows):,} finestre ({len(test_windows)/total_windows*100:.1f}%)")
    
    # Verifica ordine temporale
    def check_temporal_order(windows, set_name):
        if len(windows) < 2:
            return True
        
        for i in range(1, len(windows)):
            if windows[i]['start_pos'] <= windows[i-1]['start_pos']:
                print(f"⚠️  Ordine temporale violato in {set_name} alla posizione {i}")
                return False
        return True
    
    train_ordered = check_temporal_order(train_windows, "Train")
    val_ordered = check_temporal_order(val_windows, "Validation") 
    test_ordered = check_temporal_order(test_windows, "Test")
    
    if train_ordered and val_ordered and test_ordered:
        print("✅ Ordine temporale mantenuto in tutti i set")
    
    return train_windows, val_windows, test_windows

def sliding_window_split_multiclass(df, window_size=10,
                                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split semplificato con sliding windows statiche per multiclass - MEMORY OPTIMIZED
    """
    print("SLIDING WINDOW STRATIFIED TEMPORAL SPLIT - MULTICLASS (SEMPLIFICATO)")
    print("=" * 70)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "I ratio devono sommare a 1.0"
    
    # Crea sliding windows statiche (solo indici)
    sliding_windows = create_sliding_windows_indices(df, window_size)
    
    if len(sliding_windows) < 3:
        raise ValueError(f"Troppo poche sliding windows ({len(sliding_windows)}) per creare train/val/test set. "
                        f"Prova a ridurre window_size o aumentare la dimensione del dataset.")
    
    # Assegnazione round-robin
    train_windows, val_windows, test_windows = round_robin_window_assignment(
        sliding_windows, train_ratio, val_ratio, test_ratio
    )
    
    # Estrazione dataset usando indici (riusa la funzione esistente)
    train_data, val_data, test_data = extract_datasets_from_indices(
        df, train_windows, val_windows, test_windows
    )
    
    return train_data, val_data, test_data

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