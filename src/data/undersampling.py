def uniform_undersampling_to_minority(df, attack_col='Attack'):
    """
    Undersampling uniforme: porta tutte le classi al numero della classe minoritaria
    mantenendo distribuzione temporale uniforme senza creare buchi temporali grandi
    """
    print(f"\n--- UNDERSAMPLING UNIFORME ALLA CLASSE MINORITARIA ---")
    
    initial_samples = len(df)
    
    # Analisi distribuzione classi
    class_counts = df[attack_col].value_counts().sort_values(ascending=True)
    minority_class_name = class_counts.index[0]
    minority_class_count = class_counts.iloc[0]
    
    print(f"Distribuzione classi originale:")
    total_samples_before = 0
    for class_name, count in class_counts.items():
        percentage = (count / initial_samples) * 100
        total_samples_before += count
        marker = "👑" if class_name == minority_class_name else "📊"
        print(f"  {marker} {class_name}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nClasse minoritaria: '{minority_class_name}' con {minority_class_count:,} campioni")
    print(f"Target finale per ogni classe: {minority_class_count:,} campioni")
    
    # Calcola quanti campioni rimuovere per classe
    samples_to_remove_per_class = {}
    total_samples_to_remove = 0
    
    for class_name, count in class_counts.items():
        if class_name == minority_class_name:
            samples_to_remove_per_class[class_name] = 0
        else:
            to_remove = count - minority_class_count
            samples_to_remove_per_class[class_name] = to_remove
            total_samples_to_remove += to_remove
    
    print(f"\nCampioni da rimuovere per classe:")
    for class_name, to_remove in samples_to_remove_per_class.items():
        if to_remove > 0:
            kept_samples = class_counts[class_name] - to_remove
            reduction_pct = (to_remove / class_counts[class_name]) * 100
            print(f"  ✂️  {class_name}: -{to_remove:,} (mantieni {kept_samples:,}, -{reduction_pct:.1f}%)")
        else:
            print(f"  ✅ {class_name}: nessuna riduzione (classe minoritaria)")
    
    print(f"\nTotale campioni da rimuovere: {total_samples_to_remove:,}")
    print(f"Riduzione totale dataset: {(total_samples_to_remove/initial_samples)*100:.2f}%")
    
    # Applica undersampling uniforme per ogni classe
    indices_to_keep = []
    
    for class_name in class_counts.index:
        class_mask = df[attack_col] == class_name
        class_indices = df[class_mask].index.tolist()
        current_count = len(class_indices)
        
        if current_count <= minority_class_count:
            # Mantieni tutti i campioni (classe minoritaria)
            selected_indices = class_indices
            print(f"  {class_name}: mantieni tutti {len(selected_indices):,} campioni")
        else:
            # Undersampling uniforme mantenendo distribuzione temporale
            step = current_count / minority_class_count
            selected_indices = []
            
            for i in range(minority_class_count):
                idx_position = int(i * step)
                if idx_position < len(class_indices):
                    selected_indices.append(class_indices[idx_position])
            
            print(f"  {class_name}: {current_count:,} → {len(selected_indices):,} (step={step:.2f})")
        
        indices_to_keep.extend(selected_indices)
    
    # Mantieni ordine temporale originale
    indices_to_keep.sort()
    
    # Crea dataset sottocampionato
    df_undersampled = df.loc[indices_to_keep].copy()
    df_undersampled.reset_index(drop=True, inplace=True)
    
    # Verifica risultati finali
    final_samples = len(df_undersampled)
    final_class_counts = df_undersampled[attack_col].value_counts().sort_values(ascending=True)
    
    print(f"\n📊 Risultato undersampling uniforme:")
    print(f"  Campioni totali: {initial_samples:,} → {final_samples:,} (-{initial_samples - final_samples:,})")
    print(f"  Riduzione: {((initial_samples - final_samples)/initial_samples)*100:.2f}%")
    
    print(f"\nDistribuzione classi finale:")
    for class_name, count in final_class_counts.items():
        percentage = (count / final_samples) * 100
        expected_count = minority_class_count
        status = "✅" if count == expected_count else "⚠️"
        print(f"  {status} {class_name}: {count:,} ({percentage:.2f}%)")
    
    # Verifica bilanciamento
    unique_counts = final_class_counts.unique()
    if len(unique_counts) == 1 and unique_counts[0] == minority_class_count:
        print(f"✅ Dataset perfettamente bilanciato: tutte le classi hanno {minority_class_count:,} campioni")
    else:
        print(f"⚠️  Dataset non perfettamente bilanciato - verificare la logica")
    
    # Verifica ordine temporale (assumendo che indici crescenti = ordine temporale)
    if all(df_undersampled.index[i] <= df_undersampled.index[i+1] for i in range(len(df_undersampled)-1)):
        print("✅ Ordine temporale mantenuto")
    else:
        print("⚠️  Ordine temporale potenzialmente alterato")
    
    return df_undersampled, total_samples_to_remove

def proportional_temporal_undersampling(df, attack_col='Attack', target_percentage=0.10):
    """
    Undersampling che riduce il dataset al target_percentage mantenendo:
    - Proporzioni relative tra classi
    - Ordine temporale
    - Distribuzione uniforme nel tempo
    """
    print(f"\n--- UNDERSAMPLING PROPORZIONALE AL {target_percentage*100:.0f}% ---")
    
    initial_samples = len(df)
    target_samples = int(initial_samples * target_percentage)
    
    # Analisi distribuzione classi originale
    class_counts = df[attack_col].value_counts().sort_index()
    class_proportions = class_counts / initial_samples
    
    print(f"Dataset originale: {initial_samples:,} campioni")
    print(f"Target finale: {target_samples:,} campioni")
    print(f"\nProporzioni originali:")
    for class_name, proportion in class_proportions.items():
        original_count = class_counts[class_name]
        target_count = int(target_samples * proportion)
        print(f"  {class_name}: {proportion:.1%} ({original_count:,} → {target_count:,})")
    
    # Calcola step per ogni classe per distribuzione temporale uniforme
    selected_indices = []
    
    for class_name, original_count in class_counts.items():
        # Trova tutti gli indici per questa classe
        class_mask = df[attack_col] == class_name
        class_indices = df[class_mask].index.tolist()  # Già in ordine temporale
        
        # Calcola quanti campioni tenere per questa classe
        target_class_count = int(target_samples * class_proportions[class_name])
        
        if target_class_count == 0:
            print(f"  ⚠️  {class_name}: 0 campioni (classe troppo piccola)")
            continue
        
        if target_class_count >= original_count:
            # Mantieni tutti i campioni se il target è >= dell'originale
            class_selected = class_indices
            print(f"  ✅ {class_name}: mantieni tutti {len(class_selected):,} campioni")
        else:
            # Campionamento uniforme nel tempo
            step = len(class_indices) / target_class_count
            class_selected = []
            
            for i in range(target_class_count):
                idx_position = int(i * step)
                if idx_position < len(class_indices):
                    class_selected.append(class_indices[idx_position])
            
            print(f"  📊 {class_name}: {original_count:,} → {len(class_selected):,} (step={step:.2f})")
        
        selected_indices.extend(class_selected)
    
    # Mantieni ordine temporale globale
    selected_indices.sort()
    
    # Crea dataset sottocampionato
    df_undersampled = df.loc[selected_indices].copy()
    df_undersampled.reset_index(drop=True, inplace=True)
    
    # Verifica risultati
    final_samples = len(df_undersampled)
    final_class_counts = df_undersampled[attack_col].value_counts().sort_index()
    final_proportions = final_class_counts / final_samples
    
    print(f"\n📊 Risultato undersampling proporzionale:")
    print(f"  Campioni totali: {initial_samples:,} → {final_samples:,}")
    print(f"  Riduzione effettiva: {((initial_samples - final_samples)/initial_samples)*100:.1f}%")
    print(f"  Target: {target_percentage*100:.0f}%, Ottenuto: {(final_samples/initial_samples)*100:.1f}%")
    
    print(f"\nConfronto proporzioni:")
    print(f"{'Classe':<20} {'Originale':<10} {'Finale':<10} {'Diff':<8}")
    print("-" * 50)
    for class_name in class_counts.index:
        if class_name in final_class_counts.index:
            orig_prop = class_proportions[class_name]
            final_prop = final_proportions[class_name]
            diff = abs(orig_prop - final_prop)
            print(f"{class_name:<20} {orig_prop:.1%}      {final_prop:.1%}      {diff:.2%}")
        else:
            print(f"{class_name:<20} {class_proportions[class_name]:.1%}      0.0%       -")
    
    # Verifica ordine temporale
    if all(selected_indices[i] <= selected_indices[i+1] for i in range(len(selected_indices)-1)):
        print("✅ Ordine temporale mantenuto")
    else:
        print("⚠️  Ordine temporale compromesso")
    
    return df_undersampled, initial_samples - final_samples