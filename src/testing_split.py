# ===================================================================
# TEST DECISIVO: SPLIT TEMPORALE VS RANDOM PER MLP
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def analyze_temporal_distribution(dataset_path="resources/datasets/NF-UNSW-NB15-v3.csv"):
    """Analizza se la distribuzione dei dati cambia nel tempo"""
    
    print("🕒 ANALISI TEMPORALITÀ DATASET")
    print("="*50)
    
    # Carica dataset originale
    df = pd.read_csv(dataset_path)
    print(f"Dataset caricato: {df.shape}")
    
    # Dividi in chunks temporali
    n_chunks = 10
    chunk_size = len(df) // n_chunks
    
    temporal_stats = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
        
        chunk = df.iloc[start_idx:end_idx]
        
        # Analizza distribuzione Attack per chunk
        attack_dist = chunk['Attack'].value_counts(normalize=True)
        
        # Analizza distribuzione Label (binaria) per chunk
        if 'Label' in chunk.columns:
            binary_dist = chunk['Label'].value_counts(normalize=True)
            attack_ratio = binary_dist.get(1, 0)  # Proporzione attacchi
        else:
            attack_ratio = 1 - attack_dist.get('Benign', 0)
        
        temporal_stats.append({
            'chunk': i + 1,
            'start_pct': start_idx / len(df) * 100,
            'end_pct': end_idx / len(df) * 100,
            'attack_ratio': attack_ratio,
            'n_attack_types': len(attack_dist) - (1 if 'Benign' in attack_dist else 0),
            'top_attack': attack_dist.drop('Benign', errors='ignore').index[0] if len(attack_dist.drop('Benign', errors='ignore')) > 0 else 'None',
            'attack_diversity': attack_dist.drop('Benign', errors='ignore').std() if len(attack_dist.drop('Benign', errors='ignore')) > 1 else 0
        })
    
    # Crea DataFrame per analisi
    temporal_df = pd.DataFrame(temporal_stats)
    
    print("\n📊 DISTRIBUZIONE TEMPORALE:")
    print(temporal_df[['chunk', 'start_pct', 'attack_ratio', 'n_attack_types', 'top_attack']].to_string(index=False))
    
    # Test statistico: cambia la distribuzione nel tempo?
    attack_ratios = temporal_df['attack_ratio'].values
    ratio_variation = np.std(attack_ratios) / np.mean(attack_ratios)  # Coefficient of variation
    
    print(f"\n🧮 STATISTICHE TEMPORALI:")
    print(f"   Attack ratio medio: {np.mean(attack_ratios):.3f}")
    print(f"   Attack ratio std: {np.std(attack_ratios):.3f}")
    print(f"   Coefficient of variation: {ratio_variation:.3f}")
    
    # Interpreta risultati
    if ratio_variation > 0.3:
        temporal_significance = "ALTA"
        recommendation = "✅ Split temporale CONSIGLIATO"
        explanation = "La distribuzione cambia significativamente nel tempo"
    elif ratio_variation > 0.1:
        temporal_significance = "MEDIA"
        recommendation = "🟡 Split temporale UTILE"
        explanation = "Cambiamenti moderati nel tempo"
    else:
        temporal_significance = "BASSA"
        recommendation = "❌ Split temporale NON necessario"
        explanation = "Distribuzione stabile nel tempo"
    
    print(f"\n🎯 VERDETTO:")
    print(f"   Significatività temporale: {temporal_significance}")
    print(f"   {recommendation}")
    print(f"   Motivo: {explanation}")
    
    # Plot visualizzazione
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Attack ratio nel tempo
    ax1.plot(temporal_df['chunk'], temporal_df['attack_ratio'], marker='o', linewidth=2, markersize=6)
    ax1.set_title('Attack Ratio nel Tempo')
    ax1.set_xlabel('Chunk Temporale')
    ax1.set_ylabel('Proporzione Attacchi')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(attack_ratios), color='red', linestyle='--', alpha=0.7, label=f'Media: {np.mean(attack_ratios):.3f}')
    ax1.legend()
    
    # Numero tipi di attacco
    ax2.bar(temporal_df['chunk'], temporal_df['n_attack_types'], alpha=0.7, color='orange')
    ax2.set_title('Diversità Attacchi nel Tempo')
    ax2.set_xlabel('Chunk Temporale')
    ax2.set_ylabel('N° Tipi di Attacco')
    ax2.grid(True, alpha=0.3)
    
    # Heatmap attacchi principali
    chunk_attacks = []
    for _, row in temporal_df.iterrows():
        chunk_data = df.iloc[int(row['start_pct']/100*len(df)):int(row['end_pct']/100*len(df))]
        attack_dist = chunk_data['Attack'].value_counts()
        chunk_attacks.append(attack_dist.drop('Benign', errors='ignore'))
    
    # Top 5 attacchi più comuni
    all_attacks = set()
    for attacks in chunk_attacks:
        all_attacks.update(attacks.index[:5])  # Top 5 per chunk
    
    attack_matrix = []
    for attacks in chunk_attacks:
        row = [attacks.get(attack, 0) for attack in sorted(all_attacks)]
        attack_matrix.append(row)
    
    if len(all_attacks) > 0:
        attack_matrix = np.array(attack_matrix)
        # Normalizza per riga
        attack_matrix_norm = attack_matrix / (attack_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        sns.heatmap(attack_matrix_norm.T, 
                   xticklabels=[f'Chunk {i+1}' for i in range(n_chunks)],
                   yticklabels=sorted(all_attacks),
                   cmap='YlOrRd', 
                   ax=ax3)
        ax3.set_title('Distribuzione Attacchi nel Tempo')
        ax3.set_xlabel('Chunk Temporale')
    else:
        ax3.text(0.5, 0.5, 'Nessun attacco trovato', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Distribuzione Attacchi nel Tempo')
    
    # Attack diversity evolution
    ax4.plot(temporal_df['chunk'], temporal_df['attack_diversity'], marker='s', linewidth=2, markersize=6, color='purple')
    ax4.set_title('Diversità Attacchi (Std Dev)')
    ax4.set_xlabel('Chunk Temporale')
    ax4.set_ylabel('Standard Deviation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return temporal_df, ratio_variation, recommendation


def compare_split_strategies(dataset_path="resources/datasets/NF-UNSW-NB15-v3.csv"):
    """Confronta split temporale vs random per le performance"""
    
    print("\n🎲 CONFRONTO SPLIT STRATEGIES")
    print("="*50)
    
    df = pd.read_csv(dataset_path)
    
    # Test 1: Split temporale (come fai tu)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    temporal_train = df.iloc[:train_size]
    temporal_val = df.iloc[train_size:train_size + val_size]
    temporal_test = df.iloc[train_size + val_size:]
    
    # Test 2: Split random
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    random_train = df_shuffled.iloc[:train_size]
    random_val = df_shuffled.iloc[train_size:train_size + val_size]
    random_test = df_shuffled.iloc[train_size + val_size:]
    
    # Confronta distribuzioni
    def analyze_split_distribution(train_df, val_df, test_df, split_name):
        print(f"\n📊 {split_name.upper()} SPLIT:")
        
        sets = {'Train': train_df, 'Val': val_df, 'Test': test_df}
        distributions = {}
        
        for set_name, dataset in sets.items():
            attack_dist = dataset['Attack'].value_counts(normalize=True)
            
            # Calcola attack ratio
            attack_ratio = 1 - attack_dist.get('Benign', 0)
            
            # Calcola diversity (numero classi non-benign)
            n_attack_types = len(attack_dist.drop('Benign', errors='ignore'))
            
            distributions[set_name] = {
                'attack_ratio': attack_ratio,
                'n_attack_types': n_attack_types,
                'size': len(dataset)
            }
            
            print(f"   {set_name}: Attack ratio={attack_ratio:.3f}, "
                  f"Attack types={n_attack_types}, Size={len(dataset):,}")
        
        # Calcola stabilità tra set
        attack_ratios = [dist['attack_ratio'] for dist in distributions.values()]
        stability = 1 - (np.std(attack_ratios) / np.mean(attack_ratios))  # 1 = perfettamente stabile
        
        print(f"   📈 Stabilità tra set: {stability:.3f} (1.0 = perfetto)")
        
        return distributions, stability
    
    # Analizza entrambi gli split
    temporal_dist, temporal_stability = analyze_split_distribution(
        temporal_train, temporal_val, temporal_test, "Temporal"
    )
    
    random_dist, random_stability = analyze_split_distribution(
        random_train, random_val, random_test, "Random"
    )
    
    # Confronto finale
    print(f"\n🏆 CONFRONTO FINALE:")
    print(f"   Temporal split stabilità: {temporal_stability:.3f}")
    print(f"   Random split stabilità: {random_stability:.3f}")
    
    if random_stability > temporal_stability + 0.05:
        winner = "RANDOM SPLIT"
        explanation = "Random split ha distribuzione più stabile tra train/val/test"
        recommendation_split = "🎲 USA SPLIT RANDOM"
    elif temporal_stability > random_stability + 0.05:
        winner = "TEMPORAL SPLIT"
        explanation = "Temporal split necessario per pattern temporali"
        recommendation_split = "⏰ MANTIENI SPLIT TEMPORALE"
    else:
        winner = "PAREGGIO"
        explanation = "Entrambi gli split hanno stabilità simile"
        recommendation_split = "🤝 ENTRAMBI VALIDI - scelta libera"
    
    print(f"   🏅 Vincitore: {winner}")
    print(f"   📝 Motivo: {explanation}")
    print(f"   ✅ {recommendation_split}")
    
    return temporal_dist, random_dist, recommendation_split


def final_recommendation_for_mlp():
    """Raccomandazione finale per MLP"""
    
    print(f"\n🎯 RACCOMANDAZIONE FINALE PER MLP")
    print("="*50)
    
    print("🤔 DOMANDE CHIAVE:")
    print("1. Il tuo dataset ha concept drift temporale?")
    print("2. Vuoi simulare deployment reale (predire il futuro)?")
    print("3. La distribuzione è stabile nel tempo?")
    
    print(f"\n📋 LINEE GUIDA:")
    print("✅ USA SPLIT TEMPORALE se:")
    print("   • Coefficient of variation > 0.1")
    print("   • Vuoi robustezza a concept drift")
    print("   • Simuli scenario produzione reale")
    print("   • Dataset raccolto su periodo lungo")
    
    print(f"\n❌ USA SPLIT RANDOM se:")
    print("   • Coefficient of variation < 0.1")
    print("   • Distribuzione perfettamente stabile")
    print("   • Vuoi performance ottimali su test")
    print("   • Dataset è snapshot temporale uniforme")
    
    print(f"\n🎲 COMPROMESSO:")
    print("   • Testa ENTRAMBI gli approcci")
    print("   • Usa quello con migliori performance")
    print("   • Per tesi: giustifica la scelta con i test")


if __name__ == "__main__":
    # Esegui analisi completa
    print("🔍 ANALISI COMPLETA TEMPORALITÀ vs MLP")
    print("="*60)
    
    # Test 1: Analizza variazione temporale
    temporal_df, variation, temp_recommendation = analyze_temporal_distribution()
    
    # Test 2: Confronta split strategies  
    temporal_dist, random_dist, split_recommendation = compare_split_strategies()
    
    # Raccomandazione finale
    final_recommendation_for_mlp()
    
    print(f"\n🎯 VERDETTO FINALE:")
    print(f"   Variazione temporale: {variation:.3f}")
    print(f"   {temp_recommendation}")
    print(f"   {split_recommendation}")
    
    if "TEMPORALE" in temp_recommendation or "TEMPORAL" in split_recommendation:
        print(f"\n✅ MANTIENI il tuo split temporale attuale")
        print(f"   È giustificato anche per MLP nel tuo caso")
    else:
        print(f"\n🎲 CONSIDERA split random per migliori performance")
        print(f"   Il tuo dataset sembra temporalmente stabile")