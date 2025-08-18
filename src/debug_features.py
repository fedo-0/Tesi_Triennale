"""
Debug per confrontare features binario vs multiclasse
"""

import pandas as pd
import json
import os

def debug_feature_comparison():
    """Confronta features tra binario e multiclasse"""
    
    print("🔍 DEBUG: CONFRONTO FEATURES BINARIO VS MULTICLASS")
    print("=" * 60)
    
    try:
        # 1. Carica configurazione
        with open("../config/dataset.json", 'r') as f:
            config = json.load(f)['dataset']
        
        numeric_cols = config['numeric_columns']
        categorical_cols = config['categorical_columns']
        expected_features = len(numeric_cols) + len(categorical_cols)
        
        print(f"📋 CONFIGURAZIONE:")
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"  Categorical columns: {len(categorical_cols)}")
        print(f"  Total expected features: {expected_features}")
        
        # 2. Controlla CSV originale
        if os.path.exists("../resources/datasets/NF-UNSW-NB15-v3.csv"):
            original_df = pd.read_csv("../resources/datasets/NF-UNSW-NB15-v3.csv")
            print(f"\n📊 CSV ORIGINALE:")
            print(f"  Total columns: {len(original_df.columns)}")
            print(f"  Primi 5: {list(original_df.columns[:5])}")
            print(f"  Ultimi 5: {list(original_df.columns[-5:])}")
        
        # 3. Controlla dataset binario
        if os.path.exists("../resources/datasets/train.csv"):
            binary_train = pd.read_csv("../resources/datasets/train.csv")
            binary_features = len(binary_train.columns) - 1  # Escludi target
            
            print(f"\n✅ DATASET BINARIO:")
            print(f"  Total columns: {len(binary_train.columns)}")
            print(f"  Features (escluso target): {binary_features}")
            print(f"  Target column: presumibilmente '{binary_train.columns[-1]}'")
            print(f"  Match con config: {'✅' if binary_features == expected_features else '❌'}")
        else:
            print(f"\n❌ DATASET BINARIO: Non trovato")
        
        # 4. Controlla dataset multiclass
        if os.path.exists("../resources/datasets/train_multiclass.csv"):
            multi_train = pd.read_csv("../resources/datasets/train_multiclass.csv")
            multi_features = len(multi_train.columns) - 1  # Escludi target
            
            print(f"\n🎯 DATASET MULTICLASS:")
            print(f"  Total columns: {len(multi_train.columns)}")
            print(f"  Features (escluso target): {multi_features}")
            print(f"  Target column: presumibilmente '{multi_train.columns[-1]}'")
            print(f"  Match con config: {'✅' if multi_features == expected_features else '❌'}")
            
            # Confronto diretto
            if os.path.exists("../resources/datasets/train.csv"):
                print(f"\n🔄 CONFRONTO DIRETTO:")
                print(f"  Binario features: {binary_features}")
                print(f"  Multiclass features: {multi_features}")
                print(f"  Sono uguali: {'✅' if binary_features == multi_features else '❌'}")
                
                if binary_features != multi_features:
                    print(f"  Differenza: {abs(binary_features - multi_features)} features")
        else:
            print(f"\n❌ DATASET MULTICLASS: Non trovato")
        
        # 5. Controlla metadati multiclass
        if os.path.exists("../resources/datasets/multiclass_metadata.json"):
            with open("../resources/datasets/multiclass_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            metadata_features = len(metadata.get('feature_columns', []))
            print(f"\n📋 METADATI MULTICLASS:")
            print(f"  Features in metadata: {metadata_features}")
            print(f"  Number of classes: {metadata.get('n_classes', 'N/A')}")
            print(f"  Match con config: {'✅' if metadata_features == expected_features else '❌'}")
        else:
            print(f"\n❌ METADATI MULTICLASS: Non trovati")
        
        print(f"\n" + "=" * 60)
        print("Debug completato!")
        
    except Exception as e:
        print(f"❌ ERRORE durante debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_comparison()