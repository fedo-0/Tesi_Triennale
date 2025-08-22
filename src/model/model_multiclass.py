import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from utilities.logging_config import setup_logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class NetworkTrafficMLPMulticlass(nn.Module):
    """
    Multi-Layer Perceptron per classificazione MULTICLASS di traffico di rete
    """
    def __init__(self, input_dim, config, n_classes, class_weights=None):
        super(NetworkTrafficMLPMulticlass, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = config['embedding_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Input validation
        if self.num_layers < 2:
            raise ValueError("num_layers deve essere almeno 2 (input + output)")
        if self.n_classes < 2:
            raise ValueError("n_classes deve essere almeno 2")
        
        # Lista per memorizzare i layer
        layers = []
        
        # Primo layer: input -> hidden_dim
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Layer intermedi: hidden_dim -> hidden_dim decrescente
        prev_hidden_dim = self.hidden_dim
        for i in range(self.num_layers - 2):
            next_hidden_dim = max(self.hidden_dim // (2 ** (i + 1)), 32)
            
            layers.append(nn.Linear(prev_hidden_dim, next_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(next_hidden_dim))
            layers.append(nn.Dropout(self.dropout))
            
            prev_hidden_dim = next_hidden_dim
        
        # Layer finale: ultimo_hidden -> n_classes
        final_input_dim = prev_hidden_dim if self.num_layers > 2 else self.hidden_dim
        layers.append(nn.Linear(final_input_dim, self.n_classes))
        
        # Combina tutti i layer
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi
        self.apply(self._init_weights)
        
        # Salva class weights per riferimento
        self.class_weights = class_weights
        
        # Log informazioni modello
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Modello MULTICLASS creato:")
        logger.info(f"  Parametri totali: {total_params:,}")
        logger.info(f"  Classi: {n_classes}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Layers: {self.num_layers}")
    
    def _init_weights(self, module):
        """Inizializzazione dei pesi Xavier"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class NetworkTrafficDatasetMulticlass:
    """Classe per gestire il caricamento dati MULTICLASS"""
    
    def __init__(self, config_path="config/dataset.json", hyperparams_path="config/hyperparameters.json", 
                 model_size="small", metadata_path="resources/datasets/multiclass_metadata.json"):
        self.metadata_path = metadata_path
        self.load_configs(config_path, hyperparams_path, model_size)
        
    def load_configs(self, config_path, hyperparams_path, model_size="small"):
        """Carica le configurazioni"""
        try:
            with open(config_path, 'r') as f:
                self.dataset_config = json.load(f)['dataset']
        except:
            logger.warning(f"Impossibile caricare {config_path}, usando config di default")
            self.dataset_config = {}
        
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams_data = json.load(f)
                self.hyperparams = hyperparams_data[model_size]
        except:
            logger.warning(f"Impossibile caricare {hyperparams_path}, usando hyperparams di default")
            self.hyperparams = {
                'batch_size': 256,
                'lr': 0.001,
                'weight_decay': 1e-4,
                'embedding_dim': 128,
                'num_layers': 3,
                'dropout': 0.3
            }
        
        # Carica metadati multiclass
        with open(self.metadata_path, 'r') as f:
            self.multiclass_metadata = json.load(f)
            
        self.feature_columns = self.multiclass_metadata['feature_columns']
        self.target_column = 'multiclass_target'
        self.n_classes = self.multiclass_metadata['n_classes']
        self.class_mapping = self.multiclass_metadata['class_mapping']
        
        logger.info(f"Configurazione caricata:")
        logger.info(f"  Feature columns: {len(self.feature_columns)}")
        logger.info(f"  Target column: {self.target_column}")
        logger.info(f"  Classi: {self.n_classes}")
        logger.info(f"  Hyperparams: {self.hyperparams}")
    
    def analyze_data_distribution_multiclass(self, df, set_name):
        """Analizza la distribuzione multiclass - VERSIONE CORRETTA"""
        target_values = df[self.target_column].value_counts().sort_index()
        total = len(df)
        
        logger.info(f"\n{set_name} Set Distribution (Multiclass):")
        for class_idx, count in target_values.items():
            class_name = self.multiclass_metadata['label_encoder_classes'][class_idx]
            percentage = (count / total) * 100
            logger.info(f"  {class_name} ({class_idx}): {count:,} ({percentage:.2f}%)")
        
        # Controlla solo feature numeriche per problemi
        numeric_features = []
        problematic_features = []
        
        for col in self.feature_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_features.append(col)
                    
                    # Controlla NaN
                    if df[col].isnull().any():
                        problematic_features.append(f"{col} (NaN)")
                    
                    # Controlla infiniti
                    if np.isinf(df[col]).any():
                        problematic_features.append(f"{col} (Inf)")
                else:
                    problematic_features.append(f"{col} (non-numeric)")
            else:
                problematic_features.append(f"{col} (missing)")
        
        if problematic_features:
            logger.warning(f"⚠️  {set_name} - Feature problematiche: {problematic_features[:5]}...")
        else:
            logger.info(f"✅ {set_name} - Tutte le feature sono valide")
        
        return target_values
    
    def load_data(self, train_path, val_path, test_path):
        """Carica i dataset preprocessati multiclass - VERSIONE CORRETTA"""
        logger.info("Caricamento dataset multiclass...")
        
        # Carica i CSV
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        
        logger.info(f"Dataset caricati:")
        logger.info(f"  Training: {self.df_train.shape}")
        logger.info(f"  Validation: {self.df_val.shape}")
        logger.info(f"  Test: {self.df_test.shape}")
        
        # Analizza distribuzione
        train_dist = self.analyze_data_distribution_multiclass(self.df_train, "Training")
        val_dist = self.analyze_data_distribution_multiclass(self.df_val, "Validation")
        test_dist = self.analyze_data_distribution_multiclass(self.df_test, "Test")
        
        # Verifica feature - VERSIONE MIGLIORATA
        missing_features = []
        invalid_features = []
        
        for feature in self.feature_columns:
            if feature not in self.df_train.columns:
                missing_features.append(feature)
            elif not pd.api.types.is_numeric_dtype(self.df_train[feature]):
                invalid_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Feature mancanti: {missing_features}")
        
        if invalid_features:
            logger.warning(f"Feature non-numeriche rimosse: {invalid_features}")
            self.feature_columns = [f for f in self.feature_columns if f not in invalid_features]
        
        # Estrai features e target
        self.X_train = self.df_train[self.feature_columns].values.astype(np.float32)
        self.X_val = self.df_val[self.feature_columns].values.astype(np.float32)
        self.X_test = self.df_test[self.feature_columns].values.astype(np.float32)
        
        self.y_train = self.df_train[self.target_column].astype(np.int64)
        self.y_val = self.df_val[self.target_column].astype(np.int64)
        self.y_test = self.df_test[self.target_column].astype(np.int64)
        
        # Verifica range target
        all_targets = np.concatenate([self.y_train, self.y_val, self.y_test])
        min_target, max_target = all_targets.min(), all_targets.max()
        
        logger.info(f"Target range: {min_target} - {max_target} (atteso: 0 - {self.n_classes-1})")
        
        if min_target < 0 or max_target >= self.n_classes:
            raise ValueError(f"Target fuori range! Trovato: {min_target}-{max_target}, Atteso: 0-{self.n_classes-1}")
        
        # Verifica presenza di tutte le classi nel training
        unique_train_classes = set(self.y_train)
        expected_classes = set(range(self.n_classes))
        missing_classes = expected_classes - unique_train_classes
        
        if missing_classes:
            logger.error(f"🚨 Classi mancanti nel training set: {missing_classes}")
            for missing_class in missing_classes:
                class_name = self.multiclass_metadata['label_encoder_classes'][missing_class]
                logger.error(f"   Mancante: {class_name} (ID: {missing_class})")
            raise ValueError("Il training set deve contenere tutte le classi!")
        
        # Calcola class weights standard
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Class weights standard: {class_weights}")
        except Exception as e:
            logger.warning(f"Impossibile calcolare class weights: {e}")
            self.class_weights = None
        
        # Calcola frequenze per funzione di Loss
        self.class_freq = {}
        unique, counts = np.unique(self.y_train, return_counts=True)
        for class_id, count in zip(unique, counts):
            self.class_freq[class_id] = int(count)
        logger.info(f"Frequenze classi per Loss Function: {self.class_freq}")
        
        logger.info(f"Input dimension: {self.X_train.shape[1]}")
        
        return self.X_train.shape[1]
    
    def create_dataloaders(self):
        """Crea i DataLoader per PyTorch"""
        batch_size = self.hyperparams['batch_size']
        use_cuda = torch.cuda.is_available()
        
        # Crea TensorDataset
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.LongTensor(self.y_test)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=1 if use_cuda else 0,
            pin_memory=use_cuda
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=1 if use_cuda else 0,
            pin_memory=use_cuda
        )
        
        logger.info(f"DataLoaders creati:")
        logger.info(f"  Train: {len(train_loader)} batch di {batch_size}")
        logger.info(f"  Val: {len(val_loader)} batch di {batch_size}")
        logger.info(f"  Test: {len(test_loader)} batch di {batch_size}")
        
        return train_loader, val_loader, test_loader


def save_model_multiclass(model, hyperparams, feature_columns, filepath, 
                         n_classes, class_mapping, class_weights=None, class_freq=None):
    """Salva il modello multiclass e la configurazione"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'feature_columns': feature_columns,
        'input_dim': len(feature_columns),
        'n_classes': n_classes,
        'class_mapping': class_mapping,
        'class_weights': class_weights,
        'class_freq': class_freq,  # NUOVO: salva anche frequenze
        'model_type': 'multiclass'
    }, filepath)
    logger.info(f"Modello multiclass salvato in: {filepath}")


def load_model_multiclass(filepath, device='cpu'):
    """Carica il modello multiclass salvato"""
    checkpoint = torch.load(filepath, map_location=device)
    
    if checkpoint.get('model_type') != 'multiclass':
        logger.warning("⚠️  Questo non sembra essere un modello multiclass!")
    
    model = NetworkTrafficMLPMulticlass(
        checkpoint['input_dim'], 
        checkpoint['hyperparams'],
        checkpoint['n_classes'],
        checkpoint.get('class_weights', None)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint['hyperparams'], checkpoint['feature_columns'], checkpoint['class_mapping']