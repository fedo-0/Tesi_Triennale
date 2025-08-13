import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from utilities.logging_config import setup_logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class NetworkTrafficMLP(nn.Module):
    """
    Multi-Layer Perceptron migliorato per classificazione binaria di traffico di rete
    """
    def __init__(self, input_dim, config, class_weights=None):
        super(NetworkTrafficMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = config['embedding_dim']  # Rinominato da embedding_dim
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Input validation
        if self.num_layers < 2:
            raise ValueError("num_layers deve essere almeno 2 (input + output)")
        
        # Lista per memorizzare i layer
        layers = []
        
        # Primo layer: input -> hidden_dim
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Layer intermedi: hidden_dim -> hidden_dim (architettura più semplice)
        for i in range(self.num_layers - 2):
            # Dimensione decrescente per layer più profondi
            next_hidden_dim = max(self.hidden_dim // (2 ** (i + 1)), 32)
            
            layers.append(nn.Linear(self.hidden_dim if i == 0 else prev_hidden_dim, next_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(next_hidden_dim))
            layers.append(nn.Dropout(self.dropout))
            
            prev_hidden_dim = next_hidden_dim
        
        # Layer finale: ultimo_hidden -> 1 (classificazione binaria)
        final_input_dim = prev_hidden_dim if self.num_layers > 2 else self.hidden_dim
        layers.append(nn.Linear(final_input_dim, 1))
        
        # Combina tutti i layer in un Sequential
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi migliorata
        self.apply(self._init_weights)
        
        # Salva class weights per riferimento
        self.class_weights = class_weights
        
        # Conta parametri
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Modello creato con {total_params:,} parametri")
    
    def _init_weights(self, module):
        """Inizializzazione dei pesi migliorata"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization per layer intermedi
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class NetworkTrafficDataset:
    """Classe per gestire il caricamento e preprocessing dei dati - VERSIONE MIGLIORATA"""
    
    def __init__(self, config_path="config/dataset.json", hyperparams_path="config/hyperparameters.json", model_size="small"):
        self.load_configs(config_path, hyperparams_path, model_size)
        
    def load_configs(self, config_path, hyperparams_path, model_size="small"):
        """Carica le configurazioni"""
        with open(config_path, 'r') as f:
            self.dataset_config = json.load(f)['dataset']
        
        with open(hyperparams_path, 'r') as f:
            hyperparams_data = json.load(f)
            self.hyperparams = hyperparams_data[model_size]
            
        self.feature_columns = (self.dataset_config['numeric_columns'] + 
                               self.dataset_config['categorical_columns'])
        self.target_column = self.dataset_config['target_column']
        self.benign_label = self.dataset_config['benign_label']
        
        logger.info(f"Feature columns loaded: {len(self.feature_columns)}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Benign label: {self.benign_label}")
    
    def analyze_data_distribution(self, df, set_name):
        """Analizza la distribuzione dei dati per debug"""
        target_values = df[self.target_column].value_counts()
        total = len(df)
        
        logger.info(f"\n{set_name} Set Distribution:")
        for value, count in target_values.items():
            percentage = (count / total) * 100
            logger.info(f"  {value}: {count:,} ({percentage:.2f}%)")
        
        # Controlla se ci sono valori NaN o infiniti
        if df[self.feature_columns].isnull().any().any():
            logger.warning(f"⚠️  {set_name} set contiene valori NaN!")
        
        if np.isinf(df[self.feature_columns].values).any():
            logger.warning(f"⚠️  {set_name} set contiene valori infiniti!")
        
        return target_values
    
    def load_data(self, train_path, val_path, test_path):
        """Carica i dataset preprocessati con analisi migliorata"""
        logger.info("Caricamento dataset...")
        
        # Carica i CSV
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        
        logger.info(f"Training set: {self.df_train.shape}")
        logger.info(f"Validation set: {self.df_val.shape}")
        logger.info(f"Test set: {self.df_test.shape}")
        
        # Analizza distribuzione
        train_dist = self.analyze_data_distribution(self.df_train, "Training")
        val_dist = self.analyze_data_distribution(self.df_val, "Validation")
        test_dist = self.analyze_data_distribution(self.df_test, "Test")
        
        # Verifica che tutte le feature necessarie siano presenti
        missing_features = set(self.feature_columns) - set(self.df_train.columns)
        if missing_features:
            raise ValueError(f"Feature mancanti nel dataset: {missing_features}")
        
        # Estrai features e target
        self.X_train = self.df_train[self.feature_columns].values.astype(np.float32)
        self.X_val = self.df_val[self.feature_columns].values.astype(np.float32)
        self.X_test = self.df_test[self.feature_columns].values.astype(np.float32)
        
        # Converti target in binario se necessario
        self.y_train = self.df_train[self.target_column].astype(np.float32)
        self.y_val = self.df_val[self.target_column].astype(np.float32) 
        self.y_test = self.df_test[self.target_column].astype(np.float32)
        
        # Verifica che i target siano binari
        unique_train = np.unique(self.y_train)
        unique_val = np.unique(self.y_val)
        unique_test = np.unique(self.y_test)
        
        logger.info(f"Target values - Train: {unique_train}, Val: {unique_val}, Test: {unique_test}")
        
        if not all(set(unique) <= {0.0, 1.0} for unique in [unique_train, unique_val, unique_test]):
            logger.warning("⚠️  I target non sembrano essere binari (0/1)!")
        
        # Calcola statistiche di bilanciamento
        n_benign_train = (self.y_train == 0).sum()
        n_attack_train = (self.y_train == 1).sum()
        n_benign_val = (self.y_val == 0).sum()
        n_attack_val = (self.y_val == 1).sum()
        n_benign_test = (self.y_test == 0).sum()
        n_attack_test = (self.y_test == 1).sum()
        
        logger.info(f"\nBinary Distribution Analysis:")
        logger.info(f"Training - Benigni: {n_benign_train:,} ({n_benign_train/len(self.y_train)*100:.1f}%), "
                   f"Attacchi: {n_attack_train:,} ({n_attack_train/len(self.y_train)*100:.1f}%)")
        logger.info(f"Validation - Benigni: {n_benign_val:,} ({n_benign_val/len(self.y_val)*100:.1f}%), "
                   f"Attacchi: {n_attack_val:,} ({n_attack_val/len(self.y_val)*100:.1f}%)")
        logger.info(f"Test - Benigni: {n_benign_test:,} ({n_benign_test/len(self.y_test)*100:.1f}%), "
                   f"Attacchi: {n_attack_test:,} ({n_attack_test/len(self.y_test)*100:.1f}%)")
        
        # CONTROLLO CRITICO: Verifica se il validation set ha attacchi
        if n_attack_val == 0:
            logger.error("🚨 PROBLEMA CRITICO: Il validation set non contiene NESSUN attacco!")
            logger.error("Questo spiega perché Precision/Recall/F1 sono 0.0000")
            logger.error("Il modello non può imparare a riconoscere gli attacchi se non li vede mai in validazione!")
        
        if n_attack_test == 0:
            logger.warning("⚠️  Il test set non contiene attacchi!")
        
        # Calcola class weights per gestire lo sbilanciamento
        if n_attack_train > 0 and n_benign_train > 0:
            # Peso per la classe positiva (attacchi)
            pos_weight = n_benign_train / n_attack_train
            logger.info(f"Calculated positive class weight: {pos_weight:.3f}")
            self.class_weights = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            logger.warning("Una delle classi ha 0 campioni - non usando class weights")
            self.class_weights = None
        
        logger.info(f"Input dimension: {self.X_train.shape[1]}")
        return self.X_train.shape[1]  # Restituisce input_dim
    
    def create_dataloaders(self):
        """Crea i DataLoader per PyTorch - VERSIONE CORRETTA"""
        batch_size = self.hyperparams['batch_size']
        use_cuda = torch.cuda.is_available()
        
        # Crea TensorDataset
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.FloatTensor(self.y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.FloatTensor(self.y_test)
        )
        
        # Crea DataLoader - CORREZIONE CRITICA: shuffle=True per training
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # ✅ CORRETTO: True per training
            num_workers=4 if use_cuda else 2,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # ✅ CORRETTO: False per validation
            num_workers=2 if use_cuda else 1,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # ✅ CORRETTO: False per test
            num_workers=2 if use_cuda else 1,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        
        logger.info(f"DataLoaders creati:")
        logger.info(f"  Train: {len(train_loader)} batch di {batch_size} (shuffle=True)")
        logger.info(f"  Validation: {len(val_loader)} batch di {batch_size} (shuffle=False)")
        logger.info(f"  Test: {len(test_loader)} batch di {batch_size} (shuffle=False)")
        
        return train_loader, val_loader, test_loader


def save_model(model, hyperparams, feature_columns, filepath, class_weights=None):
    """Salva il modello e la configurazione"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'feature_columns': feature_columns,
        'input_dim': len(feature_columns),
        'class_weights': class_weights
    }, filepath)
    logger.info(f"Modello salvato in: {filepath}")


def load_model(filepath, device='cpu'):
    """Carica il modello salvato"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Ricrea il modello
    model = NetworkTrafficMLP(
        checkpoint['input_dim'], 
        checkpoint['hyperparams'],
        checkpoint.get('class_weights', None)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint['hyperparams'], checkpoint['feature_columns']


# Classe aggiuntiva per debugging
class ModelDebugger:
    """Utility per debug del modello durante training"""
    
    @staticmethod
    def analyze_batch_predictions(model, batch_x, batch_y, device):
        """Analizza le predizioni di un batch per debug"""
        model.eval()
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Statistiche
            actual_attacks = (batch_y == 1).sum().item()
            predicted_attacks = (predictions == 1).sum().item()
            correct_attacks = ((predictions == 1) & (batch_y == 1)).sum().item()
            
            logger.info(f"Batch Debug:")
            logger.info(f"  Actual attacks: {actual_attacks}/{len(batch_y)}")
            logger.info(f"  Predicted attacks: {predicted_attacks}/{len(batch_y)}")
            logger.info(f"  Correct attacks: {correct_attacks}/{actual_attacks if actual_attacks > 0 else 1}")
            logger.info(f"  Min/Max/Mean probs: {probabilities.min():.4f}/{probabilities.max():.4f}/{probabilities.mean():.4f}")
        
        model.train()
        return {
            'actual_attacks': actual_attacks,
            'predicted_attacks': predicted_attacks,
            'correct_attacks': correct_attacks,
            'probabilities': probabilities.cpu().numpy()
        }