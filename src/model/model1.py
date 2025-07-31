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
    Multi-Layer Perceptron per classificazione binaria di traffico di rete
    """
    def __init__(self, input_dim, config, class_weights=None):
        super(NetworkTrafficMLP, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = config['embedding_dim']
        self.ff_dim = config['ff_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Lista per memorizzare i layer
        layers = []
        
        # Primo layer: input -> embedding_dim
        layers.append(nn.Linear(input_dim, self.embedding_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.embedding_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Layer intermedi: embedding_dim -> ff_dim -> embedding_dim
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.embedding_dim, self.ff_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.ff_dim))
            layers.append(nn.Dropout(self.dropout))
            
            layers.append(nn.Linear(self.ff_dim, self.embedding_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.embedding_dim))
            layers.append(nn.Dropout(self.dropout))
        
        # Layer finale: embedding_dim -> 1 (classificazione binaria)
        # Rimossa Sigmoid - useremo BCEWithLogitsLoss
        layers.append(nn.Linear(self.embedding_dim, 1))
        
        # Combina tutti i layer in un Sequential
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi
        self.apply(self._init_weights)
        
        # Salva class weights per riferimento
        self.class_weights = class_weights
    
    def _init_weights(self, module):
        """Inizializzazione dei pesi con He initialization per ReLU"""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class NetworkTrafficDataset:
    """Classe per gestire il caricamento e preprocessing dei dati"""
    
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
    
    def load_data(self, train_path, val_path, test_path):
        """Carica i dataset preprocessati"""
        logger.info("Caricamento dataset...")
        
        # Carica i CSV
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        
        logger.info(f"Training set: {self.df_train.shape}")
        logger.info(f"Validation set: {self.df_val.shape}")
        logger.info(f"Test set: {self.df_test.shape}")
        
        # Verifica che tutte le feature necessarie siano presenti
        missing_features = set(self.feature_columns) - set(self.df_train.columns)
        if missing_features:
            raise ValueError(f"Feature mancanti nel dataset: {missing_features}")
        
        # Estrai features e target
        self.X_train = self.df_train[self.feature_columns].values.astype(np.float32)
        self.X_val = self.df_val[self.feature_columns].values.astype(np.float32)
        self.X_test = self.df_test[self.feature_columns].values.astype(np.float32)
        
        """ PARTE INUTILE DA CANCELLARE"""
        # Converti target in binario (0: benigno, 1: attacco)
        #self.y_train = (self.df_train[self.target_column] != self.benign_label).astype(np.float32)
        #self.y_val = (self.df_val[self.target_column] != self.benign_label).astype(np.float32)
        #self.y_test = (self.df_test[self.target_column] != self.benign_label).astype(np.float32)
        self.y_train = self.df_train[self.target_column].astype(np.float32)
        self.y_val = self.df_val[self.target_column].astype(np.float32) 
        self.y_test = self.df_test[self.target_column].astype(np.float32)
        
        # Calcola statistiche di bilanciamento
        n_benign_train = (self.y_train == 0).sum()
        n_attack_train = (self.y_train == 1).sum()
        n_benign_val = (self.y_val == 0).sum()
        n_attack_val = (self.y_val == 1).sum()
        n_benign_test = (self.y_test == 0).sum()
        n_attack_test = (self.y_test == 1).sum()
        
        logger.info(f"Input dimension: {self.X_train.shape[1]}")
        logger.info(f"Training - Benigni: {n_benign_train} ({n_benign_train/len(self.y_train)*100:.1f}%), "
                   f"Attacchi: {n_attack_train} ({n_attack_train/len(self.y_train)*100:.1f}%)")
        logger.info(f"Validation - Benigni: {n_benign_val} ({n_benign_val/len(self.y_val)*100:.1f}%), "
                   f"Attacchi: {n_attack_val} ({n_attack_val/len(self.y_val)*100:.1f}%)")
        logger.info(f"Test - Benigni: {n_benign_test} ({n_benign_test/len(self.y_test)*100:.1f}%), "
                   f"Attacchi: {n_attack_test} ({n_attack_test/len(self.y_test)*100:.1f}%)")
        
        # Calcola class weights per gestire lo sbilanciamento
        if n_attack_train > 0 and n_benign_train > 0:
            # Peso per la classe positiva (attacchi)
            pos_weight = n_benign_train / n_attack_train
            logger.info(f"Calculated positive class weight: {pos_weight:.3f}")
            self.class_weights = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            logger.warning("Una delle classi ha 0 campioni - non usando class weights")
            self.class_weights = None
        
        return self.X_train.shape[1]  # Restituisce input_dim
    
    def create_dataloaders(self):
        """Crea i DataLoader per PyTorch"""
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
        
        # Crea DataLoader con ottimizzazioni per GPU
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        
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