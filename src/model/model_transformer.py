import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from utilities.logging_config import setup_logging
from torch.utils.data import DataLoader, TensorDataset
import math

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding per Transformer
    """
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NetworkTrafficTransformer(nn.Module):
    """
    Transformer per classificazione multiclass di sequenze di pacchetti di rete
    """
    def __init__(self, config, n_classes, n_features):
        super(NetworkTrafficTransformer, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.max_seq_length = config['max_seq_length']
        
        # Input projection: features → d_model
        self.input_projection = nn.Linear(n_features, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, n_classes)
        )
        
        # Inizializzazione pesi
        self.apply(self._init_weights)
        
        # Log info modello
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Transformer MULTICLASS creato:")
        logger.info(f"  Parametri totali: {total_params:,}")
        logger.info(f"  Input features: {n_features}")
        logger.info(f"  Classi output: {n_classes}")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  Attention heads: {self.nhead}")
        logger.info(f"  Encoder layers: {self.num_layers}")
        logger.info(f"  Max sequence length: {self.max_seq_length}")
    
    def _init_weights(self, module):
        """Inizializzazione pesi"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass
        Args:
            x: (batch_size, seq_length, n_features)
            attention_mask: (batch_size, seq_length) - 1 per token reali, 0 per padding
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model) per pos_encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        
        # Crea attention mask per Transformer (inverte: 0 per token reali, -inf per padding)
        if attention_mask is not None:
            # PyTorch Transformer usa True per mascherare (ignorare)
            src_key_padding_mask = (attention_mask == 0)  # True dove c'è padding
        else:
            src_key_padding_mask = None
        
        # Transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global pooling: media pesata sui token non-padding
        if attention_mask is not None:
            # Weighted average escludendo padding
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            masked_encoded = encoded * mask_expanded
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            pooled = masked_encoded.sum(dim=1) / seq_lengths
        else:
            # Simple average pooling
            pooled = encoded.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

class NetworkTrafficDatasetTransformer:
    """
    Dataset manager per Transformer
    """
    def __init__(self, config_path="config/dataset.json", 
                 hyperparams_path="config/hyperparameters_transformer.json",
                 model_size="small", 
                 metadata_path="resources/datasets/transformer_metadata.json"):
        self.metadata_path = metadata_path
        self.load_configs(config_path, hyperparams_path, model_size)
    
    def load_configs(self, config_path, hyperparams_path, model_size):
        """Carica configurazioni"""
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams_data = json.load(f)
                self.hyperparams = hyperparams_data[model_size]
        except:
            logger.warning(f"Impossibile caricare {hyperparams_path}, usando config default")
            self.hyperparams = {
                'batch_size': 32,           # Più piccolo per sequenze
                'lr': 0.0001,              # Learning rate più basso
                'weight_decay': 1e-4,
                'd_model': 128,            # Dimensione embedding
                'nhead': 8,                # Attention heads
                'num_layers': 4,           # Transformer layers
                'dropout': 0.1,
                'max_seq_length': 50       # Lunghezza massima sequenza
            }
        
        # Carica metadati
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.n_classes = self.metadata['n_classes']
        self.n_features = self.metadata['n_features']
        self.class_mapping = self.metadata['class_mapping']
        self.max_seq_length = self.metadata['max_seq_length']
        
        logger.info(f"Configurazione Transformer caricata:")
        logger.info(f"  Classi: {self.n_classes}")
        logger.info(f"  Features per pacchetto: {self.n_features}")
        logger.info(f"  Max sequence length: {self.max_seq_length}")
        logger.info(f"  Hyperparams: {self.hyperparams}")
    
    def load_data(self, data_dir="resources/datasets"):
        """Carica dati preprocessati per Transformer"""
        logger.info("Caricamento dataset transformer...")
        
        # Carica sequenze
        self.train_sequences = np.load(f"{data_dir}/train_sequences.npy")
        self.train_labels = np.load(f"{data_dir}/train_labels.npy")
        self.train_masks = np.load(f"{data_dir}/train_masks.npy")
        
        self.val_sequences = np.load(f"{data_dir}/val_sequences.npy")
        self.val_labels = np.load(f"{data_dir}/val_labels.npy")
        self.val_masks = np.load(f"{data_dir}/val_masks.npy")
        
        self.test_sequences = np.load(f"{data_dir}/test_sequences.npy")
        self.test_labels = np.load(f"{data_dir}/test_labels.npy")
        self.test_masks = np.load(f"{data_dir}/test_masks.npy")
        
        logger.info(f"Dataset caricati:")
        logger.info(f"  Train: {self.train_sequences.shape}")
        logger.info(f"  Val: {self.val_sequences.shape}")
        logger.info(f"  Test: {self.test_sequences.shape}")
        
        # Verifica dimensioni
        expected_shape = (None, self.max_seq_length, self.n_features)
        actual_shape = self.train_sequences.shape[1:]
        logger.info(f"  Shape attesa: {expected_shape}")
        logger.info(f"  Shape effettiva: {actual_shape}")
        
        return self.n_features
    
    def create_dataloaders(self):
        """Crea DataLoader per sequenze"""
        batch_size = self.hyperparams['batch_size']
        
        # Crea TensorDataset
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_sequences),
            torch.LongTensor(self.train_labels),
            torch.FloatTensor(self.train_masks)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_sequences),
            torch.LongTensor(self.val_labels),
            torch.FloatTensor(self.val_masks)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(self.test_sequences),
            torch.LongTensor(self.test_labels),
            torch.FloatTensor(self.test_masks)
        )
        
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"DataLoaders creati:")
        logger.info(f"  Train: {len(train_loader)} batch di {batch_size}")
        logger.info(f"  Val: {len(val_loader)} batch di {batch_size}")
        logger.info(f"  Test: {len(test_loader)} batch di {batch_size}")
        
        return train_loader, val_loader, test_loader