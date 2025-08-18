import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from utilities.logging_config import setup_logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss per gestire sbilanciamento estremo"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class NetworkTrafficMLPMulticlass(nn.Module):
    """
    Multi-Layer Perceptron per classificazione MULTICLASS di traffico di rete
    BASATO sul modello binario di successo - CAMBIATE SOLO LE PARTI ESSENZIALI
    """
    def __init__(self, input_dim, config, n_classes, class_weights=None):
        super(NetworkTrafficMLPMulticlass, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes  # NUOVO: Numero di classi
        self.hidden_dim = config['embedding_dim']  # Identico al binario
        self.num_layers = config['num_layers']  # Identico al binario
        self.dropout = config['dropout']        # Identico al binario
        
        # Input validation
        if self.num_layers < 2:
            raise ValueError("num_layers deve essere almeno 2 (input + output)")
        if self.n_classes < 2:
            raise ValueError("n_classes deve essere almeno 2")
        
        # Lista per memorizzare i layer - IDENTICA AL BINARIO
        layers = []
        
        # Primo layer: input -> hidden_dim - IDENTICO AL BINARIO
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        # Layer intermedi: hidden_dim -> hidden_dim - IDENTICI AL BINARIO
        for i in range(self.num_layers - 2):
            # Dimensione decrescente per layer più profondi
            next_hidden_dim = max(self.hidden_dim // (2 ** (i + 1)), 32)
            
            layers.append(nn.Linear(self.hidden_dim if i == 0 else prev_hidden_dim, next_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(next_hidden_dim))
            layers.append(nn.Dropout(self.dropout))
            
            prev_hidden_dim = next_hidden_dim
        
        # Layer finale: ultimo_hidden -> n_classes - UNICA MODIFICA CRITICA
        final_input_dim = prev_hidden_dim if self.num_layers > 2 else self.hidden_dim
        layers.append(nn.Linear(final_input_dim, self.n_classes))  # CAMBIATO: 1 -> n_classes
        # NOTA: Niente Sigmoid/Softmax qui - useremo CrossEntropyLoss che include Softmax
        
        # Combina tutti i layer in un Sequential - IDENTICO AL BINARIO
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi - IDENTICA AL BINARIO
        self.apply(self._init_weights)
        
        # Salva class weights per riferimento - ADATTATO PER MULTICLASS
        self.class_weights = class_weights
        
        # Conta parametri
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Modello MULTICLASS creato con {total_params:,} parametri per {n_classes} classi")
    
    def _init_weights(self, module):
        """Inizializzazione dei pesi - IDENTICA AL BINARIO"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass - IDENTICO AL BINARIO"""
        return self.network(x)


class NetworkTrafficDatasetMulticlass:
    """Classe per gestire il caricamento dati MULTICLASS - ADATTATA DA BINARIO"""
    
    def __init__(self, config_path="config/dataset.json", hyperparams_path="config/hyperparameters.json", 
                 model_size="small", metadata_path="resources/datasets/multiclass_metadata.json"):
        self.metadata_path = metadata_path
        self.load_configs(config_path, hyperparams_path, model_size)
        
    def load_configs(self, config_path, hyperparams_path, model_size="small"):
        """Carica le configurazioni - IDENTICA AL BINARIO + METADATI MULTICLASS"""
        with open(config_path, 'r') as f:
            self.dataset_config = json.load(f)['dataset']
        
        with open(hyperparams_path, 'r') as f:
            hyperparams_data = json.load(f)
            self.hyperparams = hyperparams_data[model_size]
        
        # NUOVO: Carica metadati multiclass
        with open(self.metadata_path, 'r') as f:
            self.multiclass_metadata = json.load(f)
            
        self.feature_columns = self.multiclass_metadata['feature_columns']  # Da metadati multiclass
        self.target_column = 'multiclass_target'  # Colonna target multiclass
        self.n_classes = self.multiclass_metadata['n_classes']  # Numero classi
        self.class_mapping = self.multiclass_metadata['class_mapping']  # Mapping classi
        
        logger.info(f"Feature columns loaded: {len(self.feature_columns)}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Class mapping: {self.class_mapping}")
    
    def analyze_data_distribution_multiclass(self, df, set_name):
        """Analizza la distribuzione multiclass - NUOVO"""
        target_values = df[self.target_column].value_counts().sort_index()
        total = len(df)
        
        logger.info(f"\n{set_name} Set Distribution (Multiclass):")
        for class_idx, count in target_values.items():
            class_name = self.multiclass_metadata['label_encoder_classes'][class_idx]
            percentage = (count / total) * 100
            logger.info(f"  {class_name} ({class_idx}): {count:,} ({percentage:.2f}%)")
        
        # Controlla se ci sono valori NaN o fuori range - IDENTICO AL BINARIO
        if df[self.feature_columns].isnull().any().any():
            logger.warning(f"⚠️  {set_name} set contiene valori NaN!")
        
        if np.isinf(df[self.feature_columns].values).any():
            logger.warning(f"⚠️  {set_name} set contiene valori infiniti!")
        
        return target_values
    
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


class NetworkTrafficMLPMulticlass(nn.Module):
    """
    Multi-Layer Perceptron per classificazione MULTICLASS di traffico di rete
    BASATO sul modello binario di successo - CAMBIATE SOLO LE PARTI ESSENZIALI
    """
    def __init__(self, input_dim, config, n_classes, class_weights=None):
        super(NetworkTrafficMLPMulticlass, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes  # NUOVO: Numero di classi
        self.hidden_dim = config['embedding_dim']  # Identico al binario
        self.num_layers = config['num_layers']  # Identico al binario
        self.dropout = config['dropout']        # Identico al binario
        
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
        
        # Layer intermedi: hidden_dim -> hidden_dim
        for i in range(self.num_layers - 2):
            # Dimensione decrescente per layer più profondi
            next_hidden_dim = max(self.hidden_dim // (2 ** (i + 1)), 32)
            
            layers.append(nn.Linear(self.hidden_dim if i == 0 else prev_hidden_dim, next_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(next_hidden_dim))
            layers.append(nn.Dropout(self.dropout))
            
            prev_hidden_dim = next_hidden_dim
        
        # Layer finale: ultimo_hidden -> n_classes - UNICA MODIFICA CRITICA
        final_input_dim = prev_hidden_dim if self.num_layers > 2 else self.hidden_dim
        layers.append(nn.Linear(final_input_dim, self.n_classes))  # CAMBIATO: 1 -> n_classes
        # NOTA: Niente Sigmoid/Softmax qui - useremo CrossEntropyLoss che include Softmax
        
        # Combina tutti i layer in un Sequential - IDENTICO AL BINARIO
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi - IDENTICA AL BINARIO
        self.apply(self._init_weights)
        
        # Salva class weights per riferimento - ADATTATO PER MULTICLASS
        self.class_weights = class_weights
        
        # Conta parametri
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Modello MULTICLASS creato con {total_params:,} parametri per {n_classes} classi")
    
    def _init_weights(self, module):
        """Inizializzazione dei pesi - IDENTICA AL BINARIO"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass - IDENTICO AL BINARIO"""
        return self.network(x)


class NetworkTrafficDatasetMulticlass:
    """Classe per gestire il caricamento dati MULTICLASS - ADATTATA DA BINARIO"""
    
    def __init__(self, config_path="config/dataset.json", hyperparams_path="config/hyperparameters.json", 
                 model_size="small", metadata_path="resources/datasets/multiclass_metadata.json"):
        self.metadata_path = metadata_path
        self.load_configs(config_path, hyperparams_path, model_size)
        
    def load_configs(self, config_path, hyperparams_path, model_size="small"):
        """Carica le configurazioni - IDENTICA AL BINARIO + METADATI MULTICLASS"""
        with open(config_path, 'r') as f:
            self.dataset_config = json.load(f)['dataset']
        
        with open(hyperparams_path, 'r') as f:
            hyperparams_data = json.load(f)
            self.hyperparams = hyperparams_data[model_size]
        
        # NUOVO: Carica metadati multiclass
        with open(self.metadata_path, 'r') as f:
            self.multiclass_metadata = json.load(f)
            
        self.feature_columns = self.multiclass_metadata['feature_columns']  # Da metadati multiclass
        self.target_column = 'multiclass_target'  # NUOVO: Colonna target multiclass
        self.n_classes = self.multiclass_metadata['n_classes']  # NUOVO: Numero classi
        self.class_mapping = self.multiclass_metadata['class_mapping']  # NUOVO: Mapping classi
        
        logger.info(f"Feature columns loaded: {len(self.feature_columns)}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Class mapping: {self.class_mapping}")
    
    def analyze_data_distribution_multiclass(self, df, set_name):
        """Analizza la distribuzione multiclass - NUOVO"""
        target_values = df[self.target_column].value_counts().sort_index()
        total = len(df)
        
        logger.info(f"\n{set_name} Set Distribution (Multiclass):")
        for class_idx, count in target_values.items():
            class_name = self.multiclass_metadata['label_encoder_classes'][class_idx]
            percentage = (count / total) * 100
            logger.info(f"  {class_name} ({class_idx}): {count:,} ({percentage:.2f}%)")
        
        # CORREZIONE: Controlla solo colonne numeriche per NaN e infiniti
        numeric_features = []
        for col in self.feature_columns:
            if col in df.columns:
                # Verifica se la colonna è numerica
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_features.append(col)
                else:
                    logger.warning(f"⚠️  Colonna {col} non è numerica (tipo: {df[col].dtype})")
        
        if len(numeric_features) > 0:
            # Controlla NaN solo su colonne numeriche
            if df[numeric_features].isnull().any().any():
                logger.warning(f"⚠️  {set_name} set contiene valori NaN in colonne numeriche!")
            
            # Controlla infiniti solo su colonne numeriche
            try:
                if np.isinf(df[numeric_features].values).any():
                    logger.warning(f"⚠️  {set_name} set contiene valori infiniti!")
            except TypeError as e:
                logger.warning(f"⚠️  Impossibile verificare valori infiniti: {e}")
        else:
            logger.warning(f"⚠️  Nessuna colonna numerica trovata in {set_name} set!")
        
        return target_values
    
    def load_data(self, train_path, val_path, test_path):
        """Carica i dataset preprocessati multiclass - ADATTATA DA BINARIO"""
        logger.info("Caricamento dataset multiclass...")
        
        # Carica i CSV - IDENTICO AL BINARIO
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        
        logger.info(f"Training set: {self.df_train.shape}")
        logger.info(f"Validation set: {self.df_val.shape}")
        logger.info(f"Test set: {self.df_test.shape}")
        
        # 🔍 DEBUG AGGIUNTO: Analizza le colonne prima di procedere
        logger.info(f"\n=== DEBUG: ANALISI COLONNE ===")
        logger.info(f"Colonne totali nel training set: {len(self.df_train.columns)}")
        logger.info(f"Feature columns da metadati: {len(self.feature_columns)}")
        
        logger.info(f"\nTutte le colonne nel dataset:")
        for i, col in enumerate(self.df_train.columns):
            dtype = self.df_train[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(self.df_train[col])
            unique_count = self.df_train[col].nunique()
            logger.info(f"  {i:2d}. {col:25s} | {str(dtype):15s} | numeric: {is_numeric} | unique: {unique_count}")
        
        logger.info(f"\nFeature columns dai metadati:")
        problematic_cols = []
        for i, col in enumerate(self.feature_columns):
            if col in self.df_train.columns:
                dtype = self.df_train[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(self.df_train[col])
                if not is_numeric:
                    problematic_cols.append(col)
                    logger.info(f"  ❌ {i:2d}. {col:25s} | {str(dtype):15s} | PROBLEMATICA!")
                else:
                    logger.info(f"  ✅ {i:2d}. {col:25s} | {str(dtype):15s} | OK")
            else:
                logger.info(f"  ⚠️  {i:2d}. {col:25s} | MANCANTE NEL DATASET!")
        
        if problematic_cols:
            logger.error(f"🚨 TROVATE {len(problematic_cols)} COLONNE PROBLEMATICHE: {problematic_cols}")
            logger.error("Queste colonne causano l'errore np.isinf()!")
            
            # Proponi correzione automatica
            logger.info("🔧 CORREZIONE AUTOMATICA: Rimuovo colonne problematiche...")
            self.feature_columns = [col for col in self.feature_columns 
                                  if col not in problematic_cols and col in self.df_train.columns]
            logger.info(f"Feature columns corrette: {len(self.feature_columns)}")
        
        logger.info(f"=== FINE DEBUG ===\n")
        
        # Analizza distribuzione multiclass - NUOVO
        train_dist = self.analyze_data_distribution_multiclass(self.df_train, "Training")
        val_dist = self.analyze_data_distribution_multiclass(self.df_val, "Validation")
        test_dist = self.analyze_data_distribution_multiclass(self.df_test, "Test")
        
        # Verifica che tutte le feature necessarie siano presenti - IDENTICO AL BINARIO
        missing_features = set(self.feature_columns) - set(self.df_train.columns)
        if missing_features:
            raise ValueError(f"Feature mancanti nel dataset: {missing_features}")
        
        # Estrai features e target - IDENTICO AL BINARIO TRANNE TARGET
        self.X_train = self.df_train[self.feature_columns].values.astype(np.float32)
        self.X_val = self.df_val[self.feature_columns].values.astype(np.float32)
        self.X_test = self.df_test[self.feature_columns].values.astype(np.float32)
        
        # Target multiclass invece di binario - NUOVO
        self.y_train = self.df_train[self.target_column].astype(np.int64)  # int64 per CrossEntropyLoss
        self.y_val = self.df_val[self.target_column].astype(np.int64)
        self.y_test = self.df_test[self.target_column].astype(np.int64)
        
        # Verifica che i target siano nel range corretto - NUOVO
        all_targets = np.concatenate([self.y_train, self.y_val, self.y_test])
        min_target, max_target = all_targets.min(), all_targets.max()
        
        logger.info(f"Target range: {min_target} - {max_target} (expected: 0 - {self.n_classes-1})")
        
        if min_target < 0 or max_target >= self.n_classes:
            raise ValueError(f"Target values out of range! Found: {min_target}-{max_target}, Expected: 0-{self.n_classes-1}")
        
        # Calcola statistiche di bilanciamento per ogni classe - ADATTATO PER MULTICLASS
        logger.info(f"\nDetailed Class Distribution:")
        for set_name, y_data in [("Training", self.y_train), ("Validation", self.y_val), ("Test", self.y_test)]:
            unique, counts = np.unique(y_data, return_counts=True)
            logger.info(f"{set_name} set:")
            for class_idx, count in zip(unique, counts):
                class_name = self.multiclass_metadata['label_encoder_classes'][class_idx]
                percentage = (count / len(y_data)) * 100
                logger.info(f"  {class_name}: {count:,} ({percentage:.2f}%)")
        
        # CONTROLLO CRITICO: Verifica se tutte le classi sono presenti - NUOVO
        unique_train_classes = set(self.y_train)
        expected_classes = set(range(self.n_classes))
        missing_classes = expected_classes - unique_train_classes
        
        if missing_classes:
            logger.error(f"🚨 PROBLEMA CRITICO: Classi mancanti nel training set: {missing_classes}")
            for missing_class in missing_classes:
                class_name = self.multiclass_metadata['label_encoder_classes'][missing_class]
                logger.error(f"   Classe mancante: {class_name} (ID: {missing_class})")
            raise ValueError("Il training set deve contenere tutte le classi!")
        
        # Calcola class weights per gestire sbilanciamento multiclass - NUOVO
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Class weights calcolati: {class_weights}")
        except Exception as e:
            logger.warning(f"Impossibile calcolare class weights: {e}")
            self.class_weights = None
        
        logger.info(f"Input dimension: {self.X_train.shape[1]}")
        return self.X_train.shape[1]  # Restituisce input_dim
    
    def create_dataloaders(self):
        """Crea i DataLoader per PyTorch - IDENTICO AL BINARIO"""
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
        
        # Crea DataLoader
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if use_cuda else 2,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        """

        if self.class_weights is not None:
            # Calcola pesi per ogni sample (più aggressivo)
            sample_weights = []
            class_weights_dict = {}
            
            # Crea dizionario pesi per classe
            for i, weight in enumerate(self.class_weights):
                class_weights_dict[i] = weight.item() * 2.0  # Amplifica x2
            
            # Assegna peso a ogni sample
            for target in self.y_train:
                sample_weights.append(class_weights_dict[target])
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.y_train),
                replacement=True
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=sampler,  # USA SAMPLER invece di shuffle
                num_workers=4 if use_cuda else 2,
                pin_memory=use_cuda,
                persistent_workers=True if torch.get_num_threads() > 1 else False
            )
            logger.info("🎯 Usando WeightedRandomSampler AGGRESSIVO per bilanciamento")
        else:
            # Fallback normale
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4 if use_cuda else 2,
                pin_memory=use_cuda,
                persistent_workers=True if torch.get_num_threads() > 1 else False
            )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2 if use_cuda else 1,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2 if use_cuda else 1,
            pin_memory=use_cuda,
            persistent_workers=True if torch.get_num_threads() > 1 else False
        )
        
        logger.info(f"DataLoaders multiclass creati:")
        logger.info(f"  Train: {len(train_loader)} batch di {batch_size} (shuffle=True)")
        logger.info(f"  Validation: {len(val_loader)} batch di {batch_size} (shuffle=False)")
        logger.info(f"  Test: {len(test_loader)} batch di {batch_size} (shuffle=False)")
        
        return train_loader, val_loader, test_loader


def save_model_multiclass(model, hyperparams, feature_columns, filepath, 
                         n_classes, class_mapping, class_weights=None):
    """Salva il modello multiclass e la configurazione - ADATTATO DA BINARIO"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'feature_columns': feature_columns,
        'input_dim': len(feature_columns),
        'n_classes': n_classes,
        'class_mapping': class_mapping,
        'class_weights': class_weights,
        'model_type': 'multiclass'
    }, filepath)
    logger.info(f"Modello multiclass salvato in: {filepath}")


def load_model_multiclass(filepath, device='cpu'):
    """Carica il modello multiclass salvato - ADATTATO DA BINARIO"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Verifica che sia un modello multiclass
    if checkpoint.get('model_type') != 'multiclass':
        logger.warning("⚠️  Questo non sembra essere un modello multiclass!")
    
    # Ricrea il modello
    model = NetworkTrafficMLPMulticlass(
        checkpoint['input_dim'], 
        checkpoint['hyperparams'],
        checkpoint['n_classes'],                    # NUOVO
        checkpoint.get('class_weights', None)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint['hyperparams'], checkpoint['feature_columns'], checkpoint['class_mapping']


# Classe per debugging multiclass
class ModelDebuggerMulticlass:
    """Utility per debug del modello multiclass durante training"""
    
    @staticmethod
    def analyze_batch_predictions_multiclass(model, batch_x, batch_y, device, class_names):
        """Analizza le predizioni di un batch per debug multiclass"""
        model.eval()
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)  # CAMBIATO: softmax invece di sigmoid
            predictions = torch.argmax(probabilities, dim=1)  # CAMBIATO: argmax invece di threshold
            
            # Statistiche per classe
            n_classes = len(class_names)
            for class_idx in range(n_classes):
                actual_count = (batch_y == class_idx).sum().item()
                predicted_count = (predictions == class_idx).sum().item()
                correct_count = ((predictions == class_idx) & (batch_y == class_idx)).sum().item()
                
                if actual_count > 0 or predicted_count > 0:
                    logger.info(f"Classe {class_names[class_idx]} ({class_idx}):")
                    logger.info(f"  Actual: {actual_count}, Predicted: {predicted_count}, Correct: {correct_count}")
            
            # Statistiche generali
            total_correct = (predictions == batch_y).sum().item()
            total_samples = len(batch_y)
            accuracy = total_correct / total_samples
            
            logger.info(f"Batch Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
        
        model.train()
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'accuracy': accuracy
        }