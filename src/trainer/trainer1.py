import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from model.model1 import NetworkTrafficMLP, NetworkTrafficDataset, save_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Classe per gestire l'addestramento del modello"""
    
    def __init__(self, model, hyperparams, device, class_weights=None):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device
        
        # Optimizer e loss function
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        
        # Loss function con class weights per gestire sbilanciamento
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            logger.info(f"Usando BCEWithLogitsLoss con pos_weight: {class_weights.item():.3f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Usando BCEWithLogitsLoss senza class weights")
        
        # Scheduler per learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Tracking delle metriche
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def add_noise_to_input(self, x, noise_factor):
        """Aggiunge rumore gaussiano all'input per regolarizzazione"""
        if self.model.training and noise_factor > 0:
            noise = torch.randn_like(x) * noise_factor
            return x + noise
        return x
    
    def train_epoch(self, train_loader, epoch):
        """Addestra per una epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Aggiungi rumore per regolarizzazione
            if 'noise_factor' in self.hyperparams:
                batch_x = self.add_noise_to_input(batch_x, self.hyperparams['noise_factor'])
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistiche
            total_loss += loss.item()
            # Per BCEWithLogitsLoss, applichiamo sigmoid per le predizioni
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Aggiorna progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log periodico per batch grandi
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Valuta il modello sul validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        """Addestra il modello con early stopping"""
        logger.info(f"Inizio addestramento per {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parametri del modello: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Salva learning rate corrente
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)
            logger.info(f"Terminata l'epoca: {epoch+1}")
            
            # Validation
            val_loss, val_acc, val_pred, val_true, val_prob = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Salva metriche
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Calcola metriche aggiuntive per validation
            precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_pred, average='binary', zero_division=0)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info("✓ Nuovo miglior modello salvato!")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping dopo {epoch+1} epochs (patience: {patience})")
                    break
        
        # Carica il miglior modello
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Caricato il miglior modello per la valutazione finale.")
        
        return self.model
    
    def plot_training_history(self, save_path=None):
        """Plotta l'andamento dell'addestramento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        ax1.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        ax1.plot(epochs, self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning Rate
        ax3.plot(epochs, self.learning_rates, label='Learning Rate', color='orange')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Loss diff (overfitting check)
        loss_diff = [val - train for val, train in zip(self.val_losses, self.train_losses)]
        ax4.plot(epochs, loss_diff, label='Val Loss - Train Loss', color='purple')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Check')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot salvato in: {save_path}")
        plt.show()


def evaluate_model(model, test_loader, device):
    """Valuta il modello sul test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Valutazione sul test set...")
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calcola metriche
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary', zero_division=0)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"RISULTATI TEST SET")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    print(classification_report(targets, predictions, target_names=['Benign', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Aggiungi percentuali
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1%})', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, precision, recall, f1, predictions, targets, probabilities


def main_pipeline(model_size="small"):

    """Funzione principale per l'addestramento"""
    # Configura device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Inizializza dataset manager
        dataset_manager = NetworkTrafficDataset(
            model_size=model_size
        )
        
        # Carica i dati
        input_dim = dataset_manager.load_data(
            train_path="resources/datasets/train.csv",
            val_path="resources/datasets/val.csv",
            test_path="resources/datasets/test.csv"
        )
        
        # Crea DataLoader
        train_loader, val_loader, test_loader = dataset_manager.create_dataloaders()
        
        # Inizializza modello con class weights
        model = NetworkTrafficMLP(
            input_dim, 
            dataset_manager.hyperparams, 
            class_weights=dataset_manager.class_weights
        )
        logger.info(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")
        
        # Inizializza trainer
        trainer = ModelTrainer(
            model, 
            dataset_manager.hyperparams, 
            device, 
            class_weights=dataset_manager.class_weights
        )
        
        # Addestra il modello
        trained_model = trainer.train(train_loader, val_loader, epochs=100, patience=15)
        
        # Plotta la storia dell'addestramento
        os.makedirs("plots", exist_ok=True)
        trainer.plot_training_history('plots/training_history.png')
        
        # Valuta sul test set
        accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model(
            trained_model, test_loader, device
        )
        
        # Salva il modello
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/mlp_classifier_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        save_model(
            trained_model, 
            dataset_manager.hyperparams, 
            dataset_manager.feature_columns, 
            model_path,
            class_weights=dataset_manager.class_weights
        )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"ADDESTRAMENTO COMPLETATO!")
        logger.info(f"{'='*50}")
        logger.info(f"Miglior accuratezza: {accuracy:.4f}")
        logger.info(f"Modello salvato in: {model_path}")
        
    except Exception as e:
        logger.error(f"Errore durante l'addestramento: {str(e)}")
        raise


if __name__ == "__main__":
    main_pipeline()