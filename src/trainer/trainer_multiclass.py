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

from model.model_multiclass import NetworkTrafficMLPMulticlass, NetworkTrafficDatasetMulticlass, save_model_multiclass
from model.model_multiclass import FocalLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainerMulticlass:
    """Classe per gestire l'addestramento del modello MULTICLASS"""
    
    def __init__(self, model, hyperparams, device, n_classes, class_names, class_weights=None):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device
        self.n_classes = n_classes
        self.class_names = class_names
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        """
        if class_weights is not None:
            class_weights = class_weights.to(device)
            # Usa class weights MODERATI invece di amplificati
            moderate_weights = class_weights.clone()
            
            # Identifica l'indice di Benign (dovrebbe essere 2 dal tuo debug)
            benign_idx = 2  # Dall'analisi: Benign (2): 95.91%
            moderate_weights[benign_idx] = 1.0  # Benign peso normale
            
            # Limita i pesi delle altre classi a massimo 5x (invece di moltiplicare)
            for i in range(len(moderate_weights)):
                if i != benign_idx:  # Non Benign
                    moderate_weights[i] = min(moderate_weights[i], 5.0)  # Cap a massimo 5x
            
            self.criterion = nn.CrossEntropyLoss(weight=moderate_weights)
            logger.info(f"Usando CrossEntropyLoss con pesi MODERATI: {moderate_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Usando CrossEntropyLoss senza class weights")
        """
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
            # Usa Focal Loss con pesi moderati
            moderate_weights = class_weights.clone()
            
            benign_idx = 2  # Benign dal debug
            moderate_weights[benign_idx] = 1.0
            
            for i in range(len(moderate_weights)):
                if i != benign_idx:
                    moderate_weights[i] = min(moderate_weights[i], 3.0)  # Ancora più moderato per Focal Loss
            
            self.criterion = FocalLoss(alpha=1, gamma=2, weight=moderate_weights)
            logger.info(f"Usando FocalLoss con pesi moderati: {moderate_weights}")
        else:
            self.criterion = FocalLoss(alpha=1, gamma=2)
            logger.info("Usando FocalLoss senza class weights")
        
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Tracking delle metriche
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def add_noise_to_input(self, x, noise_factor):
        """Aggiunge rumore gaussiano all'input - IDENTICO AL BINARIO"""
        if self.model.training and noise_factor > 0:
            noise = torch.randn_like(x) * noise_factor
            return x + noise
        return x
    
    def train_epoch(self, train_loader, epoch):
        """Addestra per una epoch - ADATTATO PER MULTICLASS"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Contatori per classe
        class_correct = np.zeros(self.n_classes)
        class_total = np.zeros(self.n_classes)
        class_predicted = np.zeros(self.n_classes)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Debug: controlla il primo batch di ogni epoca
            if batch_idx == 0:
                unique_classes, counts = torch.unique(batch_y, return_counts=True)
                logger.info(f"Primo batch epoca {epoch}: classi presenti {unique_classes.cpu().numpy()} con counts {counts.cpu().numpy()}")
            
            # Aggiungi rumore
            if 'noise_factor' in self.hyperparams:
                batch_x = self.add_noise_to_input(batch_x, self.hyperparams['noise_factor'])
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistichs
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Statistiche per classe
            for class_idx in range(self.n_classes):
                class_mask = (batch_y == class_idx)
                class_total[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += ((predicted == class_idx) & class_mask).sum().item()
                class_predicted[class_idx] += (predicted == class_idx).sum().item()
            
            # Aggiorna progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log periodico
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Log statistiche per classe
        logger.info(f"Epoca {epoch} completata - Statistiche per classe:")
        for class_idx in range(self.n_classes):
            if class_total[class_idx] > 0:
                class_recall = class_correct[class_idx] / class_total[class_idx]
                class_precision = class_correct[class_idx] / class_predicted[class_idx] if class_predicted[class_idx] > 0 else 0
                logger.info(f"  {self.class_names[class_idx]}: Visti={int(class_total[class_idx])}, "
                           f"Predetti={int(class_predicted[class_idx])}, Corretti={int(class_correct[class_idx])}, "
                           f"Recall={class_recall:.3f}, Precision={class_precision:.3f}")
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Valuta il modello sul validation set - ADATTATO PER MULTICLASS"""
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
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Debug per validation
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        logger.info(f"Validation Debug:")
        for class_idx in range(self.n_classes):
            actual_count = (targets == class_idx).sum()
            predicted_count = (predictions == class_idx).sum()
            if actual_count > 0 or predicted_count > 0:
                logger.info(f"  {self.class_names[class_idx]}: Actual={actual_count}, Predicted={predicted_count}")
        
        return avg_loss, accuracy, predictions, targets, probabilities
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        """Addestra il modello - IDENTICO AL BINARIO"""
        logger.info(f"Inizio addestramento MULTICLASS per {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parametri del modello: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Classi: {self.n_classes} -> {self.class_names}")
        
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
            
            # Calcola metriche multiclass per validation - ADATTATO PER MULTICLASS
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_pred, average='weighted', zero_division=0
            )
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Val Precision (weighted): {precision:.4f}, Val Recall (weighted): {recall:.4f}, Val F1 (weighted): {f1:.4f}")
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
        """Plotta l'andamento dell'addestramento - IDENTICO AL BINARIO"""
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


def evaluate_model_multiclass(model, test_loader, device, class_names):
    """Valuta il modello multiclass sul test set - ADATTATO DA BINARIO"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Valutazione sul test set MULTICLASS...")
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calcola metriche multiclass
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(targets, predictions)
    
    # Metriche per classe e medie
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RISULTATI TEST SET MULTICLASS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Precision: {precision_weighted:.4f}")
    logger.info(f"Weighted Recall: {recall_weighted:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    
    # Metriche per classe
    logger.info(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            logger.info(f"  {class_name}: Precision={precision_per_class[i]:.4f}, "
                       f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Classification report dettagliato
    logger.info(f"\nClassification Report:")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix - Multiclass')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Aggiungi percentuali
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:  # Solo se c'è almeno un campione
                plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1%})', 
                        ha='center', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.show()
    
    # Plot distribuzione predizioni per classe
    plt.figure(figsize=(15, 5))
    
    # Distribuzione true labels
    plt.subplot(1, 3, 1)
    unique_true, counts_true = np.unique(targets, return_counts=True)
    true_names = [class_names[i] for i in unique_true]
    plt.bar(true_names, counts_true, color='skyblue', alpha=0.7)
    plt.title('True Label Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # Distribuzione predicted labels
    plt.subplot(1, 3, 2)
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    pred_names = [class_names[i] for i in unique_pred]
    plt.bar(pred_names, counts_pred, color='lightcoral', alpha=0.7)
    plt.title('Predicted Label Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # F1 score per classe
    plt.subplot(1, 3, 3)
    plt.bar(class_names, f1_per_class, color='lightgreen', alpha=0.7)
    plt.title('F1-Score per Class')
    plt.xticks(rotation=45)
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, precision_weighted, recall_weighted, f1_weighted, predictions, targets, probabilities


def main_pipeline_multiclass(model_size="small"):
    """Funzione principale per l'addestramento MULTICLASSE"""
    # Configura device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Inizializza dataset manager MULTICLASS
        dataset_manager = NetworkTrafficDatasetMulticlass(
            model_size=model_size,
            metadata_path="resources/datasets/multiclass_metadata.json"
        )
        
        # Carica i dati MULTICLASS
        input_dim = dataset_manager.load_data(
            train_path="resources/datasets/train_multiclass.csv",
            val_path="resources/datasets/val_multiclass.csv",
            test_path="resources/datasets/test_multiclass.csv"
        )
        
        # Crea DataLoader
        train_loader, val_loader, test_loader = dataset_manager.create_dataloaders()
        
        # Inizializza modello MULTICLASS
        model = NetworkTrafficMLPMulticlass(
            input_dim, 
            dataset_manager.hyperparams,
            dataset_manager.n_classes,
            class_weights=dataset_manager.class_weights
        )
        logger.info(f"Modello MULTICLASS creato con {sum(p.numel() for p in model.parameters())} parametri")
        
        # Inizializza trainer MULTICLASS
        class_names = dataset_manager.multiclass_metadata['label_encoder_classes']
        trainer = ModelTrainerMulticlass(
            model, 
            dataset_manager.hyperparams, 
            device,
            dataset_manager.n_classes,
            class_names,
            class_weights=dataset_manager.class_weights
        )
        
        # Addestra il modello
        trained_model = trainer.train(train_loader, val_loader, epochs=100, patience=15)
        
        # Plotta la storia dell'addestramento
        os.makedirs("plots", exist_ok=True)
        trainer.plot_training_history('plots/training_history_multiclass.png')
        
        # Valuta sul test set
        accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model_multiclass(
            trained_model, test_loader, device, class_names
        )
        
        # Salva il modello MULTICLASS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/mlp_multiclass_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        save_model_multiclass(
            trained_model, 
            dataset_manager.hyperparams, 
            dataset_manager.feature_columns, 
            model_path,
            dataset_manager.n_classes,
            dataset_manager.class_mapping,
            class_weights=dataset_manager.class_weights
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ADDESTRAMENTO MULTICLASS COMPLETATO!")
        logger.info(f"{'='*60}")
        logger.info(f"Migliori metriche:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Weighted F1: {f1:.4f}")
        logger.info(f"  Classes: {dataset_manager.n_classes}")
        logger.info(f"Modello salvato in: {model_path}")
        
        return trained_model, accuracy, f1
        
    except Exception as e:
        logger.error(f"Errore durante l'addestramento multiclass: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main_pipeline_multiclass(model_size="small")