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

from utilities.logging_config import setup_logging
from model.model_transformer import NetworkTrafficDatasetTransformer, NetworkTrafficTransformer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class ModelTrainerTransformer:
    def __init__(self, model, hyperparams, device, n_classes, class_names, class_weights=None, class_freq=None):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device
        self.n_classes = n_classes
        self.class_names = class_names
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams.get('weight_decay', 1e-4)
        )
        
        # Funzione di Loss
        self.criterion = nn.CrossEntropyLoss()
        logger.info("✅ Usando CrossEntropyLoss standard")
        
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
        """Training epoch per Transformer con attention masks"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Contatori per classe
        class_correct = np.zeros(self.n_classes)
        class_total = np.zeros(self.n_classes)
        class_predicted = np.zeros(self.n_classes)
        
        total_batches = len(train_loader)
        log_interval_batches = max(1, total_batches // 10)
        next_log_batch = log_interval_batches
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (batch_sequences, batch_labels, batch_masks) in enumerate(progress_bar):
            batch_sequences = batch_sequences.to(self.device)  # (batch, seq_len, features)
            batch_labels = batch_labels.to(self.device)        # (batch,)
            batch_masks = batch_masks.to(self.device)          # (batch, seq_len)
            
            if batch_idx == 0:
                logger.info(f"Primo batch epoca {epoch}:")
                logger.info(f"  Sequences shape: {batch_sequences.shape}")
                logger.info(f"  Labels shape: {batch_labels.shape}")
                logger.info(f"  Masks shape: {batch_masks.shape}")
                unique_classes, counts = torch.unique(batch_labels, return_counts=True)
                logger.info(f"  Classi: {unique_classes.cpu().numpy()} con counts {counts.cpu().numpy()}")
            
            # Forward pass con attention mask
            self.optimizer.zero_grad()
            outputs = self.model(batch_sequences, attention_mask=batch_masks)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistiche per Transformer (sequenze, non pacchetti singoli)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Statistiche per classe (su sequenze)
            for class_idx in range(self.n_classes):
                class_mask = (batch_labels == class_idx)
                class_total[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += ((predicted == class_idx) & class_mask).sum().item()
                class_predicted[class_idx] += (predicted == class_idx).sum().item()
            
            current_progress = (batch_idx + 1) / total_batches * 100
            progress_bar.set_postfix({
                'Progress': f'{current_progress:.1f}%',
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Sequences': total
            })
            
            should_log = (
                batch_idx == 0 or  # Primo batch
                batch_idx + 1 == total_batches or  # Ultimo batch
                batch_idx + 1 >= next_log_batch  # Ogni 10%
            )
            
            if should_log:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                accuracy_so_far = 100. * correct / total
                
                logger.info(f'Epoch {epoch} - {current_progress:.1f}% completato '
                        f'({batch_idx + 1:,}/{total_batches:,} batch) | '
                        f'Loss: {avg_loss_so_far:.4f} | '
                        f'Acc: {accuracy_so_far:.2f}% | '
                        f'Sequenze: {total:,}')
                
                if batch_idx + 1 < total_batches:
                    next_log_batch += log_interval_batches
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Log finale per classe con informazioni specifiche Transformer
        logger.info(f"Epoca {epoch} completata - Statistiche per classe (su {total:,} sequenze):")
        
        class_metrics = []
        for class_idx in range(self.n_classes):
            if class_total[class_idx] > 0:
                class_recall = class_correct[class_idx] / class_total[class_idx]
                class_precision = class_correct[class_idx] / class_predicted[class_idx] if class_predicted[class_idx] > 0 else 0
                
                # Calcola F1 score
                if class_recall + class_precision > 0:
                    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
                else:
                    class_f1 = 0.0
                
                class_metrics.append({
                    'name': self.class_names[class_idx],
                    'f1': class_f1,
                    'recall': class_recall,
                    'precision': class_precision,
                    'visti': int(class_total[class_idx]),
                    'predetti': int(class_predicted[class_idx]),
                    'corretti': int(class_correct[class_idx])
                })

        # Ordina per F1 (peggiori prima)
        class_metrics.sort(key=lambda x: x['f1'])

        # Mostra classi con F1
        for cls in class_metrics:
            # Emoji basato su F1 score
            if cls['f1'] < 0.3:
                emoji = "🔴"  # Critico
            elif cls['f1'] < 0.5:
                emoji = "🟡"  # Problematico  
            elif cls['f1'] < 0.8:
                emoji = "🟢"  # Buono
            else:
                emoji = "✅"  # Ottimo
            
            logger.info(f"  {emoji} {cls['name']:15s}: "
                    f"F1={cls['f1']:.3f} | "
                    f"R={cls['recall']:.3f} | "
                    f"P={cls['precision']:.3f} | "
                    f"Seq_viste={cls['visti']:4d}")

        # Calcola F1 macro
        macro_f1 = sum(cls['f1'] for cls in class_metrics) / len(class_metrics)
        critical_count = len([cls for cls in class_metrics if cls['f1'] < 0.2])
        logger.info(f"  📈 F1 Macro: {macro_f1:.3f} | Classi critiche: {critical_count} | "
                    f"Avg seq/batch: {total/total_batches:.1f}")  # ← NUOVO: info sequenze per batch

        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validation per Transformer con attention masks"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels, batch_masks in val_loader:
                batch_sequences = batch_sequences.to(self.device)  # (batch, seq_len, features)
                batch_labels = batch_labels.to(self.device)        # (batch,)
                batch_masks = batch_masks.to(self.device)          # (batch, seq_len)
                
                # Forward con attention mask
                outputs = self.model(batch_sequences, attention_mask=batch_masks)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Conversione arrays per analisi
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Debug per validation con informazioni Transformer
        logger.info(f"Validation Debug (su {total:,} sequenze):")
        
        total_sequences_analyzed = 0
        for class_idx in range(self.n_classes):
            actual_count = (targets == class_idx).sum()
            predicted_count = (predictions == class_idx).sum()
            total_sequences_analyzed += actual_count
            
            if actual_count > 0 or predicted_count > 0:
                # Calcola accuracy per questa classe specifica
                class_mask = (targets == class_idx)
                if actual_count > 0:
                    class_accuracy = (predictions[class_mask] == class_idx).sum() / actual_count
                    logger.info(f"  {self.class_names[class_idx]:15s}: "
                            f"Actual={actual_count:4d} | "
                            f"Predicted={predicted_count:4d} | "
                            f"Acc={class_accuracy:.3f}")
                else:
                    logger.info(f"  {self.class_names[class_idx]:15s}: "
                            f"Actual={actual_count:4d} | "
                            f"Predicted={predicted_count:4d} | "
                            f"Acc=N/A")
        
        # Statistiche aggregate
        logger.info(f"Validation Summary:")
        logger.info(f"  Total sequences: {total:,}")
        logger.info(f"  Correctly classified: {correct:,}")
        logger.info(f"  Overall accuracy: {accuracy:.2f}%")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        
        # Analisi distribuzione confidenza
        avg_confidence = probabilities.max(axis=1).mean()
        low_confidence_count = (probabilities.max(axis=1) < 0.5).sum()
        
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        logger.info(f"  Low confidence predictions (<0.5): {low_confidence_count:,} ({low_confidence_count/total*100:.1f}%)")
        
        return avg_loss, accuracy, predictions, targets, probabilities

    def train(self, train_loader, val_loader, epochs=100, patience=15):
        """Addestra il modello Transformer"""
        logger.info(f"Inizio addestramento TRANSFORMER per {epochs} epochs...")
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
            
            # Calcola metriche multiclass per validation
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
                logger.info("✅ Nuovo miglior modello salvato!")
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
        """Plotta l'andamento dell'addestramento Transformer"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        ax1.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        ax1.plot(epochs, self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Transformer Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Transformer Model Accuracy')
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

def evaluate_model_transformer(model, test_loader, device, class_names):
    """Valuta il modello Transformer sul test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Valutazione sul test set TRANSFORMER...")
    
    with torch.no_grad():
        for batch_sequences, batch_labels, batch_masks in tqdm(test_loader, desc="Testing Transformer"):
            batch_sequences = batch_sequences.to(device)  # (batch, seq_len, features)
            batch_labels = batch_labels.to(device)        # (batch,)
            batch_masks = batch_masks.to(device)          # (batch, seq_len)
            
            outputs = model(batch_sequences, attention_mask=batch_masks)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
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
    logger.info(f"RISULTATI TEST SET TRANSFORMER MULTICLASS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Precision: {precision_weighted:.4f}")
    logger.info(f"Weighted Recall: {recall_weighted:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    
    # Metriche per classe
    logger.info(f"\nPer-Class Metrics (Sequences):")
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            logger.info(f"  {class_name}: Precision={precision_per_class[i]:.4f}, "
                       f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Classification report dettagliato
    logger.info(f"\nClassification Report (Transformer):")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix - Transformer Multiclass (Sequences)')
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
    
    # Plot distribuzione predizioni per classe - MODIFICATO per Transformer
    plt.figure(figsize=(15, 5))
    
    # Distribuzione true labels
    plt.subplot(1, 3, 1)
    unique_true, counts_true = np.unique(targets, return_counts=True)
    true_names = [class_names[i] for i in unique_true]
    plt.bar(true_names, counts_true, color='skyblue', alpha=0.7)
    plt.title('True Sequence Label Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Sequence Count')
    
    # Distribuzione predicted labels
    plt.subplot(1, 3, 2)
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    pred_names = [class_names[i] for i in unique_pred]
    plt.bar(pred_names, counts_pred, color='lightcoral', alpha=0.7)
    plt.title('Predicted Sequence Label Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Sequence Count')
    
    # F1 score per classe
    plt.subplot(1, 3, 3)
    plt.bar(class_names, f1_per_class, color='lightgreen', alpha=0.7)
    plt.title('F1-Score per Class (Sequences)')
    plt.xticks(rotation=45)
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # NUOVO: Analisi attention/confidence specifiche per Transformer
    logger.info(f"\n--- ANALISI TRANSFORMER SPECIFICHE ---")
    
    # Analisi distribuzione confidence
    confidence_scores = probabilities.max(axis=1)
    avg_confidence = confidence_scores.mean()
    std_confidence = confidence_scores.std()
    
    logger.info(f"Confidence Analysis:")
    logger.info(f"  Average confidence: {avg_confidence:.3f} ± {std_confidence:.3f}")
    logger.info(f"  High confidence (>0.9): {(confidence_scores > 0.9).sum():,} ({(confidence_scores > 0.9).mean()*100:.1f}%)")
    logger.info(f"  Medium confidence (0.5-0.9): {((confidence_scores >= 0.5) & (confidence_scores <= 0.9)).sum():,}")
    logger.info(f"  Low confidence (<0.5): {(confidence_scores < 0.5).sum():,} ({(confidence_scores < 0.5).mean()*100:.1f}%)")
    
    # Analisi lunghezza sequenze vs performance (se possibile)
    logger.info(f"Total sequences classified: {len(predictions):,}")
    
    return accuracy, precision_weighted, recall_weighted, f1_weighted, predictions, targets, probabilities

def main_pipeline_transformer(model_size="small"):
    """Pipeline principale per Transformer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Dataset manager per Transformer
        dataset_manager = NetworkTrafficDatasetTransformer(
            model_size=model_size,
            metadata_path="resources/datasets/transformer_metadata.json"
        )
        
        # Carica dati sequenziali
        n_features = dataset_manager.load_data("resources/datasets")
        train_loader, val_loader, test_loader = dataset_manager.create_dataloaders()
        
        # Modello Transformer
        model = NetworkTrafficTransformer(
            config=dataset_manager.hyperparams,
            n_classes=dataset_manager.n_classes,
            n_features=n_features
        )
        
        # Trainer (quasi identico, cambia solo il training loop)
        class_names = dataset_manager.metadata['label_encoder_classes']
        trainer = ModelTrainerTransformer(
            model, 
            dataset_manager.hyperparams, 
            device,
            dataset_manager.n_classes,
            class_names
        )
        
        # Training
        trained_model = trainer.train(train_loader, val_loader, epochs=50, patience=10)
        
        # Plot training history
        import os
        os.makedirs("plots", exist_ok=True)
        trainer.plot_training_history('plots/training_history_transform.png')
        
        accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model_transformer(
            trained_model, test_loader, device, class_names
        )
        
        # Salvataggio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/transformer_multiclass_{timestamp}.pth"
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'hyperparams': dataset_manager.hyperparams,
            'n_features': n_features,
            'n_classes': dataset_manager.n_classes,
            'class_mapping': dataset_manager.class_mapping,
            'model_type': 'transformer_multiclass'
        }, model_path)
        
        logger.info(f"🎯 TRANSFORMER TRAINING COMPLETATO")
        logger.info(f"Modello salvato: {model_path}")
        
        return trained_model, accuracy, f1
    
    except Exception as e:
        logger.error(f"Errore durante training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main_pipeline_transformer(model_size="small")