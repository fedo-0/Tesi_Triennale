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
from model.model_transformer import NetworkTrafficTransformer, NetworkTrafficDatasetTransformer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class ModelTrainerTransformer:

    def __init__(self, model, hyperparams, device, n_classes, class_names, dataset_manager, class_weights=None):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device
        self.n_classes = n_classes
        self.class_names = class_names
        self.dataset_manager = dataset_manager

        # Tracking delle metriche
        self.train_losses = []
        self.val_losses = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_accuracies = []
        self.learning_rates = []
        
        self.last_val_predictions = None
        self.last_val_targets = None
        
        # Optimizer con learning rate più basso per Transformer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            logger.info("✅ Usando CrossEntropyLoss con class weights")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Usando CrossEntropyLoss standard")
        
        # Scheduler più appropriato per Transformer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=8, min_lr=1e-6
        )
        
        logger.info(f"Trainer Transformer inizializzato:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {hyperparams['lr']}")
        logger.info(f"  Batch size: {hyperparams['batch_size']}")
        logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    def add_noise_to_sequences(self, sequences, noise_factor):
        """Aggiunge rumore gaussiano alle sequenze temporali"""
        if self.model.training and noise_factor > 0:
            noise = torch.randn_like(sequences) * noise_factor
            return sequences + noise
        return sequences
    
    def train_epoch(self, train_loader, epoch):

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
        
        for batch_idx, (batch_sequences, batch_targets) in enumerate(progress_bar):
            batch_sequences = batch_sequences.to(self.device)  # (batch_size, seq_len, features)
            batch_targets = batch_targets.to(self.device)      # (batch_size,)
            
            if batch_idx == 0:
                unique_classes, counts = torch.unique(batch_targets, return_counts=True)
                logger.info(f"Primo batch epoca {epoch}: classi {unique_classes.cpu().numpy()} con counts {counts.cpu().numpy()}")
            
            # Aggiunta rumore (se configurato)
            if 'noise_factor' in self.hyperparams:
                batch_sequences = self.add_noise_to_sequences(batch_sequences, self.hyperparams['noise_factor'])
            
            # Separa features numeriche e categoriche
            numeric_features, categorical_features = self.dataset_manager.separate_features(batch_sequences)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(numeric_features, categorical_features)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping più aggressivo per Transformer
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Statistiche
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_targets.size(0)
            correct += (predicted == batch_targets).sum().item()
            
            # Statistiche per classe
            for class_idx in range(self.n_classes):
                class_mask = (batch_targets == class_idx)
                class_total[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += ((predicted == class_idx) & class_mask).sum().item()
                class_predicted[class_idx] += (predicted == class_idx).sum().item()
            
            current_progress = (batch_idx + 1) / total_batches * 100
            progress_bar.set_postfix({
                'Progress': f'{current_progress:.1f}%',
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            should_log = (
                batch_idx == 0 or
                batch_idx + 1 == total_batches or
                batch_idx + 1 >= next_log_batch
            )
            
            if should_log:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                accuracy_so_far = 100. * correct / total
                
                logger.info(f'Epoch {epoch} - {current_progress:.1f}% completato '
                        f'({batch_idx + 1:,}/{total_batches:,} batch) | '
                        f'Loss: {avg_loss_so_far:.4f} | '
                        f'Acc: {accuracy_so_far:.2f}%')
                
                if batch_idx + 1 < total_batches:
                    next_log_batch += log_interval_batches
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Log finale per classe
        logger.info(f"Epoca {epoch} completata - Statistiche temporali per classe:")
        
        class_metrics = []
        for class_idx in range(self.n_classes):
            if class_total[class_idx] > 0:
                class_recall = class_correct[class_idx] / class_total[class_idx]
                class_precision = class_correct[class_idx] / class_predicted[class_idx] if class_predicted[class_idx] > 0 else 0
                
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
                    f"Sequenze={cls['visti']:4d}")

        # Calcola F1 macro
        macro_f1 = sum(cls['f1'] for cls in class_metrics) / len(class_metrics)
        critical_count = len([cls for cls in class_metrics if cls['f1'] < 0.2])
        logger.info(f"  📈 F1 Macro: {macro_f1:.3f} | Classi critiche: {critical_count}")

        return avg_loss, accuracy
    
    def validate(self, val_loader):

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Separa features numeriche e categoriche
                numeric_features, categorical_features = self.dataset_manager.separate_features(batch_sequences)
                
                outputs = self.model(numeric_features, categorical_features)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Debug per validation
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        logger.info(f"Validation Debug (sequenze temporali):")
        for class_idx in range(self.n_classes):
            actual_count = (targets == class_idx).sum()
            predicted_count = (predictions == class_idx).sum()
            if actual_count > 0 or predicted_count > 0:
                logger.info(f"  {self.class_names[class_idx]}: Actual={actual_count}, Predicted={predicted_count}")
        
        return avg_loss, accuracy, predictions, targets, probabilities
    
    def train(self, train_loader, val_loader, epochs=100, patience=10):

        logger.info(f"Inizio addestramento TRANSFORMER TEMPORALE per {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parametri del modello: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Classi: {self.n_classes} -> {self.class_names}")
        logger.info(f"Sequenze per batch: {self.hyperparams['batch_size']}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Salva learning rate corrente
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)
            logger.info(f"Terminata l'epoca: {epoch+1}")
            
            # Validation
            val_loss, val_acc, val_pred, val_true, val_prob = self.validate(val_loader)
            
            # Salva l'ultima validazione
            self.last_val_predictions = val_pred
            self.last_val_targets = val_true

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

            self.val_precisions.append(precision)
            self.val_recalls.append(recall)
            
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
                if val_loss > train_loss * 1.2:
                    overfitting_gap = val_loss - train_loss
                    logger.info(f"⚠️  Overfitting rilevato: val_loss ({val_loss:.4f}) > train_loss ({train_loss:.4f}), gap: {overfitting_gap:.4f}")
                    logger.info(f"Stopping per overfitting dopo {epoch+1} epochs")
                    break
                if patience_counter >= patience:
                    logger.info(f"Early stopping dopo {epoch+1} epochs (patience: {patience})")
                    break
        
        # Carica il miglior modello
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Caricato il miglior modello per la valutazione finale.")

            # Rifai validazione con il miglior modello per avere predizioni corrette
            logger.info("Ricalcolo validazione finale con il miglior modello...")
            val_loss_final, val_acc_final, val_pred_final, val_true_final, val_prob_final = self.validate(val_loader)
            self.last_val_predictions = val_pred_final
            self.last_val_targets = val_true_final

        return self.model
    
    def plot_training_history(self, save_path=None, val_predictions=None, val_targets=None):

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        # Plot principale con Loss, Accuracy, Precision e Recall
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        ax1.plot(epochs, self.train_losses, label='Training Loss', color='blue', linewidth=2)
        ax1.plot(epochs, self.val_losses, label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Transformer Loss (Sequenze Temporali)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
        ax2.set_title('Transformer Accuracy (Sequenze Temporali)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision e Recall
        precision_history = self.val_precisions if hasattr(self, 'val_precisions') and self.val_precisions else []
        recall_history = self.val_recalls if hasattr(self, 'val_recalls') and self.val_recalls else []
        
        # Precision
        if precision_history:
            ax3.plot(epochs, precision_history, label='Validation Precision (Weighted)', 
                    color='orange', linewidth=2)
            ax3.set_title('Transformer Precision (Sequenze Temporali)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Precision data not available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Transformer Precision', fontsize=14, fontweight='bold')
        
        # Recall
        if recall_history:
            ax4.plot(epochs, recall_history, label='Validation Recall (Weighted)', 
                    color='purple', linewidth=2)
            ax4.set_title('Transformer Recall (Sequenze Temporali)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Recall data not available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Transformer Recall', fontsize=14, fontweight='bold')
        
        # Calcola e mostra metriche finali se disponibili
        if val_predictions is not None and val_targets is not None:
            # Metriche globali
            final_accuracy = accuracy_score(val_targets, val_predictions)
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='weighted', zero_division=0
            )
            precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='macro', zero_division=0
            )
            
            # Aggiungi testo con metriche finali
            metrics_text = (
                f"Final Validation Metrics (Temporal):\n"
                f"Accuracy: {final_accuracy:.4f}\n"
                f"Weighted F1: {f1_w:.4f}\n"
                f"Weighted Precision: {precision_w:.4f}\n"
                f"Weighted Recall: {recall_w:.4f}\n"
                f"Macro F1: {f1_m:.4f}\n"
                f"Macro Precision: {precision_m:.4f}\n"
                f"Macro Recall: {recall_m:.4f}"
            )
            
            # Posiziona il testo nell'area del grafico
            fig.text(0.02, 0.98, metrics_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            logger.info("\n" + "="*60)
            logger.info("METRICHE FINALI VALIDATION TRANSFORMER TEMPORALE")
            logger.info("="*60)
            logger.info(f"Accuracy: {final_accuracy:.4f}")
            logger.info(f"Weighted - Precision: {precision_w:.4f}, Recall: {recall_w:.4f}, F1: {f1_w:.4f}")
            logger.info(f"Macro - Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1: {f1_m:.4f}")
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot salvato in: {save_path}")
        
        #plt.show()
        
        # Crea plot separato per metriche per classe se disponibili
        if val_predictions is not None and val_targets is not None:
            self._plot_per_class_metrics(val_predictions, val_targets, save_path)
        else:
            if self.last_val_predictions is not None and self.last_val_targets is not None:
                self._plot_per_class_metrics(self.last_val_predictions, self.last_val_targets, save_path)

    def _plot_per_class_metrics(self, val_predictions, val_targets, base_save_path=None):

        from sklearn.metrics import precision_recall_fscore_support
        import matplotlib.cm as cm
        
        # Calcola metriche per classe
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            val_targets, val_predictions, average=None, zero_division=0
        )
        
        # Calcola F1 weighted per "aggregated"
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            val_targets, val_predictions, average='weighted', zero_division=0
        )
        
        # Crea figure con layout 2x2
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x_pos = np.arange(len(self.class_names))
        
        # F1 Score per classe con colori diversi + aggregated
        colors = cm.Set3(np.linspace(0, 1, len(self.class_names)))

        # Aggiunge F1 weighted per "aggregated"
        x_pos_agg = np.arange(len(self.class_names) + 1)
        f1_with_agg = np.append(f1_per_class, f1_weighted)
        class_names_with_agg = list(self.class_names) + ['Aggregated']
        
        # Colori: classi normali + grigio per aggregated
        colors_agg = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
        
        bars1 = ax1.bar(x_pos_agg, f1_with_agg, color=colors_agg, alpha=1.0)
        ax1.set_title('F1-Score per Classe + Aggregated (Transformer)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classi')
        ax1.set_ylabel('F1-Score')
        ax1.set_xticks(x_pos_agg)
        ax1.set_xticklabels(class_names_with_agg, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
         
        # Precision per classe
        bars2 = ax2.bar(x_pos, precision_per_class, color=colors, alpha=1.0)
        ax2.set_title('Precision per Classe (Transformer)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classi')
        ax2.set_ylabel('Precision')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Recall per classe
        bars3 = ax3.bar(x_pos, recall_per_class, color=colors, alpha=1.0)
        ax3.set_title('Recall per Classe (Transformer)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Classi')
        ax3.set_ylabel('Recall')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Support (numero sequenze) per classe
        bars4 = ax4.bar(x_pos, support_per_class, color=colors, alpha=1.0)
        ax4.set_title('Support per Classe (Numero Sequenze)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Classi')
        ax4.set_ylabel('Numero Sequenze')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva il plot per classe se specificato
        if base_save_path:
            per_class_save_path = base_save_path.replace('.png', '_transformer_per_class_metrics.png')
            plt.savefig(per_class_save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot salvato in: {per_class_save_path}")
        
        #plt.show()
        
        # Log delle metriche per classe
        logger.info("\n" + "="*70)
        logger.info("METRICHE PER CLASSE TRANSFORMER TEMPORALE (VALIDATION)")
        logger.info("="*70)
        
        # Crea una tabella ordinata per F1
        class_metrics_data = []
        for i, class_name in enumerate(self.class_names):
            class_metrics_data.append({
                'name': class_name,
                'f1': f1_per_class[i],
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'support': support_per_class[i]
            })
        
        # Aggiungi metriche aggregate
        class_metrics_data.append({
            'name': 'AGGREGATED (Weighted)',
            'f1': f1_weighted,
            'precision': precision_recall_fscore_support(val_targets, val_predictions, average='weighted', zero_division=0)[0],
            'recall': precision_recall_fscore_support(val_targets, val_predictions, average='weighted', zero_division=0)[1],
            'support': len(val_targets)
        })
        
        # Ordina per F1 (peggiori prima, ma aggregated sempre ultimo)
        class_metrics_individual = [x for x in class_metrics_data if 'AGGREGATED' not in x['name']]
        class_metrics_individual.sort(key=lambda x: x['f1'])
        class_metrics_aggregated = [x for x in class_metrics_data if 'AGGREGATED' in x['name']]
        class_metrics_data = class_metrics_individual + class_metrics_aggregated
        
        for cls_data in class_metrics_data:
            if cls_data['f1'] < 0.3:
                status = "🔴"
            elif cls_data['f1'] < 0.5:
                status = "🟡"
            elif cls_data['f1'] < 0.8:
                status = "🟢"
            else:
                status = "✅"
            
            if 'AGGREGATED' in cls_data['name']:
                status = "📊"
                logger.info("-" * 70)
            
            logger.info(f"{status} {cls_data['name']:20s}: "
                    f"F1={cls_data['f1']:.4f} | "
                    f"P={cls_data['precision']:.4f} | "
                    f"R={cls_data['recall']:.4f} | "
                    f"Seq={int(cls_data['support']):5d}")


def evaluate_model_transformer(model, test_loader, device, class_names, dataset_manager):

    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Valutazione sul test set TRANSFORMER TEMPORALE...")
    
    with torch.no_grad():
        for batch_sequences, batch_targets in tqdm(test_loader, desc="Testing Temporal Sequences"):
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Separa features numeriche e categoriche
            numeric_features, categorical_features = dataset_manager.separate_features(batch_sequences)
            
            outputs = model(numeric_features, categorical_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
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
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RISULTATI TEST SET TRANSFORMER TEMPORALE")
    logger.info(f"{'='*70}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Precision: {precision_weighted:.4f}")
    logger.info(f"Weighted Recall: {recall_weighted:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    
    # Classification report dettagliato
    logger.info(f"\nClassification Report (Temporal Sequences):")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix - Transformer Temporale')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Aggiungi percentuali
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1%})', 
                        ha='center', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    #plt.show()
    
    return accuracy, precision_weighted, recall_weighted, f1_weighted, predictions, targets, probabilities


def main_pipeline_transformer(model_size="small", window_size=8):
    """Pipeline principale per training Transformer temporale"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        dataset_manager = NetworkTrafficDatasetTransformer(
            model_size=model_size,
            metadata_path="resources/datasets/transformer_metadata.json"
        )
        
        feature_dim = dataset_manager.load_processed_data(
            train_csv_path="resources/datasets/train_transformer_processed.csv",
            val_csv_path="resources/datasets/val_transformer_processed.csv",
            test_csv_path="resources/datasets/test_transformer_processed.csv"
        )
        

        train_loader, val_loader, test_loader = dataset_manager.create_dataloaders(
            window_size=window_size, seed=42
        )
        
        # Modello Transformer
        model = NetworkTrafficTransformer(
            dataset_manager.hyperparams,
            dataset_manager.n_classes,
            dataset_manager.feature_groups,
            dataset_manager.vocab_sizes
        )

        
        # Trainer
        class_names = dataset_manager.temporal_metadata['label_encoder_classes']
        trainer = ModelTrainerTransformer(
            model, 
            dataset_manager.hyperparams, 
            device,
            dataset_manager.n_classes,
            class_names,
            dataset_manager,
            class_weights=dataset_manager.class_weights
        )
        
        # Training con patience maggiore per Transformer
        trained_model = trainer.train(train_loader, val_loader, epochs=100, patience=10)
        
        # Plot training history
        import os
        os.makedirs("plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.plot_training_history(f'plots/transformer_training_history_{timestamp}.png')
        
        # Evaluation
        accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model_transformer(
            trained_model, test_loader, device, class_names, dataset_manager
        )
        
        # Salvataggio modello
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/transformer_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        
        from model.model_transformer import save_model_transformer
        save_model_transformer(
            trained_model,
            dataset_manager.hyperparams,
            dataset_manager.feature_columns,
            model_path,
            dataset_manager.n_classes,
            dataset_manager.class_mapping,
            dataset_manager.feature_groups,
            dataset_manager.vocab_sizes,
            dataset_manager.class_weights,
            dataset_manager.temporal_metadata['temporal_config']
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ADDESTRAMENTO TRANSFORMER COMPLETATO")
        logger.info(f"{'='*70}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {f1:.4f}")
        logger.info(f"Modello salvato: {model_path}")
        logger.info(f"Plot salvato: plots/training_history_transformer.png")
        
        return trained_model, accuracy, f1
        
    except Exception as e:
        logger.error(f"Errore durante training Transformer: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main_pipeline_transformer(model_size="small", window_size=8)