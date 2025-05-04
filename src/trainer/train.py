import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
from sklearn.model_selection import KFold
import random

from data.preprocessing import NetworkFlowDataset, load_dataset
from model.binary_classificator import NetworkTrafficCNN

import logging
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Set seed for reproducibility across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")

class EarlyStopping:
    """
    Early stopping handler to stop training when validation performance worsens.
    """
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping
            verbose (bool): If True, prints a message for each improvement
            delta (float): Minimum change to qualify as an improvement
            path (str): Path to save the best model
            trace_func (function): Function used for logging
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.improvement = False

    def __call__(self, val_loss, model, epoch, optimizer, metrics):
        """
        Check if validation loss has improved.
        """
        score = -val_loss
        self.improvement = False

        if self.best_score is None:
            self.improvement = True
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, metrics)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.improvement = True
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, metrics)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, epoch, optimizer, metrics):
        """
        Save model when validation loss improves.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': metrics.get('train_loss', None),
            'val_loss': val_loss,
            'val_metrics': {
                'accuracy': metrics.get('val_acc', None),
                'precision': metrics.get('val_precision', None),
                'recall': metrics.get('val_recall', None),
                'f1': metrics.get('val_f1', None)
            }
        }, self.path)
        
        self.val_loss_min = val_loss

def run_training_pipeline(config_path, csv_path, outputModel_path, outputResults_path, model_size, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    logger.info("Config: %s, CSV: %s, Model: %s, Results: %s, Model-size: %s",
            config_path, csv_path, outputModel_path, outputResults_path, model_size)
    
    hyperparams_path="config/hyperparameters.json"
    # Load hyperparameters from JSON
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    assert model_size in hyperparams, f"Model size '{model_size}' not in '{hyperparams_path}'"
    params = hyperparams[model_size]

    batch_size = params["batch_size"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    dropout = params["dropout"]
    fc_dim = params["ff_dim"]  # We map feed-forward dim to fc layer size

    # Aggiungiamo parametri per k-fold cross-validation
    use_cross_validation = params.get("use_cross_validation", False)
    num_folds = params.get("num_folds", 5)

    os.makedirs(outputModel_path, exist_ok=True)
    os.makedirs(outputResults_path, exist_ok=True)

    # -------------------------------
    # 1. Load configuration from JSON and dataset from CSV
    # -------------------------------
    df, numeric_columns, categorical_columns, target_column = load_dataset(config_path, csv_path)

    # Check class balance
    class_counts = df[target_column].value_counts()
    print(f"Class distribution: {class_counts}")
    print(f"Class proportions: {class_counts / len(df)}")

    # -------------------------------
    # 2. Create a timestamp for this run
    # -------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_metrics_path = f"{outputResults_path}run_metrics_{timestamp}/"
    os.makedirs(run_metrics_path, exist_ok=True)

    # -------------------------------
    # 3. Create the full dataset
    # -------------------------------
    full_dataset = NetworkFlowDataset(df, numeric_columns, categorical_columns, target_column)
    
    # Parameters for the model
    numeric_dim = len(numeric_columns)
    categorical_info = {
        col: (full_dataset.cat_dims[col], 
            min(16, full_dataset.cat_dims[col] // 2) if full_dataset.cat_dims[col] > 100 else 
            min(8, full_dataset.cat_dims[col] // 2 if full_dataset.cat_dims[col] > 2 else 2))
        for col in categorical_columns
    }

    # Print embedding dimensions for better understanding
    print("\nEmbedding dimensions:")
    for col, (vocab_size, embed_dim) in categorical_info.items():
        print(f"{col}: vocabulary size = {vocab_size}, embedding dimension = {embed_dim}")

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_cross_validation:
        # -------------------------------
        # 4a. K-Fold Cross Validation
        # -------------------------------
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_results = []
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
            print(f"\n{'='*50}")
            print(f"FOLD {fold+1}/{num_folds}")
            print(f"{'='*50}")
            
            # Sample validation set from training data
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
            
            # Further split train_ids into train and validation
            train_ids_list = list(train_ids)
            val_size = int(0.15 * len(train_ids_list))
            val_ids = train_ids_list[:val_size]
            actual_train_ids = train_ids_list[val_size:]
            
            train_subsampler = SubsetRandomSampler(actual_train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            
            # Create data loaders
            train_loader = DataLoader(
                full_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
            )
            val_loader = DataLoader(
                full_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
            )
            test_loader = DataLoader(
                full_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=4,
                worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
            )
            
            # Reset the model for each fold
            model = NetworkTrafficCNN(
                numeric_dim=numeric_dim,
                categorical_info=categorical_info,
                conv_channels=64,
                kernel_size=5,
                fc_dim=fc_dim,
                dropout=dropout
            )
            model = model.to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            # Define model path for this fold
            fold_model_path = f"{outputModel_path}fold_{fold+1}_model_{timestamp}.pt"
            
            # Initialize early stopping
            early_stopping = EarlyStopping(
                patience=7, 
                verbose=True, 
                path=fold_model_path,
                trace_func=print
            )
            
            # Training loop for this fold
            fold_metrics = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, early_stopping, num_epochs=30, fold=fold+1
            )
            
            # Evaluate on test set
            model.load_state_dict(torch.load(fold_model_path)['model_state_dict'])
            test_metrics = evaluate_model(model, test_loader, criterion, device)
            
            # Save fold metrics
            fold_results.append({
                'fold': fold+1,
                'test_loss': test_metrics['loss'],
                'test_acc': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'history': fold_metrics
            })
            
            # Save confusion matrix for this fold
            cm = confusion_matrix(test_metrics['targets'], test_metrics['predictions'])
            cm_df = pd.DataFrame(cm, 
                                index=['Actual Benign', 'Actual Malicious'],
                                columns=['Predicted Benign', 'Predicted Malicious'])
            cm_csv_path = f"{run_metrics_path}fold_{fold+1}_confusion_matrix.csv"
            cm_df.to_csv(cm_csv_path)
            
            # Save history for this fold
            history_df = pd.DataFrame(fold_metrics)
            history_df.to_csv(f"{run_metrics_path}fold_{fold+1}_history.csv", index=False)
        
        # Aggregate results across all folds
        print("\n" + "="*50)
        print("CROSS-VALIDATION RESULTS")
        print("="*50)
        
        avg_metrics = {
            'test_loss': np.mean([fold['test_loss'] for fold in fold_results]),
            'test_acc': np.mean([fold['test_acc'] for fold in fold_results]),
            'test_precision': np.mean([fold['test_precision'] for fold in fold_results]),
            'test_recall': np.mean([fold['test_recall'] for fold in fold_results]),
            'test_f1': np.mean([fold['test_f1'] for fold in fold_results])
        }
        
        std_metrics = {
            'test_loss': np.std([fold['test_loss'] for fold in fold_results]),
            'test_acc': np.std([fold['test_acc'] for fold in fold_results]),
            'test_precision': np.std([fold['test_precision'] for fold in fold_results]),
            'test_recall': np.std([fold['test_recall'] for fold in fold_results]),
            'test_f1': np.std([fold['test_f1'] for fold in fold_results])
        }
        
        print("Average metrics across all folds:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
        
        # Save cross-validation results
        cv_results_df = pd.DataFrame([{
            'fold': fold['fold'],
            'test_loss': fold['test_loss'],
            'test_acc': fold['test_acc'],
            'test_precision': fold['test_precision'],
            'test_recall': fold['test_recall'],
            'test_f1': fold['test_f1']
        } for fold in fold_results])
        
        cv_results_df.to_csv(f"{outputResults_path}cv_results_{timestamp}.csv", index=False)
        
        # Add summary row with averages
        cv_summary = pd.DataFrame([{
            'fold': 'Average',
            'test_loss': f"{avg_metrics['test_loss']:.4f} ± {std_metrics['test_loss']:.4f}",
            'test_acc': f"{avg_metrics['test_acc']:.4f} ± {std_metrics['test_acc']:.4f}",
            'test_precision': f"{avg_metrics['test_precision']:.4f} ± {std_metrics['test_precision']:.4f}",
            'test_recall': f"{avg_metrics['test_recall']:.4f} ± {std_metrics['test_recall']:.4f}",
            'test_f1': f"{avg_metrics['test_f1']:.4f} ± {std_metrics['test_f1']:.4f}"
        }])
        
        cv_results_with_summary = pd.concat([cv_results_df, cv_summary])
        cv_results_with_summary.to_csv(f"{outputResults_path}cv_summary_{timestamp}.csv", index=False)
        
        # Create a final model using all data
        print("\nTraining final model on all data...")
        final_model_path = f"{outputModel_path}final_model_{timestamp}.pt"
        train_final_model(
            full_dataset, numeric_dim, categorical_info, fc_dim, dropout,
            lr, weight_decay, batch_size, device, final_model_path, seed
        )
        
    else:
        # -------------------------------
        # 4b. Standard Train/Val/Test Split
        # -------------------------------
        # Define split ratios (70% train, 15% validation, 15% test)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        # Split the dataset with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], 
            generator=generator
        )

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        # Create DataLoaders with worker seed initialization
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

        # -------------------------------
        # 5. Instantiate the model, loss and optimizer
        # -------------------------------
        model = NetworkTrafficCNN(
            numeric_dim=numeric_dim,
            categorical_info=categorical_info,
            conv_channels=64,
            kernel_size=5,
            fc_dim=fc_dim,
            dropout=dropout
        )
        model = model.to(device)

        # Binary cross entropy loss is more appropriate for binary classification
        criterion = nn.CrossEntropyLoss()
        # Using AdamW optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Model save path
        model_save_path = f"{outputModel_path}network_classifier_{timestamp}.pt"
        
        # Initialize early stopping with improved handler
        early_stopping = EarlyStopping(
            patience=7, 
            verbose=True, 
            path=model_save_path,
            trace_func=print
        )

        # -------------------------------
        # 6. Training and Validation Loop with Early Stopping
        # -------------------------------
        print("\nStarting training...")
        train_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, early_stopping, num_epochs=30
        )
        
        # Save training history
        history_df = pd.DataFrame(train_metrics)
        history_csv_path = f"{outputResults_path}training_history_{timestamp}.csv"
        history_df.to_csv(history_csv_path, index=False)
        print(f"Training history saved to {history_csv_path}")

        # -------------------------------
        # 7. Evaluate on test set
        # -------------------------------
        # Load the best model
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        print("\nTest Set Evaluation:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")

        # Create confusion matrix and save as CSV
        cm = confusion_matrix(test_metrics['targets'], test_metrics['predictions'])
        cm_df = pd.DataFrame(cm, 
                            index=['Actual Benign', 'Actual Malicious'],
                            columns=['Predicted Benign', 'Predicted Malicious'])
        cm_csv_path = f"{outputResults_path}confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"Confusion matrix saved to {cm_csv_path}")

        # Print confusion matrix in console
        print("\nConfusion Matrix:")
        print(cm_df)

        # -------------------------------
        # 8. Save model summary and metrics to a file
        # -------------------------------
        save_model_summary(
            outputResults_path, timestamp, checkpoint, numeric_dim, 
            categorical_columns, categorical_info, batch_size, 
            optimizer, test_metrics, cm
        )

        # -------------------------------
        # 9. Example inference on a sample
        # -------------------------------
        run_example_inference(model, test_dataset, device)

        # -------------------------------
        # 10. Create a tabular visualization of results using pandas
        # -------------------------------
        create_best_epochs_summary(train_metrics, outputResults_path, timestamp)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, early_stopping, num_epochs=30, fold=None):
    """
    Train the model with early stopping and validation.
    Returns training metrics history.
    """
    # For tracking metrics
    train_metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    fold_text = f" (Fold {fold})" if fold else ""
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy'] 
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
        
        # Save metrics
        train_metrics['epoch'].append(epoch + 1)
        train_metrics['train_loss'].append(train_loss)
        train_metrics['train_acc'].append(train_acc)
        train_metrics['val_loss'].append(val_loss)
        train_metrics['val_acc'].append(val_acc)
        train_metrics['val_precision'].append(val_precision)
        train_metrics['val_recall'].append(val_recall)
        train_metrics['val_f1'].append(val_f1)
        
        # Print stats
        print(f"Epoch {epoch+1}/{num_epochs}{fold_text}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss, model, epoch, optimizer, {
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
        # Print if there was an improvement
        if early_stopping.improvement:
            print(f"  New best model saved!")
    
    print("Training complete!")
    return train_metrics


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for numeric_features, cat_features, y in loader:
        # Move data to the appropriate device
        numeric_features = numeric_features.to(device)
        cat_features = {col: val.to(device) for col, val in cat_features.items()}
        y = y.to(device)
        
        # Forward pass
        outputs = model(numeric_features, cat_features)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * numeric_features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validate the model and return metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for numeric_features, cat_features, y in loader:
            # Move data to the appropriate device
            numeric_features = numeric_features.to(device)
            cat_features = {col: val.to(device) for col, val in cat_features.items()}
            y = y.to(device)
            
            # Forward pass
            outputs = model(numeric_features, cat_features)
            loss = criterion(outputs, y)
            
            # Statistics
            running_loss += loss.item() * numeric_features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_precision = precision_score(all_targets, all_preds, average='binary')
    val_recall = recall_score(all_targets, all_preds, average='binary')
    val_f1 = f1_score(all_targets, all_preds, average='binary')
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'predictions': all_preds,
        'targets': all_targets
    }


def evaluate_model(model, loader, criterion, device):
    """
    Evaluate the model on a dataset and return metrics.
    """
    return validate(model, loader, criterion, device)


def save_model_summary(output_path, timestamp, checkpoint, numeric_dim, categorical_columns, 
                      categorical_info, batch_size, optimizer, test_metrics, cm):
    """
    Save a comprehensive model summary to file.
    """
    with open(f"{output_path}model_summary_{timestamp}.txt", 'w') as f:
        f.write("Network Traffic Binary Classifier Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Architecture:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Numeric features: {numeric_dim}\n")
        f.write(f"Categorical features: {len(categorical_columns)}\n")
        f.write("Embedding dimensions:\n")
        for col, (vocab_size, embed_dim) in categorical_info.items():
            f.write(f"  {col}: {vocab_size} → {embed_dim}\n")
        
        f.write("\nTraining Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}\n")
        f.write(f"Best epoch: {checkpoint['epoch'] + 1}\n")
        
        f.write("\nValidation Metrics (Best Model):\n")
        f.write("-" * 20 + "\n")
        for metric, value in checkpoint['val_metrics'].items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        f.write("\nTest Set Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 20 + "\n")
        f.write("                 Predicted\n")
        f.write("               Benign  Malicious\n")
        f.write(f"Actual Benign    {cm[0,0]}      {cm[0,1]}\n")
        f.write(f"      Malicious  {cm[1,0]}      {cm[1,1]}\n")

    print(f"Model summary saved to {output_path}model_summary_{timestamp}.txt")


def run_example_inference(model, test_dataset, device):
    """
    Run inference on a single example and print results.
    """
    print("\nExample prediction on a single sample:")
    model.eval()
    with torch.no_grad():
        sample_numeric, sample_cat, sample_target = test_dataset[0]
        sample_numeric = sample_numeric.unsqueeze(0).to(device)
        sample_cat = {col: sample_cat[col].unsqueeze(0).to(device) for col in sample_cat}
        
        logits = model(sample_numeric, sample_cat)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1)
        
        print(f"Prediction: {'Malicious' if prediction.item() == 1 else 'Benign'} (class {prediction.item()})")
        print(f"Confidence: {probs[0][prediction.item()].item():.4f}")
        print(f"Actual label: {'Malicious' if sample_target.item() == 1 else 'Benign'} (class {sample_target.item()})")


def train_final_model(full_dataset, numeric_dim, categorical_info, fc_dim, dropout, 
                     lr, weight_decay, batch_size, device, model_save_path, seed):
    """
    Train a final model on the entire dataset after cross-validation.
    """
    # Create DataLoader for the full dataset
    full_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    # Create the model
    model = NetworkTrafficCNN(
        numeric_dim=numeric_dim,
        categorical_info=categorical_info,
        conv_channels=64,
        kernel_size=5,
        fc_dim=fc_dim,
        dropout=dropout
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train for a fixed number of epochs (using the average best epoch from CV)
    num_epochs = 15  # This could be derived from cross-validation results
    
    print(f"Training final model on full dataset for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, full_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
    
    # Save the final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
    }, model_save_path)
    
    print(f"Final model saved to {model_save_path}")
    
    # Run example inference on a sample from the dataset
    run_example_inference(model, full_dataset, device)


def create_best_epochs_summary(train_metrics, output_path, timestamp):
    """
    Create a summary of the best epochs based on validation metrics.
    """
    # For visualization of important metrics through epochs
    best_epochs = np.argsort(train_metrics['val_f1'])[-5:]  # Get indices of 5 best epochs by F1 score
    best_epochs_df = pd.DataFrame({
        'Epoch': [train_metrics['epoch'][i] for i in best_epochs],
        'Val Loss': [f"{train_metrics['val_loss'][i]:.4f}" for i in best_epochs],
        'Val Accuracy': [f"{train_metrics['val_acc'][i]:.4f}" for i in best_epochs],
        'Val Precision': [f"{train_metrics['val_precision'][i]:.4f}" for i in best_epochs],
        'Val Recall': [f"{train_metrics['val_recall'][i]:.4f}" for i in best_epochs],
        'Val F1': [f"{train_metrics['val_f1'][i]:.4f}" for i in best_epochs],
    })

    print("\nTop 5 Best Epochs by F1 Score:")
    print(best_epochs_df.to_string(index=False))

    # Save this table
    best_epochs_csv_path = f"{output_path}best_epochs_{timestamp}.csv"
    best_epochs_df.to_csv(best_epochs_csv_path, index=False)
    print(f"Best epochs summary saved to {best_epochs_csv_path}")