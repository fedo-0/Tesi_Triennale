import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

from data.preprocessing import NetworkFlowDataset, load_dataset
from model.binary_classificator import NetworkTrafficCNN

import logging
logger = logging.getLogger(__name__)

def run_training_pipeline (config_path, csv_path, outputModel_path, outputResults_path):
    logger.info("Config: %s, CSV: %s, Model: %s, Results: %s",
            config_path, csv_path, outputModel_path, outputResults_path)

    os.makedirs(outputModel_path, exist_ok=True)
    os.makedirs(outputResults_path, exist_ok=True)

    # -------------------------------
    # 1. Load configuration from JSON and dataset from CSV
    # -------------------------------
    df, numeric_columns, categorical_columns, target_column = load_dataset(config_path, csv_path)

    # Print for verification
    #print("Numeric columns:", len(numeric_columns))
    #print("Categorical columns:", len(categorical_columns))
    #print("Target column:", target_column)
    #print("Dataset shape:", df.shape)

    # Check class balance
    class_counts = df[target_column].value_counts()
    print(f"Class distribution: {class_counts}")
    print(f"Class proportions: {class_counts / len(df)}")

    # -------------------------------
    # 2. Split the dataset into train, validation, and test sets
    # -------------------------------
    # Create the full dataset
    full_dataset = NetworkFlowDataset(df, numeric_columns, categorical_columns, target_column)

    # Define split ratios (70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create DataLoaders
    batch_size = 64  # Increased batch size for better training stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # -------------------------------
    # 3. Define parameters for the model
    # -------------------------------
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

    # -------------------------------
    # 4. Instantiate the model, loss and optimizer
    # -------------------------------
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Improved architecture with multiple convolutional layers
    model = NetworkTrafficCNN(
        #numeric_dim=numeric_dim, 
        numeric_dim=len(numeric_columns),
        categorical_info=categorical_info,
        conv_channels=64,  # Increased number of channels
        kernel_size=5,     # Larger kernel for better feature capture
        fc_dim=128         # Larger fully connected layer
    )
    model = model.to(device)

    # Binary cross entropy loss is more appropriate for binary classification
    criterion = nn.CrossEntropyLoss()
    # Using AdamW optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # -------------------------------
    # 5. Training and Validation Loop with Early Stopping
    # -------------------------------
    def train_epoch(model, loader, criterion, optimizer, device):
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
        
        return val_loss, val_acc, val_precision, val_recall, val_f1, all_preds, all_targets

    # Training configuration
    num_epochs = 30
    early_stopping_patience = 7
    best_val_loss = float('inf')
    early_stopping_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"{outputModel_path}network_classifier_{timestamp}.pt"

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

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
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
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': {
                    'accuracy': val_acc,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }
            }, model_save_path)
            print(f"  Model saved to {model_save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    print("Training complete!")

    # -------------------------------
    # 6. Save training history using pandas instead of matplotlib
    # -------------------------------
    # Create DataFrame from training metrics
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

    test_loss, test_acc, test_precision, test_recall, test_f1, all_preds, all_targets = validate(model, test_loader, criterion, device)

    print("\nTest Set Evaluation:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

    # Create confusion matrix and save as CSV instead of plotting
    cm = confusion_matrix(all_targets, all_preds)
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
    with open(f"{outputResults_path}model_summary_{timestamp}.txt", 'w') as f:
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
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 20 + "\n")
        f.write("                 Predicted\n")
        f.write("               Benign  Malicious\n")
        f.write(f"Actual Benign    {cm[0,0]}      {cm[0,1]}\n")
        f.write(f"      Malicious  {cm[1,0]}      {cm[1,1]}\n")

    print(f"Model summary saved to {outputResults_path}model_summary_{timestamp}.txt")
    print(f"Evaluation complete!")

    # -------------------------------
    # 9. Example inference on a sample
    # -------------------------------
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

    # -------------------------------
    # 10. Create a tabular visualization of results using pandas
    # -------------------------------
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
    best_epochs_csv_path = f"{outputResults_path}best_epochs_{timestamp}.csv"
    best_epochs_df.to_csv(best_epochs_csv_path, index=False)
    print(f"Best epochs summary saved to {best_epochs_csv_path}")

if __name__ == "__main__":
    run_training_pipeline(config_path = "config/dataset.json", csv_path = "resources/datasets/NF-UNSW-NB15-v3.csv",
                          outputModel_path="output/models/", outputResults_path="output/results/")