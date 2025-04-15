import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json

# Importa la classe Dataset dal file preprocessing.py
from src.data.preprocessing import NetworkFlowDataset, load_dataset

# Importa il modello dal file binary_classificator.py
from src.model.binary_classificator import NetworkTrafficCNN

# -------------------------------
# 1. Carica la configurazione dal file JSON e il dataset dal CSV
# -------------------------------
config_path = "config/dataset.json"  # Percorso del file JSON di configurazione
csv_path = "resources/datasets/NF-UNSW-NB15-v3.csv"  # Percorso del file CSV

df, numeric_columns, categorical_columns, target_column = load_dataset(config_path, csv_path)

# Stampa per verifica
print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)
print("Target column:", target_column)
print("Dataset shape:", df.shape)

# -------------------------------
# 2. Crea il dataset e il DataLoader
# -------------------------------
dataset = NetworkFlowDataset(df, numeric_columns, categorical_columns, target_column)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# 3. Definisci parametri per il modello
# -------------------------------
numeric_dim = len(numeric_columns)
# Definisci i parametri per ogni feature categoriale; qui i valori (vocab_size, embed_dim) devono essere
# impostati in base alle caratteristiche del dataset. I valori esemplificativi sono quelli seguenti:
categorical_info = {
    col: (dataset.cat_dims[col], 8 if dataset.cat_dims[col] > 100 else 4)
    for col in categorical_columns
}

# -------------------------------
# 4. Istanzia il modello, loss e ottimizzatore
# -------------------------------
model = NetworkTrafficCNN(numeric_dim=numeric_dim, categorical_info=categorical_info)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 5. Training Loop
# -------------------------------
# Training Loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for numeric_features, cat_features, y in dataloader:
        # Controlla NaN nei dati numerici
        if torch.isnan(numeric_features).any():
            print("NaN nei numeric_features")

        # Controlla NaN nelle features categoriali
        for col, val in cat_features.items():
            if torch.isnan(val.float()).any():  # cast a float per sicurezza
                print(f"NaN nei cat_features[{col}]")

        # Forward pass
        outputs = model(numeric_features, cat_features)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()

        # Verifica se ci sono NaN nei gradienti
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN nel gradiente di: {name}")

        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# -------------------------------
# 6. Esempio di Inferenza
# -------------------------------
model.eval()
with torch.no_grad():
    sample_numeric, sample_cat, sample_target = dataset[0]
    sample_numeric = sample_numeric.unsqueeze(0)
    # Assicurati di aggiungere la dimensione batch per le feature categoriali
    sample_cat = {col: sample_cat[col].unsqueeze(0) for col in sample_cat}
    logits = model(sample_numeric, sample_cat)
    prediction = torch.argmax(F.softmax(logits, dim=1), dim=1)
    print("Predizione:", prediction.item(), "Target:", sample_target)
