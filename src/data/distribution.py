import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("resources/datasets/NF-UNSW-NB15-v3.csv")

# Controllo presenza colonne
assert 'Label' in df.columns, "'Label' non trovata"

# Imposta dimensione figura
plt.figure(figsize=(16, 4))

# Plot: punti (scatter)
plt.scatter(df.index, df['Label'], c=df['Label'], cmap='coolwarm', s=3, alpha=0.6)

# Calcolo degli split
total_len = len(df)
train_end = int(total_len * 0.8)
val_end = int(total_len * 0.9)

# Linee verticali per mostrare i cut 80-10-10
plt.axvline(train_end, color='green', linestyle='--', linewidth=1.5, label='Fine Training (80%)')
plt.axvline(val_end, color='orange', linestyle='--', linewidth=1.5, label='Fine Validation (90%)')

# Etichette
plt.title("Distribuzione temporale degli attacchi (Label: 0=benigno, 1=attacco)")
plt.xlabel("Indice (riga del dataset)")
plt.ylabel("Label")
plt.yticks([0, 1], labels=['Benigno (0)', 'Attacco (1)'])
plt.xticks(ticks=range(0, total_len, total_len // 10))  # ogni 10%
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
