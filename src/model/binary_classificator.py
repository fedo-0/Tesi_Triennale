import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkTrafficCNN(nn.Module):
    def __init__(self, numeric_dim, categorical_info, conv_channels=32, kernel_size=3, fc_dim=64):
        """
        numeric_dim: numero di feature numeriche (es. 38)
        categorical_info: dizionario con chiave il nome della feature categoriale e valore una tupla (vocab_size, embed_dim)
        conv_channels: numero di canali per il layer di convoluzione 1D.
        kernel_size: dimensione del filtro convolutivo.
        fc_dim: dimensione del layer fully connected intermedio.
        """
        super(NetworkTrafficCNN, self).__init__()

        self.categorical_info = categorical_info
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for col, (vocab_size, embed_dim) in categorical_info.items():
            self.embeddings[col] = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            total_embed_dim += embed_dim

        # Dimensione totale in input: feature numeriche + somma dei dimensioni degli embedding
        self.total_input_dim = numeric_dim + total_embed_dim

        # Layer convolutivo 1D: il modello tratterà l'input come [batch_size, 1, total_input_dim]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)  # riduce la dimensione spaziale a 1
        self.fc1 = nn.Linear(conv_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 2)  # output binario: due classi

    def forward(self, numeric_features, categorical_features):
        # numeric_features: [batch_size, numeric_dim]
        # categorical_features: dizionario {col_name: tensor [batch_size]}
        embed_list = []
        for col in self.categorical_info.keys():
            embed = self.embeddings[col](categorical_features[col])  # [batch_size, embed_dim]
            embed_list.append(embed)
        if embed_list:
            cat_features = torch.cat(embed_list, dim=1)  # [batch_size, total_embed_dim]
            x = torch.cat([numeric_features, cat_features], dim=1)  # [batch_size, total_input_dim]
        else:
            x = numeric_features

        # Aggiungiamo la dimensione canale: forma diventa [batch_size, 1, total_input_dim]
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # output: [batch_size, conv_channels, 1]
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, conv_channels]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# Test standalone
if __name__ == "__main__":
    # Esempio di utilizzo:
    numeric_dim = 38  # Assicurati che corrisponda al numero di feature numeriche nel tuo dataset
    categorical_info = {
        "L4_SRC_PORT": (65536, 8),
        "L4_DST_PORT": (65536, 8),
        "PROTOCOL": (10, 4),
        "L7_PROTO": (20, 4),
        "TCP_FLAGS": (64, 4),
        "CLIENT_TCP_FLAGS": (64, 4),
        "SERVER_TCP_FLAGS": (64, 4),
        "ICMP_TYPE": (10, 3),
        "ICMP_IPV4_TYPE": (10, 3),
        "DNS_QUERY_ID": (1000, 4),
        "DNS_QUERY_TYPE": (50, 3)
    }
    model = NetworkTrafficCNN(numeric_dim=numeric_dim, categorical_info=categorical_info)
    dummy_numeric = torch.rand((10, numeric_dim))
    dummy_cat = {col: torch.randint(0, info[0], (10,)) for col, info in categorical_info.items()}
    outputs = model(dummy_numeric, dummy_cat)
    print("Output shape:", outputs.shape)  # Deve essere [10, 2]
