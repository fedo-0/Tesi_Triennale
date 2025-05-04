import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NetworkTrafficCNN(nn.Module):
    def __init__(self, numeric_dim, categorical_info, conv_channels=64, kernel_size=5, fc_dim=128, dropout=0.3):
        """
        Enhanced CNN model for network traffic classification.
        
        Args:
            numeric_dim: number of numeric features
            categorical_info: dictionary with categorical feature name as key and tuple (vocab_size, embed_dim) as value
            conv_channels: number of channels for 1D convolution layers
            kernel_size: size of the convolutional filter
            fc_dim: dimension of the fully connected intermediate layer
        """
        super(NetworkTrafficCNN, self).__init__()

        self.categorical_info = categorical_info
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        
        # Create embeddings for each categorical feature
        for col, (vocab_size, embed_dim) in categorical_info.items():
            self.embeddings[col] = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            total_embed_dim += embed_dim

        # Total input dimension: numeric features + sum of embedding dimensions
        self.total_input_dim = numeric_dim + total_embed_dim
        
        # Batch normalization for input features to stabilize training
        self.input_bn = nn.BatchNorm1d(self.total_input_dim)
        
        # Using multiple convolutional layers with increasing depth
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels*2, kernel_size=kernel_size-2, padding=(kernel_size-2) // 2)
        self.bn2 = nn.BatchNorm1d(conv_channels*2)
        
        # Global pooling to reduce spatial dimension
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers with dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels*2, fc_dim)
        self.bn3 = nn.BatchNorm1d(fc_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, fc_dim // 2)
        self.bn4 = nn.BatchNorm1d(fc_dim // 2)
        self.fc3 = nn.Linear(fc_dim // 2, 2)  # Binary classification: 2 classes

    def forward(self, numeric_features, categorical_features):
        """
        Forward pass of the model.
        
        Args:
            numeric_features: Tensor of shape [batch_size, numeric_dim]
            categorical_features: Dictionary {col_name: tensor [batch_size]}
            
        Returns:
            logits: Tensor of shape [batch_size, 2] for binary classification
        """
        # Process categorical features through embeddings
        embed_list = []
        for col in self.categorical_info.keys():
            embed = self.embeddings[col](categorical_features[col])  # [batch_size, embed_dim]
            embed_list.append(embed)
        
        # Concatenate all embedded features with numeric features
        if embed_list:
            cat_features = torch.cat(embed_list, dim=1)  # [batch_size, total_embed_dim]
            x = torch.cat([numeric_features, cat_features], dim=1)  # [batch_size, total_input_dim]
        else:
            x = numeric_features
        
        # Verifica dinamica delle dimensioni
        if x.size(1) != self.total_input_dim:
            raise ValueError(f"Input dimension mismatch: got {x.size(1)}, expected {self.total_input_dim}")
                
        # Apply batch normalization to input
        x = self.input_bn(x)
        
        # Add channel dimension: shape becomes [batch_size, 1, total_input_dim]
        x = x.unsqueeze(1)
        
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global pooling
        x = self.pool(x)  # output: [batch_size, conv_channels*2, 1]
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, conv_channels*2]
        
        # Fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Output layer
        logits = self.fc3(x)
        return logits
    
    def predict_proba(self, numeric_features, categorical_features):
        """
        Returns probability estimates for samples.
        
        Args:
            numeric_features: Tensor of shape [batch_size, numeric_dim]
            categorical_features: Dictionary {col_name: tensor [batch_size]}
            
        Returns:
            probabilities: Tensor of shape [batch_size, 2] containing class probabilities
        """
        logits = self.forward(numeric_features, categorical_features)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

# Test standalone
if __name__ == "__main__":
    # Example usage
    numeric_dim = 38  # Make sure this matches the number of numeric features in your dataset
    categorical_info = {
        f"cat_{i}": (np.random.randint(5, 1000), 4) for i in range(10)
    }
    model = NetworkTrafficCNN(numeric_dim=numeric_dim, categorical_info=categorical_info)
    
    # Print model architecture
    print(model)
    
    # Test with dummy data
    batch_size = 16
    dummy_numeric = torch.rand((batch_size, numeric_dim))
    dummy_cat = {col: torch.randint(0, info[0], (batch_size,)) for col, info in categorical_info.items()}
    
    # Forward pass
    outputs = model(dummy_numeric, dummy_cat)
    print("Output shape:", outputs.shape)  # Should be [batch_size, 2]
    
    # Test probability output
    probs = model.predict_proba(dummy_numeric, dummy_cat)
    print("Probability shape:", probs.shape)  # Should be [batch_size, 2]
    print("Sum of probabilities:", probs.sum(dim=1))  # Should be all close to 1.0