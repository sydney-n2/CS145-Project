import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCNModel(nn.Module):
   def __init__(self, num_features, hidden_channels):
       super(GCNModel, self).__init__()
       self.conv1 = GCNConv(num_features, hidden_channels)
       self.bn1 = BatchNorm(hidden_channels)
       self.conv2 = GCNConv(hidden_channels, hidden_channels)
       self.bn2 = BatchNorm(hidden_channels)
       self.conv3 = GCNConv(hidden_channels, hidden_channels)  # Additional layer
       self.bn3 = BatchNorm(hidden_channels)
       self.fc1 = nn.Linear(hidden_channels, hidden_channels)  # More complex final layer
       self.fc2 = nn.Linear(hidden_channels, 1)               # Output layer remains the same


   def forward(self, x, edge_index):
       x = self.conv1(x, edge_index)
       x = self.bn1(x)
       x = F.relu(x)
       x = F.dropout(x, p=0.1, training=self.training)  # Adjusted dropout rate


       x = self.conv2(x, edge_index)
       x = self.bn2(x)
       x = F.relu(x)
       x = F.dropout(x, p=0.1, training=self.training)  # Consistent dropout rate


       x = self.conv3(x, edge_index)  # Additional convolution layer
       x = self.bn3(x)
       x = F.relu(x)
       x = F.dropout(x, p=0.1, training=self.training)


       x = self.fc1(x)  # Additional dense layer
       x = F.relu(x)
       x = self.fc2(x)


       return F.sigmoid(x)  # Ensure output is sigmoid for binary classification