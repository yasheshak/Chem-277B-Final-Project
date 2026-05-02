import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool, Sequential, AttentionalAggregation


class SimpleGNN(nn.Module):
    def __init__(self, num_node_features=1, hidden_channels=128):
        # call constructor from parent class 
        super().__init__()

        self.simpleGNN = Sequential('x, edge_index, batch', [
            # CONVOLUTION LAYERS: learn molecular represenation 
            # layer 1: directly bonded atoms
            (GCNConv(num_node_features, hidden_channels), 'x, edge_index -> x'),
            nn.ReLU(),
            # layer 2: local chemical environment (i.e., funct groups)
            (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
            nn.ReLU(),
            # layer 3: adds global context
            (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
            nn.ReLU(),

            # POOLING: down-sample/reduce dimensions 
            (global_mean_pool, 'x, batch -> x'),

            # LINEAR LAYERS: turn into actual predictions 
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)])


    def forward(self, x, edge_index, batch):
        return self.simpleGNN(x, edge_index, batch).reshape(-1)
    

# inherit from nn.Module 
class AttentionGCN(nn.Module):
    def __init__(self, num_node_features=1, hidden_channels=256):
        # call constructor from parent class 
        super().__init__()

        # define attention pooling -> map embedding/importance
        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)))

        self.attention = Sequential('x, edge_index, batch', [
            (GCNConv(num_node_features, 64), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(64, 128), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(128, 256), 'x, edge_index -> x'),
            nn.ReLU(),

            # POOLING: modified to attention pooling 
            (self.attention_pool, 'x, batch -> x'),

            # LINEAR LAYERS: turn into actual predictions 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)])


    def forward(self, x, edge_index, batch):
        return self.attention(x, edge_index, batch).reshape(-1)