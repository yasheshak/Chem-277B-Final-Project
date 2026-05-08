import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from fairchem.core.datasets import AseDBDataset
from torch.nn import Linear
from torch_geometric.nn import SchNet
from torch_geometric.data import Data
from torch_cluster import radius_graph
import matplotlib.pyplot as plt
from read_multi_ase import *
from extract_ab import *


class EmbeddedSchNet(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: int = 12,
                 max_num_neighbors: int = 32,
                 readout: str = "add",
                 dipole: bool = False,
                 mean: float = None,
                 std: float = None,
                 atomref: torch.Tensor = None,
                 extra_feat_dim: int = 1,
                 train_mean: float = None,
                 train_std: float = None):
        super().__init__()

        #Create SchNet module with parameters
        self.schnet = SchNet(
            hidden_channels = hidden_channels,
            num_filters = num_filters,
            num_interactions = num_interactions,
            num_gaussians = num_gaussians,
            cutoff = cutoff,
            max_num_neighbors = max_num_neighbors,
            readout = readout,
            dipole = dipole,
            mean = mean,
            std = std,
            atomref = atomref)


        #Adjusted linear layer for two homo lumo gaps
        self.schnet.lin2 = Linear(self.schnet.hidden_channels // 2, 2)

        #Extra Linear layer to handle extra embedded features
        self.linear = Linear(extra_feat_dim, hidden_channels)

        #Keep track of training mean and std for normalization function
        self.mean = train_mean
        self.std = train_std

    def forward(self, data):
        #Extract each value from data
        atomic_num, positions, batch, extra_feat = data.z, data.pos, data.batch, data.extra_feat

        #Initialize atom embeddings through SchNet
        atom_embeddings = self.schnet.embedding(atomic_num)

        # print(extra_feat.shape)

        #Project extra features on linear layer
        extra_linear = self.linear(extra_feat)

        #Combine extra features and initialized atom embeddings
        atom_embeddings = atom_embeddings + extra_linear

        #Use SchNet utilities to extract edge indexes and edge weighs based on position and batch
        edge_index, edge_weight = self.schnet.interaction_graph(positions, batch)
        #Use SchNet utilities to obtain edge attribute based on edge weights
        edge_attr = self.schnet.distance_expansion(edge_weight)

        #Iterate through each interaction block manually to simulate Schnet
        for interaction_block in self.schnet.interactions:
            atom_embeddings = atom_embeddings + interaction_block(atom_embeddings, edge_index, edge_weight, edge_attr)

        atom_embeddings = self.schnet.lin1(atom_embeddings)

        atom_embeddings = self.schnet.act(atom_embeddings)

        atom_embeddings = self.schnet.lin2(atom_embeddings)

        #Ensure the readout is based on model's hyperparameter
        output = self.schnet.readout(atom_embeddings, batch)

        return output


def train(model: EmbeddedSchNet, train_data: list):
    model.train()
    #Keep track of total loss for all data
    total_train_loss = 0

    for data in train_data:
        data = data.to(device)

        #Reset optimizers
        optimizer.zero_grad()

        #Forward step internally performed by PyTorch to obtain predictions (Same as model.forward(data))
        y_pred = model(data)

        y_target = data.y.view(-1, 2)

        y_mask = data.y_mask.view(-1, 2)

        #Determine train loss based on loss function with predictions and targets
        train_loss = loss_function(y_pred, y_target)
        train_loss = (train_loss * y_mask).sum() / y_mask.sum()

        #Backward step to calculate gradients
        train_loss.backward()
        #Updated optimizers
        optimizer.step()

        #Add train loss of current data to total train loss
        total_train_loss += train_loss.item()

    return total_train_loss / len(train_data)

@torch.no_grad()
def evaluate(model: EmbeddedSchNet, val_data: list):
    model.eval()
    total_val_loss = 0

    for data in val_data:
        data = data.to(device)

        #Forward step internally performed by PyTorch to obtain predictions (Same as model.forward(data))
        y_pred = model(data)

        y_target = data.y.view(-1, 2)

        y_mask = data.y_mask.view(-1, 2)

        #Determine train loss based on loss function with predictions and targets
        val_loss = loss_function(y_pred, y_target)
        val_loss = (val_loss * y_mask).sum() / y_mask.sum()

        #Add validation loss of current data to total validation loss
        total_val_loss += val_loss.item()

    return total_val_loss / len(val_data)

@torch.no_grad()
def test(model: EmbeddedSchNet, test_data: list):
    model.eval()

    total_mae = 0
    total_mse = 0
    n_molecules = 0

    for data in test_data:
        data = data.to(device)

        y_pred = model(data)
        y_target = data.y.view(-1, 2)

        y_mask = data.y_mask.view(-1, 2)


        y_pred = (y_pred * model.std + model.mean) * y_mask
        y_target = (y_target * model.std + model.mean) * y_mask


        mae = torch.abs(y_pred - y_target).sum()
        mse = ((y_pred - y_target) ** 2).sum()

        total_mae += mae.item()
        total_mse += mse.item()
        n_molecules += y_mask.sum()

    mean_mae = total_mae / n_molecules
    rmse = (total_mse / n_molecules) ** 0.5

    return mean_mae, rmse


def plot_losses(train_loss, val_loss):
    """Plot training vs validation loss from a history dict."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss',
             linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SchNet Model Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#Determine the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Initialize model with desired parameters
bio_model = EmbeddedSchNet(hidden_channels = 128, num_filters = 256, cutoff = 8, num_gaussians = 60, num_interactions = 6, max_num_neighbors = 40, readout = "mean", extra_feat_dim = 2).to(device)
#Create AdamW optimizer based on model's parameters and desired learning rate and weight decay
optimizer = torch.optim.AdamW(bio_model.parameters(), lr=5e-5, weight_decay=1e-4)
#Select loss function for model
loss_function = torch.nn.SmoothL1Loss(reduction = "none")

#Load data and specified features with helper functions
bio_sample = process_file(file="./train_4M/data0000.aselmdb", molecule_type= "biomolecules", max_molecules = 20000)
bio_data = get_data(bio_sample, ["lowdin_charges"])
bio_train, bio_val, bio_test = split_data(bio_data, 0.2, 0.2)

#Scale features of all sets with training data fit scaler
train_scaler = feature_scaler(bio_train)
bio_train = scale_features(bio_train, train_scaler)
bio_val = scale_features(bio_val, train_scaler)
bio_test = scale_features(bio_test, train_scaler)

#Obtain training data mean and std to normalize all of data
bio_model.mean, bio_model.std = obtain_mean_std(bio_train)
bio_train = normalize_target(bio_train, bio_model.mean, bio_model.std)
bio_val = normalize_target(bio_val, bio_model.mean, bio_model.std)
bio_test = normalize_target(bio_test, bio_model.mean, bio_model.std)

#Finish loading the data with specific batch size
bio_train_loader = DataLoader(bio_train, batch_size=32, shuffle=True)
bio_val_loader = DataLoader(bio_val, batch_size=32)
bio_test_loader = DataLoader(bio_test, batch_size=32)

#Set training epochs and initialize arrays to store training and validation losses
epochs = 50
bio_train_losses = np.zeros(epochs)
bio_val_losses = np.zeros(epochs)

for epoch in range(epochs):
    train_loss = train(bio_model, bio_train_loader)
    val_loss = evaluate(bio_model, bio_val_loader)

    bio_train_losses[epoch] = train_loss
    bio_val_losses[epoch] = val_loss
    if epoch % 5 == 0:
        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#Plot training and validation losses vs epochs
plot_losses(bio_train_losses, bio_val_losses)

#Run model with test set to determine performance values
mae, rmse = test(bio_model, bio_test_loader)

print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")