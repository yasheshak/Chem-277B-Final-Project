import torch
from torch_geometric.loader import DataLoader
import numpy as np
from fairchem.core.datasets import AseDBDataset
from torch.nn import Linear
from torch_geometric.nn import SchNet
from torch_geometric.data import Data
from torch_cluster import radius_graph
from read_multi_ase import *
from extract_ab import *
import pandas as pd
import argparse

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

        all_pred.append(y_pred.cpu().numpy())
        all_true.append(y_target.cpu().numpy())

        mae = torch.abs(y_pred - y_target).sum()
        mse = ((y_pred - y_target) ** 2).sum()

        total_mae += mae.item()
        total_mse += mse.item()
        n_molecules += y_mask.sum()

    mean_mae = total_mae / n_molecules
    rmse = (total_mse / n_molecules) ** 0.5

    return mean_mae, rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Test Pretrained SchNet Model with specificied number of molecules")
    parser.add_argument("--test_size", type = int, default = 500, help = "Number of molecules to test with")
    args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

saved_model = torch.load("./SchNet/SchNet_Final_Weights.pt", map_location = device, weights_only = False)

config = saved_model["config"]

model = EmbeddedSchNet(
    hidden_channels=config["hidden_channels"],
    num_filters=config["num_filters"],
    num_gaussians=config["num_gaussians"],
    cutoff=config["cutoff"],
    num_interactions=config["num_interactions"],
    max_num_neighbors=config["max_num_neighbors"],
    readout=config["readout"],
    extra_feat_dim=config["extra_feat_dim"]
)

model.mean = saved_model["norm"]["mean"]
model.std = saved_model["norm"]["std"]

train_scaler = saved_model["scaler"]

model.load_state_dict(saved_model["model_weights"])
model.to(device)
model.eval()

test_file = "./OMol25_data/data0002.aselmdb"

test_sample = process_file(file = test_file, molecule_type = "biomolecules", max_molecules = args.test_size)
test_data = get_data(test_sample, ["lowdin_charges"])

test_data = scale_features(test_data, train_scaler)
test_data = normalize_target(test_data, model.mean, model.std)

test_loader = DataLoader(test_data, batch_size = 32)

all_pred = []
all_true = []


mae, rmse = test(model, test_loader)

print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

all_pred = np.concatenate(all_pred, axis = 0)
all_true = np.concatenate(all_true, axis = 0)

df = pd.DataFrame({
    "Mol": np.arange(1, all_true.shape[0] + 1, 1),
    "Alpha Pred": all_pred[:, 0],
    "Alpha True": all_true[:, 0],
    "Beta Pred": all_pred[:, 1],
    "Beta True": all_true[:, 1]
})

markdown_table = df.to_markdown(index=False)

with open("SchNet/Final_SchNet_Predictions.txt", "w") as f:
    f.write(markdown_table)