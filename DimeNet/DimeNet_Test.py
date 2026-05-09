import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import DimeNetPlusPlus, radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, softmax
from torch_geometric.typing import SparseTensor
from read_multi_ase import *
from extract_ab import *
import pandas as pd
import argparse


def triplets(edge_index, num_nodes):
    row, col = edge_index
    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class DimeNetPP_EmbeddedAttention(nn.Module):
    """DimeNet++ with internal feature injection + attention readout + spin conditioning."""
    def __init__(self,
                 hidden_channels: int = 64,
                 out_channels: int = 2,
                 num_blocks: int = 4,
                 int_emb_size: int = 64,
                 basis_emb_size: int = 8,
                 out_emb_channels: int = 256,
                 num_spherical: int = 7,
                 num_radial: int = 6,
                 cutoff: float = 7.0,
                 extra_feat_dim: int = 2,
                 train_mean: float = None,
                 train_std: float = None):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = 32

        self.dimenet = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff)

        self.feat_proj = nn.Linear(extra_feat_dim, hidden_channels)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels))
        self.spin_proj = nn.Linear(1, out_channels)
        self.mean = train_mean
        self.std = train_std

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        extra_feat = data.extra_feat

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.linalg.cross(pos_ij, pos_jk).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.dimenet.rbf(dist)
        sbf = self.dimenet.sbf(dist, angle, idx_kj)

        x = self.dimenet.emb(z, rbf, i, j)

        feat_node = self.feat_proj(extra_feat)
        x = x + feat_node[i] + feat_node[j]

        P = self.dimenet.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(
            self.dimenet.interaction_blocks, self.dimenet.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        attn_scores = self.attn_gate(P).squeeze(-1)
        attn_weights = softmax(attn_scores, batch)
        weighted = P * attn_weights.unsqueeze(-1)
        mol_repr = scatter(weighted, batch, dim=0, reduce='sum')

        out = self.output_proj(mol_repr)
        spin = data.spin.unsqueeze(-1).float()
        out = out + self.spin_proj(spin)
        return out


@torch.no_grad()
def test(model, test_data):
    model.eval()
    total_mae = 0
    total_mse = 0
    n_molecules = 0
    mean = model.mean.to(device)
    std = model.std.to(device)

    for data in test_data:
        data = data.to(device)

        y_pred = model(data)
        y_target = data.y.view(-1, 2)
        y_mask = data.y_mask.view(-1, 2)

        #Denormalize predictions and targets back to eV
        y_pred = (y_pred * std + mean) * y_mask
        y_target = (y_target * std + mean) * y_mask

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
    parser = argparse.ArgumentParser(description="Test Pretrained DimeNet++ Model with specified number of molecules")
    parser.add_argument("--test_size", type=int, default=500, help="Number of molecules to test with")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #Load saved model
    saved_model = torch.load("./DimeNet/DimeNet_Final_Weights.pt", map_location=device, weights_only=False)
    config = saved_model["config"]

    model = DimeNetPP_EmbeddedAttention(
        hidden_channels=config["hidden_channels"],
        num_blocks=config["num_blocks"],
        int_emb_size=config["int_emb_size"],
        basis_emb_size=config["basis_emb_size"],
        out_emb_channels=config["out_emb_channels"],
        num_spherical=config["num_spherical"],
        num_radial=config["num_radial"],
        cutoff=config["cutoff"],
        extra_feat_dim=config["extra_feat_dim"]
    )

    model.mean = saved_model["norm"]["mean"]
    model.std = saved_model["norm"]["std"]
    model.load_state_dict(saved_model["model_weights"])
    model.to(device)
    model.eval()

    #Load test data from separate file
    test_file = "./OMol25_data/data0002.aselmdb"
    test_sample = process_file(file=test_file, molecule_type="biomolecules", max_molecules=args.test_size)
    test_data = get_data(test_sample)
    test_data = normalize_target(test_data, model.mean, model.std)
    test_loader = DataLoader(test_data, batch_size=32)

    #Run predictions
    all_pred = []
    all_true = []

    mae, rmse = test(model, test_loader)

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    #Save predictions to file
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    df = pd.DataFrame({
        "Mol": np.arange(1, all_true.shape[0] + 1, 1),
        "Alpha Pred": all_pred[:, 0],
        "Alpha True": all_true[:, 0],
        "Beta Pred": all_pred[:, 1],
        "Beta True": all_true[:, 1]
    })

    markdown_table = df.to_markdown(index=False)
    with open("DimeNet/Final_DimeNet_Predictions.txt", "w") as f:
        f.write(markdown_table)

    print(f"Predictions saved to DimeNet/Final_DimeNet_Predictions.txt")
