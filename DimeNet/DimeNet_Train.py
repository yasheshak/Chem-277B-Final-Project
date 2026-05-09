import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import DimeNetPlusPlus, radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, softmax
from torch_geometric.typing import SparseTensor
from read_multi_ase import *
from extract_ab import *
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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        model.load_state_dict(self.shadow)


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

        #Create DimeNet++ module with parameters
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

        #Extra linear layer to project embedded features into edge embeddings
        self.feat_proj = nn.Linear(extra_feat_dim, hidden_channels)

        #Attention gate: learns per-atom importance for readout
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1))

        #Output projection from molecular representation to alpha/beta predictions
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels))

        #Spin conditioning layer
        self.spin_proj = nn.Linear(1, out_channels)

        #Keep track of training mean and std for normalization
        self.mean = train_mean
        self.std = train_std

    def forward(self, data):
        #Extract each value from data
        z, pos, batch = data.z, data.pos, data.batch
        extra_feat = data.extra_feat

        #Build radius graph for interatomic interactions
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        #Compute triplets for angular information
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        #Calculate distances and angles
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.linalg.cross(pos_ij, pos_jk).norm(dim=-1)
        angle = torch.atan2(b, a)

        #Compute radial and spherical basis functions
        rbf = self.dimenet.rbf(dist)
        sbf = self.dimenet.sbf(dist, angle, idx_kj)

        #Initialize edge embeddings through DimeNet++ embedding block
        x = self.dimenet.emb(z, rbf, i, j)

        # === INJECT FEATURES into edge embeddings before message passing ===
        feat_node = self.feat_proj(extra_feat)
        x = x + feat_node[i] + feat_node[j]

        #Run output blocks and interaction blocks (directional message passing)
        P = self.dimenet.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(
            self.dimenet.interaction_blocks, self.dimenet.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        # === ATTENTION READOUT: learn per-atom importance ===
        attn_scores = self.attn_gate(P).squeeze(-1)
        attn_weights = softmax(attn_scores, batch)
        weighted = P * attn_weights.unsqueeze(-1)
        mol_repr = scatter(weighted, batch, dim=0, reduce='sum')

        #Project molecular representation to output predictions
        out = self.output_proj(mol_repr)

        #Add spin conditioning
        spin = data.spin.unsqueeze(-1).float()
        out = out + self.spin_proj(spin)

        return out


def train(model, train_data):
    model.train()
    #Keep track of total loss for all data
    total_train_loss = 0
    loss_function = nn.SmoothL1Loss(reduction='none')

    for data in train_data:
        data = data.to(device)

        #Reset optimizers
        optimizer.zero_grad()

        #Forward step to obtain predictions
        y_pred = model(data)

        y_target = data.y.view(-1, 2)
        y_mask = data.y_mask.view(-1, 2)

        #Determine train loss with masked loss function
        train_loss = loss_function(y_pred, y_target)
        train_loss = (train_loss * y_mask).sum() / y_mask.sum()

        #Backward step to calculate gradients
        train_loss.backward()
        #Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        #Update optimizers
        optimizer.step()
        #Update EMA shadow weights
        ema.update(model)

        #Add train loss of current data to total train loss
        total_train_loss += train_loss.item()

    return total_train_loss / len(train_data)


@torch.no_grad()
def evaluate(model, val_data):
    model.eval()
    total_val_loss = 0
    loss_function = nn.SmoothL1Loss(reduction='none')

    for data in val_data:
        data = data.to(device)

        #Forward step to obtain predictions
        y_pred = model(data)

        y_target = data.y.view(-1, 2)
        y_mask = data.y_mask.view(-1, 2)

        #Determine validation loss with masked loss function
        val_loss = loss_function(y_pred, y_target)
        val_loss = (val_loss * y_mask).sum() / y_mask.sum()

        #Add validation loss of current data to total validation loss
        total_val_loss += val_loss.item()

    return total_val_loss / len(val_data)


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

        mae = torch.abs(y_pred - y_target).sum()
        mse = ((y_pred - y_target) ** 2).sum()

        total_mae += mae.item()
        total_mse += mse.item()
        n_molecules += y_mask.sum()

    mean_mae = total_mae / n_molecules
    rmse = (total_mse / n_molecules) ** 0.5

    return mean_mae, rmse


def plot_losses(train_loss, val_loss):
    """Plot training vs validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DimeNet++ Model Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedded DimeNet++ Model for HOMO-LUMO Gap Prediction")
    parser.add_argument("--data_path", nargs="+", default=["./OMol25_data/data0000.aselmdb"], help="Path(s) to .aselmdb file(s)")
    parser.add_argument("--num_molecules", type=int, default=500, help="Number of molecules to use")
    parser.add_argument("--molecule_type", type=str, default="biomolecules", help="Molecule type filter (or 'all')")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--int_emb_size", type=int, default=64)
    parser.add_argument("--basis_emb_size", type=int, default=8)
    parser.add_argument("--out_emb_channels", type=int, default=256)
    parser.add_argument("--num_spherical", type=int, default=7)
    parser.add_argument("--num_radial", type=int, default=6)
    parser.add_argument("--cutoff", type=float, default=7.0)
    parser.add_argument("--extra_feat_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--save_model", type=str, default=None, help="Path to save model weights")
    args = parser.parse_args()

    #Determine the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #Initialize model with desired parameters
    bio_model = DimeNetPP_EmbeddedAttention(
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size,
        basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        cutoff=args.cutoff,
        extra_feat_dim=args.extra_feat_dim).to(device)

    #Create AdamW optimizer based on model's parameters and desired learning rate and weight decay
    optimizer = torch.optim.AdamW(bio_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    #Create EMA for weight smoothing
    ema = EMA(bio_model, decay=0.999)

    #Load data and specified features with helper functions
    mol_type = None if args.molecule_type == "all" else args.molecule_type
    bio_sample = process_file(file=args.data_path, molecule_type=mol_type, max_molecules=args.num_molecules)
    bio_data = get_data(bio_sample)
    bio_train, bio_val, bio_test = split_data(bio_data)

    #Obtain training data mean and std to normalize all of data
    bio_model.mean, bio_model.std = obtain_mean_std(bio_train)
    bio_train = normalize_target(bio_train, bio_model.mean, bio_model.std)
    bio_val = normalize_target(bio_val, bio_model.mean, bio_model.std)
    bio_test = normalize_target(bio_test, bio_model.mean, bio_model.std)

    #Finish loading the data with specific batch size
    bio_train_loader = DataLoader(bio_train, batch_size=args.batch_size, shuffle=True)
    bio_val_loader = DataLoader(bio_val, batch_size=args.batch_size)
    bio_test_loader = DataLoader(bio_test, batch_size=args.batch_size)

    #Set training parameters
    warmup_epochs = 5
    best_val = float('inf')
    patience_counter = 0
    best_state = None
    bio_train_losses = []
    bio_val_losses = []

    for epoch in range(args.epochs):
        #Linear warmup for first 5 epochs
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_loss = train(bio_model, bio_train_loader)
        val_loss = evaluate(bio_model, bio_val_loader)

        #Step scheduler after warmup period
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        bio_train_losses.append(train_loss)
        bio_val_losses.append(val_loss)

        #Save best model state
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in bio_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.1e} | Patience: {patience_counter}/{args.patience}")

        #Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    #Select best weights: compare EMA vs best checkpoint
    ema.apply(bio_model)
    mae_ema, _ = test(bio_model, bio_val_loader)
    if best_state is not None:
        bio_model.load_state_dict(best_state)
        mae_best, _ = test(bio_model, bio_val_loader)
    else:
        mae_best = float('inf')

    if mae_ema < mae_best:
        ema.apply(bio_model)
        print(f"Using EMA weights (val MAE: {mae_ema:.4f})")
    else:
        print(f"Using best checkpoint (val MAE: {mae_best:.4f})")

    print(f"Best Val Loss: {best_val:.4f}")

    #Plot training and validation losses vs epochs
    plot_losses(bio_train_losses, bio_val_losses)

    #Run model with test set to determine performance values
    mae, rmse = test(bio_model, bio_test_loader)
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    #Save model weights and config
    if args.save_model:
        torch.save({
            "model_weights": bio_model.state_dict(),
            "config": {
                "hidden_channels": args.hidden_channels,
                "num_blocks": args.num_blocks,
                "int_emb_size": args.int_emb_size,
                "basis_emb_size": args.basis_emb_size,
                "out_emb_channels": args.out_emb_channels,
                "num_spherical": args.num_spherical,
                "num_radial": args.num_radial,
                "cutoff": args.cutoff,
                "extra_feat_dim": args.extra_feat_dim
            },
            "norm": {
                "mean": bio_model.mean,
                "std": bio_model.std
            }
        }, args.save_model)
        print(f"Model saved to {args.save_model}")
