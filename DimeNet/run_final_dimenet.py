"""
DimeNet++ Embedded Internal + Attention — Alpha/Beta HOMO-LUMO Gap Prediction

Usage:
    python run_dimenet.py --data_path ./data0000.aselmdb --num_molecules 1000
    python run_dimenet.py --data_path ./data0000.aselmdb ./data0001.aselmdb --num_molecules 20000
    python run_dimenet.py --data_path ./data0000.aselmdb --num_molecules 100 --molecule_type all
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import DimeNetPlusPlus, radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, softmax
from torch_geometric.typing import SparseTensor
from read_multi_ase_att import *
from extract_ab_att_emb import *


# ============================================================
# Model
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        model.load_state_dict(self.shadow)


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
    def __init__(self, hidden_channels=64, out_channels=2, num_blocks=4,
                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, cutoff=7.0,
                 extra_feat_dim=2, train_mean=None, train_std=None):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = 32

        self.dimenet = DimeNetPlusPlus(
            hidden_channels=hidden_channels, out_channels=hidden_channels,
            num_blocks=num_blocks, int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels,
            num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff)

        self.feat_proj = nn.Linear(extra_feat_dim, hidden_channels)
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.Tanh(),
            nn.Linear(hidden_channels, 1))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
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

        # Internal feature injection
        feat_node = self.feat_proj(extra_feat)
        x = x + feat_node[i] + feat_node[j]

        P = self.dimenet.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(
            self.dimenet.interaction_blocks, self.dimenet.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        # Attention readout
        attn_scores = self.attn_gate(P).squeeze(-1)
        attn_weights = softmax(attn_scores, batch)
        weighted = P * attn_weights.unsqueeze(-1)
        mol_repr = scatter(weighted, batch, dim=0, reduce='sum')

        out = self.output_proj(mol_repr)
        spin = data.spin.unsqueeze(-1).float()
        out = out + self.spin_proj(spin)
        return out


# ============================================================
# Training / Evaluation
# ============================================================
def train_step(model, loader, optimizer, ema, device):
    model.train()
    total_loss = 0
    loss_fn = nn.SmoothL1Loss(reduction='none')
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        target = data.y.view(-1, 2)
        mask = data.y_mask.view(-1, 2)
        loss = (loss_fn(pred, target) * mask).sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        ema.update(model)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    loss_fn = nn.SmoothL1Loss(reduction='none')
    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y.view(-1, 2)
        mask = data.y_mask.view(-1, 2)
        loss = (loss_fn(pred, target) * mask).sum() / mask.sum()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_mae = 0
    total_mse = 0
    n = 0
    mean = model.mean.to(device)
    std = model.std.to(device)
    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y.view(-1, 2)
        mask = data.y_mask.view(-1, 2)
        pred_eV = (pred * std + mean) * mask
        target_eV = (target * std + mean) * mask
        total_mae += torch.abs(pred_eV - target_eV).sum().item()
        total_mse += ((pred_eV - target_eV) ** 2).sum().item()
        n += mask.sum().item()
    return total_mae / n, (total_mse / n) ** 0.5


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DimeNet++ Embedded+Attention Alpha/Beta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DimeNet++ Embedded+Attention HOMO-LUMO Gap Prediction")
    parser.add_argument("--data_path", nargs="+", required=True, help="Path(s) to .aselmdb file(s)")
    parser.add_argument("--num_molecules", type=int, default=1000, help="Number of molecules to use")
    parser.add_argument("--molecule_type", type=str, default="biomolecules", help="Molecule type filter (or 'all')")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--save_model", type=str, default=None, help="Path to save model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    mol_type = None if args.molecule_type == "all" else args.molecule_type
    bio_sample = process_file(file=args.data_path, molecule_type=mol_type, max_molecules=args.num_molecules)
    bio_data = get_data(bio_sample)
    bio_train, bio_val, bio_test = split_data(bio_data)

    train_mean, train_std = obtain_mean_std(bio_train)
    bio_train = normalize_target(bio_train, train_mean, train_std)
    bio_val = normalize_target(bio_val, train_mean, train_std)
    bio_test = normalize_target(bio_test, train_mean, train_std)

    train_loader = DataLoader(bio_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(bio_val, batch_size=args.batch_size)
    test_loader = DataLoader(bio_test, batch_size=args.batch_size)

    print(f"Train: {len(bio_train)}, Val: {len(bio_val)}, Test: {len(bio_test)}")
    print(f"Mean: {train_mean}, Std: {train_std}")

    # Model
    model = DimeNetPP_EmbeddedAttention(train_mean=train_mean, train_std=train_std).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    ema = EMA(model, decay=0.999)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    warmup_epochs = 5
    best_val = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        t_loss = train_step(model, train_loader, optimizer, ema, device)
        v_loss = evaluate(model, val_loader, device)

        if epoch >= warmup_epochs:
            scheduler.step(v_loss)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | LR: {lr:.1e} | Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Select best weights
    ema.apply(model)
    mae_ema, _ = test(model, val_loader, device)
    model.load_state_dict(best_state)
    mae_best, _ = test(model, val_loader, device)

    if mae_ema < mae_best:
        ema.apply(model)
        print(f"Using EMA weights (val MAE: {mae_ema:.4f})")
    else:
        print(f"Using best checkpoint (val MAE: {mae_best:.4f})")

    print(f"Best Val Loss: {best_val:.4f}")
    plot_losses(train_losses, val_losses)

    # Test
    mae, rmse = test(model, test_loader, device)
    print(f"\nTest MAE: {mae:.4f} eV")
    print(f"Test RMSE: {rmse:.4f} eV")

    # Save
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")
