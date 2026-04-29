# PyTorch
import torch
from torch.utils.data import random_split

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader

class SchNetModel(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: int = 5,
                 max_num_neighbors: int = 32,
                 readout: str = "add",
                 dipole: bool = False,
                 mean: float = None,
                 std: float = None,
                 atomref: torch.Tensor = None,
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

        #Keep track of training mean and std for normalization function
        self.mean = train_mean
        self.std = train_std

    #Forward step for model
    def forward(self, data):
        out = self.schnet(data.z, data.pos, batch=data.batch)
        return out

#Determine the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Initialize model with desired parameters
bio_model = SchNetModel().to(device)
#Create ADAM optimizer based on model's parameters and desired learning rate
optimizer = torch.optim.Adam(bio_model.parameters(), lr=5e-5)
#Select loss function for model
loss_function = torch.nn.SmoothL1Loss()


def train(model: SchNetModel, train_data: list):
    model.train()
    #Keep track of total loss for all data
    total_train_loss = 0

    for data in train_data:
        data = data.to(device)

        #Reset optimizers
        optimizer.zero_grad()

        #Forward step internally performed by PyTorch to obtain predictions (Same as model.forward(data))
        y_pred = model(data)
        y_target = data.y

        #Determine train loss based on loss function with predictions and targets
        train_loss = loss_function(y_pred, y_target)

        #Backward step to calculate gradients
        train_loss.backward()
        #Updated optimizers
        optimizer.step()

        #Add train loss of current data to total train loss
        total_train_loss += train_loss.item()

    return total_train_loss / len(train_data)

@torch.no_grad()
def evaluate(model: SchNetModel, val_data: list):
    model.eval()
    total_val_loss = 0

    for data in val_data:
        data = data.to(device)

        #Forward step internally performed by PyTorch to obtain predictions (Same as model.forward(data))
        y_pred = model(data)
        y_target = data.y

        #Determine validation loss based on loss function with predictions and targets
        val_loss = loss_function(y_pred, y_target)

        #Add validation loss of current data to total validation loss
        total_val_loss += val_loss.item()

    return total_val_loss / len(val_data)

@torch.no_grad()
def test(model: SchNetModel, test_data: list):
    model.eval()

    total_mae = 0
    total_mse = 0
    n_molecules = 0

    for data in test_data:
        data = data.to(device)

        y_pred = model(data)
        y_target = data.y.view(-1, 1)


        y_pred = y_pred * model.std + model.mean
        y_target = y_target * model.std + model.mean


        mae = torch.abs(y_pred - y_target).sum()
        mse = ((y_pred - y_target) ** 2).sum()

        total_mae += mae.item()
        total_mse += mse.item()
        n_molecules += y_target.size(0)

    mean_mae = total_mae / n_molecules
    rmse = (total_mse / n_molecules) ** 0.5

    return mean_mae, rmse
