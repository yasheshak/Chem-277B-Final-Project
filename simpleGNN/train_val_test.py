import torch

from tqdm import tqdm


class GNNTrainer(): 

    def __init__(self, model,
                 loss_function, 
                 optimizer): 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) # move to device (cpu/gpu/mps as specified)

        self.loss_function = loss_function
        self.optimizer = optimizer


    def train_validate(self, train_loader, val_loader, epochs, print_every = 10):

        # initialize variables 
        model = self.model 
        device = self.device 
        loss_function = self.loss_function
        optimizer = self.optimizer

        # for keeping track of total loss
        history = {'train_loss': [], 'val_loss': [],
                'train_mae': [], 'val_mae': [],
                'train_rmse': [], 'val_rmse': []}
    
        for epoch in tqdm(range(epochs), desc='Training'):
            # TRAINING
            model.train()
            
            # initialize losses to 0 at start of each epoch 
            epoch_train_loss = 0.0
            epoch_train_mae = 0.0
            epoch_train_rmse = 0.0
            n_train = len(train_loader)

            # iterate thru each batch 
            for batch in train_loader:
                batch = batch.to(device) # move to specified device 

                # inputs [x, edge_index, batch]
                predictions = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.to(device) # reshape 

                # compare to target and calculate loss 
                loss = loss_function(predictions, targets)

                optimizer.zero_grad() # zero out grads
                loss.backward() # back-prop 
                optimizer.step() # update/adjust 

                # calculate losses 
                epoch_train_loss += loss.item() # from optimizer 
                epoch_train_mae += torch.abs(predictions - targets).sum().item()
                epoch_train_rmse += ((predictions - targets) ** 2).sum().item()


            # calculate avg loss
            avg_train_loss = epoch_train_loss / n_train
            avg_train_mae = epoch_train_mae / n_train
            avg_train_rmse = (epoch_train_rmse / n_train) ** 0.5

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_mae = 0.0
            epoch_val_rmse = 0.0
            n_val = len(val_loader)

            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)

                    val_predictions = model(val_batch.x, val_batch.edge_index, val_batch.batch).reshape(-1)
                    val_targets = val_batch.y.reshape(-1).float().to(device)

                    val_loss = loss_function(val_predictions, val_targets)

                    epoch_val_loss += val_loss.item()

                    epoch_val_mae += torch.abs(val_predictions - val_targets).sum().item()
                    epoch_val_rmse += ((val_predictions - val_targets) ** 2).sum().item()

            avg_val_loss = epoch_val_loss / n_val
            avg_val_mae = epoch_val_mae / n_val
            avg_val_rmse = (epoch_val_rmse / n_val) ** 0.5


            # Record
            history['train_loss'].append(avg_train_loss)
            history['train_mae'].append(avg_train_mae)
            history['train_rmse'].append(avg_train_rmse)

            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(avg_val_mae)
            history['val_rmse'].append(avg_val_rmse)
            
            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

        return history


    def test(self, test_loader):

        model = self.model 
        device = self.device 
        loss_function = self.loss_function

        history = {'test_loss': [], 'test_mae': [], 'test_rmse': []}

        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        n = len(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                predictions = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.to(device)

                loss = loss_function(predictions, targets)

                total_loss += loss.item()
                total_mae += torch.abs(predictions - targets).sum().item()
                total_rmse += ((predictions - targets) ** 2).sum().item()

        avg_loss = total_loss / n
        avg_mae = total_mae / n
        avg_rmse = (total_mse / n) ** 0.5

        history['test_loss'].append(avg_loss)
        history['test_mae'].append(avg_mae)
        history['test_rmse'].append(avg_rmse)

        print(
            f"Test Loss: {avg_loss:.4f} | "
            f"Test MAE: {avg_mae:.4f} | "
            f"Test RMSE: {avg_rmse:.4f}"
        )

        return history
    