from train_val_test import *        # for performing train/val/test
from simpleGNN import *             # all developed simple GCN variants 

import torch.optim as optim 


def main(): 



    # ========= Simplest GNN (3-layer GCN) =========

    # initialize model 
    simpleGNN = SimpleGNN(num_node_features=1, 
                          hidden_channels=128) 
    
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(simpleGNN.parameters(), lr=1e-4)
    
    simpleGNNTrainer = GNNTrainer(model = simpleGNN, 
                                  loss_function = loss_fn,
                                  optimizer = optimizer)

    train_val_results = simpleGNNTrainer.train_validate()

    plot_train_val(history = train_val_results,
                   model_name = 'SimpleGNN')

    

if __name__ == "__main__": 

    main()