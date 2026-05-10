# 277B Final - Team 4

### Project title: 
Graph Neural Networks for HOMO LUMO gap Prediction of Molecules using OMol25

### Abstract: 
The development of novel materials is largely limited by cost. Effective models for screening molecular compounds for desirable electronic properties prior to development and synthesis of materials can dramatically reduce this burden. Molecular machine learning methods such as Machine Learning Interatomic Potentials (MLIPs)  have taken prominence in recent years in the fields of chemistry and materials science due to their speed and accuracy. MLIPs are a promising alternative to computationally expensive Density Functional Theory (DFT) calculations. Graph Neural Networks (GNNS) are the dominant machine learning architecture used to model and predict molecular properties. In this project, we explore three different GNNs. We begin with a traditional graph convolutional network (SimpleGNN) to better understand the limitations of this simple architecture for our application, before exploring other more complex architectures like SchNet and DimeNet++, which are optimized for modeling quantum chemical interactions. Our models are trained on a subset of OMol25. Using a dataset of 20k biomolecules (split 64/16/20 for train/val/test), we observed poor performance for SimpleGNN. SchNet and DimeNet++ achieved a test MAE of 0.7624 and 0.5951, respectively, and an RMSE of 1.1869 and 0.9754, respectively. DimeNet++ demonstrates higher overall accuracy because of the angular information it incorporates into its embedding. 

## Directories

1. **EDA/**: Initial exploratory data analysis notebooks
2. **simpleGNN/**: Notebooks and scripts demonstrating SimpleGNN
3. **SchNet/**: Iterations of SchNet development, helper scripts, final model weights and test script
4. **DimeNet/**: Iterations of DimeNet++ development, helper scripts, final model weights and test script

## Setup

    make setup          # Creates conda environment (homo-lumo)
    conda activate homo-lumo
    make install-pyg    # Installs PyG extensions + fairchem + ase-db-backends
    make test           # Verifies all imports work
    make download-data  # Downloads OMol25 data files

## Model Demonstration

Run the final pretrained models on test data (uses `SchNet_Test.py` and `DimeNet_Test.py` with saved weights):

    make test-schnet                        # Default: 500 molecules (~15 sec)
    make test-dimenet                       # Default: 500 molecules (~30 sec)

    make test-schnet ARGS="--test_size 1000"   # ~20 sec
    make test-schnet ARGS="--test_size 2000"   # ~35 sec
    make test-dimenet ARGS="--test_size 1000"  # ~55 sec
    make test-dimenet ARGS="--test_size 2000"  # ~1 min 20 sec

Runtimes estimated on Apple M3 Pro (CPU only).

## Training

To retrain models from scratch (uses `SchNet_Train.py` and `DimeNet_Train.py`):

    make schnet ARGS="--num_molecules 20000 --save_model ./SchNet/SchNet_Final_Weights.pt"
    make dimenet ARGS="--num_molecules 20000 --save_model ./DimeNet/DimeNet_Final_Weights.pt"

Note: Training requires GPU. Estimated training time on CPU is 6+ hours (SchNet) and 20+ hours (DimeNet++). Data is automatically split 64/16/20 (train/val/test) internally.

## Helper Scripts

Each model folder (`SchNet/`, `DimeNet/`) contains its own versions of the following helper scripts, tailored to that model's data pipeline:

- **`read_multi_ase.py`**: Loads `.aselmdb` data files and extracts desired molecule type and amount
- **`extract_ab.py`**: Pre-processes extracted molecules: alpha/beta gap extraction, feature engineering (Löwdin charges, electronegativity), train/val/test splits, and target normalization

## Cleanup

    make clean          # Removes conda environment
