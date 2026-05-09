# 277B Final - Team 4

### Project title: 
Graph Neural Networks for HOMO LUMO gap Prediction of Molecules using OMol25

### Abstract: 
The development of novel materials is largely limited by cost. Effective models for screening molecular compounds for desirable electronic properties prior to development and synthesis of materials can dramatically reduce this burden. Molecular machine learning methods such as Machine Learning Interatomic Potentials (MLIPs)  have taken prominence in recent years in the fields of chemistry and materials science due to their speed and accuracy. MLIPs are a promising alternative to computationally expensive Density Functional Theory (DFT) calculations. Graph Neural Networks (GNNS) are the dominant machine learning architecture used to model and predict molecular properties. In this paper, we explore three different GNNs. We begin with a traditional graph convolutional network (SimpleGNN) to better understand the limitations of this simple architecture for our application, before exploring other more complex architectures like SchNet and DimeNet++, which are optimized for modeling quantum chemical interactions. Our models are trained on a subset of OMol25. For 20k molecular compounds, we observed poor performance for SimpleGNN. SchNet and DimeNet++ achieved a final test MAE of 0.7624 and 0.6112, respectively, and an RMSE of 1.1869 and 0.9933, respectively. DimeNet++ demonstrates higher overall accuracy because of the angular information it incorporates into its embedding. 

## Directories

1. **EDA**: Initial exploratory data analysis notebooks
2. **Data Preprocessing**: Helper scripts for loading and preprocessing data for model training
3. **Hyperparameter Tuning**: Notebooks containing helper scripts implementing SchNet baseline sweep and grid search for hyperparameter tuning
4. **simpleGNN**: Notebooks demonstrating working simpleGNN
5. **SchNet**: Notebooks demonstrating iterations of SchNet development. Includes final version of SchNet
6. **DimeNet**: Notebooks demonstrating iterations of DimeNet++ development. Includes final version of DimeNet++
7. **Demonstration**: Contains necessary scripts to run test demonstrations of final models

## Model Demonstration

Instructions to run the final versions of our trained SchNet and DimeNet models:

In the root directory, run:

    make environment
    make SchNet        # Tests on default number of molecules (500)
    make DimeNet       # Tests on default number of molecules (500)
    make clean

The estimated runtimes for the default targets are **X** (SchNet) and **Y** (DimeNet). 

For further experimentation, other targets may be used:

    make SchNet_1000   # est runtime: _
    make SchNet_2000   # est runtime: _
    make DimeNet_1000  # est runtime: _
    make DimeNet_2000  # est runtime: _

## Helper Script Descriptions

1. **`read_multi_ase.py`**  
   Includes functions to load in data and extract desired molecule type and amount

2. **`extract.py`**  
   Pre-processes the extracted files by:  
   i. train/val/test splits  
   ii. normalizing target  
   iii. preparing torch loaders
