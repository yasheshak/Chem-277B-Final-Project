# 277B Final - Team 4
Project Title: ____ <br>
Abstract: ____<br>
Directories<br>
	1. EDA: Initial exploratory data analysis notebooks.
	2. Data Preprocessing: Helper scripts for loading and preprocessing data for model training 
	3. Hyperparameter Tuning: Notebooks containing helper scripts implementing SchNet baseline sweep and grid search for hyperparameter tuning.
	4. simpleGNN: Notebooks demonstrating working simpleGNN
	5. SchNet: Notebooks demonstrating iterations of SchNet development. Includes final version of SchNet.
	6. DimeNet: Notebooks demonstrating iterations of DimeNet++ development. Includes final version of DimeNet++.
	7. Demonstration: Contains necessary scripts to run test demonstrations of final models.

Model demonstration<br>
Instructions to run  the final versions of our trained SchNet and DimeNet model:<br>
In the ‘Demonstration’ directory, run:<br>
	1. make environment
	2. make SchNet: Tests on default number of molecules (500)
	3. make DimeNet: Tests on default number of molecules (500)
	4. make clean

The estimated runtimes for the default targets are X (SchNet) and Y (DimeNet). For further experimentation, other targets may be used:
	1. make SchNet_1000 (est runtime: _)
	2. make SchNet_2000 (est runtime: _)
	3. make DimeNet_1000 (est runtime: _)
	4. make DimeNet_2000 (est runtime: _)

Helper script descriptions
	1. ‘read_multi_ase.py’
		Includes functions to load in data and extract desired molecule type and amount.
	2. ‘extract.py’
		Pre–processes the extracted files by 
		i. train/val/test splits
		ii. normalizing target
		iii. preparing torch loaders
