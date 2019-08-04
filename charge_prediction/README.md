This code predicits the charge on an atom (could be muliken or others depening on training data) using Random Forest Regression method. The algorithm is based on the one discussed in J. Chem. Inf. Model.2018583579-590. Atom centered Atomic fingerprints (as introduced in J. Chem. If. Comput. Sci., Vol. 25, No. 2, 1985) are used as features for each of the atoms in a molecule.

The training data should be named as 'train_charges.csv'. To train the model, run 'train.py'; this will output the file 'charges.model'. Using this run the file 'predict.py' to predict new charges from data given in 'test_charges.csv'. The results will be in 'predict_charges.csv'. All structures of molecules for training and prediction should be given in the folder 'structure/..' in XYZ format.

The data is taken from this Kaggle dataset: https://www.kaggle.com/c/champs-scalar-coupling.
