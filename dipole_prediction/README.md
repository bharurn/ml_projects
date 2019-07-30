This code predicits the dipole of a molecules using Kernel Regression method. Sorted Coulumb matrices are used as features for each 
molecule.

The training data should be named as 'train_dipoles.csv'. To train the model, run 'train.py'; this will output the file 'dipoleX.model', 
'dipoleY.model' and 'dipoleZ.model'. Using this run the file 'predict.py' to predict new charges from data given in 'test_dipoles.csv'.
The results will be in 'predict_charges.csv'. All structures of molecules for training and prediction should be given in the folder 
'structure/..' in XYZ format.

The precitions from the alogrithm are currently not very accurate. It has to be worked on and changed.
