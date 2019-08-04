This code predicits the dipole of a molecules using Deep Neural Network Regression method. Sorted Coulumb matrices are used as features for each 
molecule.

The training data should be named as 'train_dipoles.csv'. To train the model, run 'train.py'; this will output the files 'dipoleX.model', 
'dipoleY.model' and 'dipoleZ.model'. Using this run the file 'predict.py' to predict new charges from data given in 'test_dipoles.csv'.
The results will be in 'predict_charges.csv'. All structures of molecules for training and prediction should be given in the folder 
'structure/..' in XYZ format.

The predictions from the alogrithm are currently not very accurate. It is probably because the sorted coulumb matrices are not effective as features. Either some other features or another ML algorithm has to be used.

The data is taken from this Kaggle dataset: https://www.kaggle.com/c/champs-scalar-coupling.
