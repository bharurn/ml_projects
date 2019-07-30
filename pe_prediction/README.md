This code predicits the potential energy of a molecules using Kernel Regression method. Eigenvalues of the coulumb matrices are used as features for each molecule.

The training data should be named as 'train_dipoles.csv'. To train the model, run 'train.py'; this will output the file 'pe.model'. Using this run the file 'predict.py' to predict new charges from data given in 'test_pe.csv'. The results will be in 'predict_pe.csv'. All structures of molecules for training and prediction should be given in the folder 'structure/..' in XYZ format.

The predictions from the alogrithm are quite accurate, especially compared with the dipole moment prediction algorithm which used the sorted Coulumb matrices itself (instead of the eigenvalues) as features.
