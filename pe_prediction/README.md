This code predicits the potential energy of a molecules using Kernel Regression method. Eigenvalues of the coulumb matrices are used as features for each molecule.

The training data should be named as 'train_pe.csv'. To train the model, run 'train.py'; this will output the file 'pe.model'. Using this run the file 'predict.py' to predict new charges from data given in 'test_pe.csv'. The results will be in 'predict_pe.csv'. All structures of molecules for training and prediction should be given in the folder 'structure/..' in XYZ format.

Compared to the dipole prediciton algorithm, the predictions for potential energy are more accurate. This shows that the Coulomb matrix is an adequate feature for prediciting potential energies but not molecular dipoles.

The data is taken from this Kaggle dataset: https://www.kaggle.com/c/champs-scalar-coupling.
