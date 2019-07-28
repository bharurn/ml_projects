from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import pybel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

X = []

# set RF-parameter
n_estimators = 100
min_samples_split = 6
random_state = 0
n_jobs = -1
max_depth = 6
min_samples_leaf = 6
max_len = 4

def xyz_to_mol(fname):
    
    mol_xyz = next(pybel.readfile("xyz", "test.xyz"))
    mol_block = mol_xyz.write(format="mol")
    return Chem.MolFromMolBlock(mol_block, removeHs=False)

m = xyz_to_mol("test.xyz")

m = Chem.MolFromMolFile("test.mol", removeHs=False)

for at in m.GetAtoms():                             
      aid = at.GetIdx()
      # generate atom-centered AP fingerprint
      fp = AllChem.GetHashedAtomPairFingerprintAsBitVec(m, maxLength=max_len, fromAtoms=[aid])
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(fp, arr)
      X.append(arr)

Y = [-0.493528, 0.060056, 0.060213, -0.493508, 0.144458, 0.144462, 0.144461, 0.144464, 0.144459, 0.144464]

rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,\
                             min_samples_split=min_samples_split,n_jobs=n_jobs,\
                             min_samples_leaf=min_samples_leaf)
rf.fit(X, Y)
# write the model to file
joblib.dump(rf, 'test.model', compress=9)
