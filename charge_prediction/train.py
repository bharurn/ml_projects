from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import pybel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd

# set RF-parameter
n_estimators = 100
min_samples_split = 6
random_state = 0
n_jobs = -1
max_depth = 6
min_samples_leaf = 6
max_len = 4

def xyz_to_mol(fname):#to convert xyz file to mol for RDKit
    
    mol_xyz = next(pybel.readfile("xyz", fname))
    mol_block = mol_xyz.write(format="mol")
    return Chem.MolFromMolBlock(mol_block, removeHs=False) #make sure H are included

data = pd.read_csv("train_charges.csv") #read data

X = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = xyz_to_mol("structures/" + mol_name +".xyz") #load the molecule
    
    if m is None or len(m.GetAtoms()) != len(data[data.molecule_name==mol_name]): #check if the structure is loaded and no mismatch
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    for at in m.GetAtoms(): # cycle through all atoms                            
        aid = at.GetIdx()
        # generate atom-centered AP fingerprint
        fp = AllChem.GetHashedAtomPairFingerprint(m, maxLength=max_len, fromAtoms=[aid])
        arr = np.zeros(1,)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr) #all APs in one matrix

Y = data['mulliken_charge'].to_numpy() #all charges in one array

#run and fit RandomForestRegression method
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,\
                             min_samples_split=min_samples_split,n_jobs=n_jobs,\
                             min_samples_leaf=min_samples_leaf)
rf.fit(X, Y)

joblib.dump(rf, 'charges.model', compress=9)
