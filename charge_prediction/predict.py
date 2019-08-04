from rdkit.Chem import AllChem
from rdkit import DataStructs
import chemml
import numpy as np
from sklearn.externals import joblib
import pandas as pd

max_len = 4

rf = joblib.load('charges.model') #load model

data = pd.read_csv("test_charges.csv") #load test data

X = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = chemml.xyz_to_rdkit("structures/" + mol_name +".xyz")
    
    if m is None or len(m.GetAtoms()) != len(data[data.molecule_name==mol_name]): #check for file loading errors
        data = data[data.molecule_name!=mol_name]
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    for at in m.GetAtoms():                            
        aid = at.GetIdx()
        # generate atom-centered AP fingerprint
        fp = AllChem.GetHashedAtomPairFingerprint(m, maxLength=max_len, fromAtoms=[aid])
        arr = np.zeros(1,)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
       
#save predicted values to csv
data['predicted'] = rf.predict(X)
data.to_csv(r"predicted_charges.csv")
