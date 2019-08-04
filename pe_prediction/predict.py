import pybel
import chemml
from sklearn.externals import joblib

chemml.max_atoms = 30

data = chemml.pd.read_csv("test_pe.csv") #load test data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = next(pybel.readfile("xyz", "structures/" + mol_name +".xyz"))
    
    if m is None: #check if the structure is loaded
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    C.append(chemml.CoulombMatrixEig(m))
    
kr = joblib.load('pe.model') #load model

data['predicted-pe'] = kr.predict(C)

data.to_csv(r"predicted_pe.csv")
