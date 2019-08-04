import pybel
import chemml
from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib

chemml.max_atoms = 30

data = chemml.pd.read_csv("potential_energy.csv") #read data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = next(pybel.readfile("xyz", "structures/" + mol_name +".xyz")) #load the molecule
    
    if m is None: #check if the structure is loaded
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    C.append(chemml.CoulombMatrixEig(m))
        
kr = KernelRidge(alpha=1.0)
kr.fit(C, data['potential_energy'].to_numpy())
joblib.dump(kr, 'pe.model', compress=9)
