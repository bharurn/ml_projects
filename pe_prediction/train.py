import pybel
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib

max_atoms = 30

def getdist(a1, a2):
    
    x1 = a1.coords[0]
    y1 = a1.coords[1]
    z1 = a1.coords[2]
    x2 = a2.coords[0]
    y2 = a2.coords[1]
    z2 = a2.coords[2]
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

def CoulombMatrix(mol):
    cm = np.zeros((max_atoms, max_atoms))
    mag = []
    for i in mol:
        c = np.zeros(max_atoms)
        for j in mol:
            if i.idx == j.idx:
                c[j.idx-1] = (0.5*j.atomicnum**2.4)
            elif i.idx != j.idx:
                c[j.idx-1] = i.atomicnum*j.atomicnum/getdist(i, j)
        mag.append(np.linalg.norm(c))
        cm[i.idx-1] = c
    
    return np.linalg.eigvals(cm)

m = next(pybel.readfile("xyz", "test.xyz"))
CoulombMatrix(m)

data = pd.read_csv("potential_energy.csv") #read data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = next(pybel.readfile("xyz", "structures/" + mol_name +".xyz")) #load the molecule
    
    if m is None: #check if the structure is loaded
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    C.append(CoulombMatrix(m))
        
kr = KernelRidge(alpha=1.0)
kr.fit(C, data['potential_energy'].to_numpy())
joblib.dump(kr, 'pe.model', compress=9)
