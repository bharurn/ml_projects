'''
    Python script to allow for
    easy application of ML in Chemistry
    
    Created by Bharath Raghavan
    
    Created: 4 Aug 2019
'''

import numpy as np
import pandas as pd
from rdkit import Chem
import pybel

max_atoms = 20

def xyz_to_rdkit(fname):#to convert xyz file to mol for RDKit
    #fname - xyz file name
    mol_xyz = next(pybel.readfile("xyz", fname))
    mol_block = mol_xyz.write(format="mol")
    return Chem.MolFromMolBlock(mol_block, removeHs=False) #make sure H are included

def getdist(a1, a2): #get distance between two pybel atom objects
    #a1 and a2 = pybel atoms
    x1 = a1.coords[0]
    y1 = a1.coords[1]
    z1 = a1.coords[2]
    x2 = a2.coords[0]
    y2 = a2.coords[1]
    z2 = a2.coords[2]
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

def CoulombMatrixEig(mol): #calculate and return the Eigenvalues of the coulomb matrix
    #mol - pybel molecule
    cm = np.zeros((max_atoms, max_atoms))
    for i in mol:
        c = np.zeros(max_atoms)
        for j in mol:
            if i.idx == j.idx:
                c[j.idx-1] = (0.5*j.atomicnum**2.4)
            elif i.idx != j.idx:
                c[j.idx-1] = i.atomicnum*j.atomicnum/getdist(i, j)
        cm[i.idx-1] = c
    
    return np.linalg.eigvals(cm)

def CoulombMatrix(mol): #calculate coulomb matrix sorted by magnitude
    
    cm = np.zeros((max_atoms, max_atoms))
    mag = [] #to record magnitude of each row
    for i in mol:
        c = np.zeros(max_atoms)
        for j in mol:
            if i.idx == j.idx:
                c[j.idx-1] = (0.5*j.atomicnum**2.4)
            elif i.idx != j.idx:
                c[j.idx-1] = i.atomicnum*j.atomicnum/getdist(i, j)
        mag.append(np.linalg.norm(c)) #append magnitude
        cm[i.idx-1] = c
    
    mag.extend([0]*(max_atoms-len(mol.atoms))) #add the extra with zeros
        
    sorter = pd.DataFrame(list(zip(mag, cm)), columns=['mag', 'cm'])
        
    sorter = sorter.sort_values('mag', ascending=False) #use dataframe to sort by mag
    
    s = sorter['cm'].to_numpy()
    
    scm = np.array([i for i in s]) #covert to numpy matrix
    
    return scm
