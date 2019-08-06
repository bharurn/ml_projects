import chemml
from rdkit.Chem import AllChem
from rdkit import DataStructs
import tensorflow as tf


max_len = 4

data = chemml.pd.read_csv("magnetic_shielding_tensors.csv", nrows=100000) #read data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = chemml.xyz_to_rdkit("structures/" + mol_name +".xyz") #load the molecule
    
    if m is None or len(m.GetAtoms()) != len(data[data.molecule_name==mol_name]): #check if the structure is loaded and no mismatch
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    for at in m.GetAtoms(): # cycle through all atoms                            
        aid = at.GetIdx()
        # generate atom-centered AP fingerprint
        fp = AllChem.GetHashedAtomPairFingerprint(m, maxLength=max_len, fromAtoms=[aid])
        arr = chemml.np.zeros(1,)
        DataStructs.ConvertToNumpyArray(fp, arr)
        C.append(arr) #all APs in one matrix
  
X = chemml.np.array(C)
# all mag sheilding tensors
Y = chemml.np.column_stack((data['XX'].to_numpy(),data['YX'].to_numpy(), data['ZX'].to_numpy(), data['XY'].to_numpy()
                            , data['YY'].to_numpy(), data['ZY'].to_numpy(), data['XZ'].to_numpy(), data['YZ'].to_numpy(), data['ZZ'].to_numpy() ))

feature_columns = [tf.feature_column.numeric_column('x', shape=X.shape[1:])]

estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns,
     model_dir='train_mst',    
     hidden_units=[500, 300],   
     label_dimension = 9,
     optimizer=tf.train.ProximalAdagradOptimizer(      
          learning_rate=0.1,      
          l1_regularization_strength=0.001    
      )
)

#convert X and Y to tf inputs
train_input = tf.estimator.inputs.numpy_input_fn(   
           x={"x": X},    
           y=Y,    
           #batch_size=128,    
           shuffle=False,    
           num_epochs=None)

estimator.train(input_fn = train_input,steps=5000)
