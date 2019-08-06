import chemml
from rdkit.Chem import AllChem
from rdkit import DataStructs
import tensorflow as tf

max_len = 4

data = chemml.pd.read_csv("test_mst.csv") #load test data

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

feature_columns = [tf.feature_column.numeric_column('x', shape=X.shape[1:])]

test = tf.estimator.DNNRegressor(feature_columns=feature_columns,
     model_dir='train_mst',    
     hidden_units=[500, 300],
     label_dimension = 9
)
     
x_input = tf.estimator.inputs.numpy_input_fn(x={"x": X}, shuffle=False)

y = test.predict(input_fn=x_input) #predict from model

a = list(y)

valsXX = []
valsYX = []
valsZX = []
valsXY = []
valsYY = []
valsZY = []
valsXZ = []
valsYZ = []
valsZZ = []

#extract output values from list
for i in a:
    valsXX.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[0] ))
    valsYX.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[1] ))
    valsZX.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[2] ))
    valsXY.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[3] ))
    valsYY.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[4] ))
    valsZY.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[5] ))
    valsXZ.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[6] ))
    valsYZ.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[7] ))
    valsZZ.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[8] ))
    
data['predicted-XX'] = valsXX
data['predicted-YX'] = valsYX
data['predicted-ZX'] = valsZX
data['predicted-XY'] = valsXY
data['predicted-YY'] = valsYY
data['predicted-ZY'] = valsZY
data['predicted-XZ'] = valsXZ
data['predicted-YZ'] = valsYZ
data['predicted-ZZ'] = valsZZ

data.to_csv(r"predicted_mst.csv")
