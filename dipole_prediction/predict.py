import chemml

chemml.max_atoms = 30

data = chemml.pd.read_csv("test_dipoles.csv") #load test data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = next(chemml.pb.readfile("xyz", "structures/" + mol_name +".xyz"))
    
    if m is None: #check if the structure is loaded
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    C.append(chemml.CoulombMatrix(m))
    
X = chemml.np.array(C)

feature_columns = [chemml.tf.feature_column.numeric_column('x', shape=X.shape[1:])]

test = chemml.tf.estimator.DNNRegressor(feature_columns=feature_columns,
     model_dir='train_dipole',    
     hidden_units=[500, 300],
     label_dimension = 3
)
     
x_input = chemml.tf.estimator.inputs.numpy_input_fn(x={"x": X}, shuffle=False)
     
y = test.predict(input_fn=x_input)

a = list(y)

valsX = []
valsY = []
valsZ = []

for i in a:
    valsX.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[0] ))
    valsY.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[1] ))
    valsZ.append( float( str(i).split(':')[1].split('([')[1].split('],')[0].split(',')[2] ))
    
data['predicted-X'] = valsX
data['predicted-Y'] = valsY
data['predicted-Z'] = valsZ

data.to_csv(r"predicted_dipoles.csv")
