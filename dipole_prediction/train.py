import chemml

chemml.max_atoms = 30

data = chemml.pd.read_csv("dipole_moments.csv", nrows=5000) #read data

C = []

mols = list(dict.fromkeys(data['molecule_name'])) #get unique molecule names

for mol_name in mols:
    m = next(chemml.pb.readfile("xyz", "structures/" + mol_name +".xyz")) #load the molecule
    
    if m is None: #check if the structure is loaded
        data = data[data.molecule_name!=mol_name] #if there is mismatch, delete this entry from dataframe
        print ("Error in loading molecule: " + str(mol_name) + ". Skipping...")
        continue
    
    C.append(chemml.CoulombMatrix(m))
    
X = chemml.np.array(C)
            
Y = chemml.np.column_stack((data['X'].to_numpy(),data['Y'].to_numpy(), data['Z'].to_numpy()))

feature_columns = [chemml.tf.feature_column.numeric_column('x', shape=X.shape[1:])]

estimator = chemml.tf.estimator.DNNRegressor(feature_columns=feature_columns,
     model_dir='train_dipole',    
     hidden_units=[500, 300],   
     label_dimension = 3,
     optimizer=chemml.tf.train.ProximalAdagradOptimizer(      
          learning_rate=0.1,      
          l1_regularization_strength=0.001    
      )
)

train_input = chemml.tf.estimator.inputs.numpy_input_fn(   
           x={"x": X},    
           y=Y,    
           #batch_size=128,    
           shuffle=False,    
           num_epochs=None)

estimator.train(input_fn = train_input,steps=5000)
