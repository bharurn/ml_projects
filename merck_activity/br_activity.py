from time import time
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from math import sqrt

t0 = time()
data = pd.read_csv("train_activity.csv", nrows=5000) #read data
print("done reading data in %0.3fs" % (time() - t0))

data = data.drop('MOLECULE', axis=1)

x_train, x_test, y_train, y_test = train_test_split(data.drop('Act',axis=1), 
           data['Act'], test_size=0.5,
            random_state=2)

#run and fit Bayesian Ridge method
model = BayesianRidge()

t0 = time()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("done fitting in %0.3fs" % (time() - t0))

f = open("bayesian_ridge.txt", "w")

f.write("R Squared:" + str(r2_score(y_test, y_pred)))

f.write("RMSE: " + str(sqrt(mean_squared_error(y_test, y_pred))))

f.write("MAE:" + str(mean_absolute_error(y_test, y_pred)))