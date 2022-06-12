#Neural network#
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
import numpy as np 


## Function to pull data and parse it properly ##
def extract_data(filename):
    reynolds = []
    drag =[]
    data_file = open(filename)
    data_file = data_file.read().split('\n')
    for elements in data_file:
        if len(elements.split(',')) == 2:
            reynolds.append(float(elements.split(',')[0]))
            drag.append(float(elements.split(',')[1]))
    return reynolds, drag

## Pulls the data from the text file ##
r1000,d1000 = extract_data('rpm1000.txt')
r1400,d1400 = extract_data('rpm1400.txt')
r1800,d1800 = extract_data('rpm1800.txt')
r2200,d2200 = extract_data('rpm2200.txt')
r2600,d2600 = extract_data('rpm2600.txt')
r3000,d3000 = extract_data('rpm3000.txt')
r3400,d3400 = extract_data('rpm3400.txt')
sp=[1000]*7 + [1400]*7 + [1800]*7 + [2200]*7 + [2600]*7 + [3000]*7 + [3400]*7
r = r1000+r1400+r1800+r2200+r2600+r3000+r3400
d = d1000+d1400+d1800+d2200+d2600+d3000+d3400
params = []


for i in range(len(sp)):
    params.append(r[i]*sp[i])
params = np.array(params)


X_train, X_test, y_train, y_test = train_test_split(np.array(sp).reshape(-1,1), np.array(d).reshape(-1,1), test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
