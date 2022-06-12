from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


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

## Combine the arrays togeather ##
sp=[1000]*7 + [1400]*7 + [1800]*7 + [2200]*7 + [2600]*7 + [3000]*7 + [3400]*7
r = r1000+r1400+r1800+r2200+r2600+r3000+r3400
d = d1000+d1400+d1800+d2200+d2600+d3000+d3400
params = []

## Create array with the two inputs for the machine learning ##
for i in range(len(sp)):
    p = [0,0]
    p[0] = r[i]
    p[1] = sp[i]
    params.append(p)
params = np.array(params)

## Split data to train the model ##
X_train, X_test, y_train, y_test = train_test_split(params, d, test_size=0.2)

## Create an instance of the LinearRegression class to train ##
clf = LinearRegression()

# Train the model ##
clf.fit(np.array(X_train).reshape(-1, 2), np.array(y_train).reshape(-1, 1))

## Determine how good of a fit there is ##
confidence = clf.score(np.array(X_test).reshape(-1, 2), np.array(y_test).reshape(-1, 1))
print (confidence)

## Predict the data for the inputs ##
drag = clf.predict(np.array(params).reshape(-1, 2))

## Pull data back into 1D arrays to plot ##
reynolds = []
spin = []
for elements in params:
    reynolds.append(elements[0])
    spin.append(elements[1])

## Function to determine the square of errors between model and data ##
def error(x,x1):
    error = []
    if len(x) == len(x1):
        for i in range(len(x)):
            df = abs(x[i] - x1[i])
            error.append(df*df)
    else:
        print ('X array length dosnt match')
    error = np.sum(np.array(error))
    return error
print(error(drag,d))


## Create figure and data for contour plot ##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(1000,3400,100)
y = np.linspace(70000,110000,100)
X, Y = np.meshgrid(x, y)
reynolds = []
spin = []
params = []
for x in range(len(X[0])):
    for i in range(len(X[x])):
        params.append([Y[x][i],X[x][i]])

## model predicting the outputs for the mesh ##
drag = clf.predict(np.array(params).reshape(-1, 2))
drag = drag.reshape(100,100)
ax.contour(X,Y,drag,100)

## Plot the data ##
ax.scatter(ys=r1000,zs=d1000,xs=[1000]*7)
ax.scatter(ys=r1400,zs=d1400,xs=[1400]*7)
ax.scatter(ys=r1800,zs=d1800,xs=[1800]*7)
ax.scatter(ys=r2200,zs=d2200,xs=[2200]*7)
ax.scatter(ys=r2600,zs=d2600,xs=[2600]*7)
ax.scatter(ys=r3000,zs=d3000,xs=[3000]*7)
ax.scatter(ys=r3400,zs=d3400,xs=[3400]*7)

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

# Add the arrays togeather ##
sp=[1000]*7 + [1400]*7 + [1800]*7 + [2200]*7 + [2600]*7 + [3000]*7 + [3400]*7
r = r1000+r1400+r1800+r2200+r2600+r3000+r3400
d = d1000+d1400+d1800+d2200+d2600+d3000+d3400

## Input function, this is the function to be optimised to fit the data ## 
def f(spin, reynolds,a,b,c,d):
    return a + spin*reynolds*b - (spin*reynolds)*(spin*reynolds)*c + (spin*reynolds)*(spin*reynolds)*(spin*reynolds)*d

## Calculates the difference between the function and the experimental data ##
def difference(index,a,b,c,d):
    diff = d1000[index] - f(1000,r1000[index],a,b,c,d) + d1400[index] - f(1400,r1400[index],a,b,c,d) + d1800[index] - f(1800,r1800[index],a,b,c,d) + d2200[index] - f(2200,r2200[index],a,b,c,d) + d2600[index] - f(2600,r2600[index],a,b,c,d) + d3000[index] - f(3000,r3000[index],a,b,c,d) + d3400[index] - f(3400,r3400[index],a,b,c,d)
    return diff

## Holds parameters of constants ##
params = [0,0,0,0]

## returns the squared differnece between the function and the data points ##
def fun(x0, b,c,d):
    a=x0
    diff = difference(index = 0,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 1,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 2,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 3,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 4,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 5,a=a,b=b,c=c,d=d)**2
    diff = diff + difference(index = 6,a=a,b=b,c=c,d=d)**2
    params[0]=float(x0)
    params[1]=b
    params[2]=c
    params[3]=d
    print (params)
    return diff 

## Calls the minimise fucntion from the scipy library ##
def minimise(a=0.1,b=0.0000000005,c=0.00000000000000000001,d=0.0000000000000000000000000001):
    min = minimize(fun=fun,x0 = 0.1,args=(0.0000000005,0.00000000000000000001,0.0000000000000000000000000001),method='TNC')
    print(min)


## Creates a mesh and plots the calculated function on top of the data ##
x = np.linspace(1000,3400,100)
y = np.linspace(70000,110000,100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
minimise()


def error(x,x1):
    error = []
    if len(x) == len(x1):
        for i in range(len(x)):
            df = abs(x[i] - x1[i])
            error.append(df*df)
    else:
        print ('X array length dosnt match')
    error = np.sum(np.array(error))
    return error

a=params[0]
b=params[1]
c=params[2]
e=params[3]
func = []

for i in range(len(sp)):
    func.append(f(sp[i],r[i],a,b,c,e))    

print(error(d,func))

X, Y = np.meshgrid(x, y)
Z = f(spin=X, reynolds=Y,a=float(params[0]),b=float(params[1]),c=float(params[2]),d=float(params[3]))
ax.contour(X, Y, Z,100,colors=['#808080', '#A0A0A0', '#C0C0C0'])
plt.xlabel('Spin / rev/s')
plt.ylabel('Reynolds number')
#plt.zlabel('Lift coefficent')
plt.show()
