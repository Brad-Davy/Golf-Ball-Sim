### Aoki paper data ###
import math
import numpy as np  

def extract_data(filename):
    velocity = []
    Cd =[]
    filename = "C:\\Users\\bradl\\OneDrive\\Documents\\Useful\\Uni\\3rdYear\\Project\\Data\\AokiPaper\\" + filename
    data_file = open(filename)
    data_file = data_file.read().split('\n')
    for elements in data_file:
        if len(elements.split(',')) == 2:
            velocity.append(float(elements.split(',')[0]))
            Cd.append(float(elements.split(',')[1]))
    return velocity, Cd

extract_data('cd.txt')
fit = np.polyfit(np.log(x),y,1)

def func(X,fit):
    X = float(X)
    f1 = float(fit[0])
    f2 = float(fit[1])
    return math.exp(f2)*math.exp(f1*X)

fitted = []

for i in range(len(x)):
    fitted.append(func(x[i],fit))


plt.plot(x,fitted)
plt.scatter(x,y)
plt.show()