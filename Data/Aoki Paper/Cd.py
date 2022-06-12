import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import scipy.interpolate


### Aoki paper data ###
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

x,y = extract_data('cd.txt')
interp = scipy.interpolate.interp1d(x, y)
top_fitx = x[:3]
top_fity = y[:3]
top_fit = np.polyfit(top_fitx,top_fity,1)
bottom_fitx = x[15:]
bottom_fity = y[15:]
bottom_fit = np.polyfit(bottom_fitx,bottom_fity,1)

def func(i):
    try:
        f = interp(i)
    except:
        if i < 18:
            f = top_fit[1] + top_fit[0]*i
        if i > 55:
            f = bottom_fit[1] + bottom_fit[0]*i
    return f

p = np.arange(0,80,1)
P = []
for elements in p:
    P.append(func(elements))
plt.plot(p,P,label = 'Fit', color = 'black')
plt.grid()
plt.style.use('seaborn')
plt.xlabel('Velocity \ m/s',fontname="Arial", fontsize=14)
plt.ylabel('Drag Coefficent',fontname="Arial", fontsize=14)
plt.scatter(x,y,color = 'red', label = 'Aoki Data')
plt.legend()
plt.savefig('AokiCD.eps', format='eps', dpi=2000)
plt.show()