import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, curve_fit
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


x,y = extract_data('cl.txt')
top_fit2 = np.polyfit(x[2:],y[2:],2)
top_fit = np.polyfit(x[3:],y[3:],1)
bottom_fit = np.polyfit(x,y,1)
interp = scipy.interpolate.interp1d(x,y)


plt.scatter(x,y,color = 'red')
plt.style.use('seaborn')
plt.legend()
plt.xlabel('Spin / rpm',fontname="Arial", fontsize=14)
plt.ylabel('Cl',fontname="Arial", fontsize=14)
plt.grid()
plt.savefig('AokiCl.eps', format='eps', dpi=2000)
plt.show()