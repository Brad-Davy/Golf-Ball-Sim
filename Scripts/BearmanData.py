import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate
from scipy.optimize import optimize, curve_fit

def extract_data(filename):
    velocity = []
    Cd =[]
    filename = 'C:\\Users\\bradl\\OneDrive\\Documents\\Useful\\Uni\\3rdYear\\Project\\Data\\Bearman\\' + filename
    data_file = open(filename)
    data_file = data_file.read().split('\n')
    for elements in data_file:
        if len(elements.split(',')) == 2:
            velocity.append(float(elements.split(',')[0]))
            Cd.append(float(elements.split(',')[1]))
    return velocity, Cd
x,y = extract_data('CdVelocity.txt')
z = np.polyfit(x, y, 12)
f = np.poly1d(z)
Y = []
X = []

bottomfit = np.polyfit(x[8:],y[8:],1)

def Bearman_Cd_func(i):
    if i > 69:
        new_y = bottomfit[0]*i + bottomfit[1]
    if i < 15:
        new_y = 0.5271980521935997
    elif 14<i<70:
        new_y = f(i)
    return new_y
