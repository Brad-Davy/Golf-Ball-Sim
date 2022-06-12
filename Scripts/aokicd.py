import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, curve_fit
import scipy.interpolate


### Aoki paper data ###
def extract_data(filename):
    velocity = []
    Cd =[]
    filename = "../Data/AokiPaper/" + filename
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

def A_cd_func(i):
    try:
        f = interp(i)
    except:
        if i < 18:
            f = top_fit[1] + top_fit[0]*i
        if i > 55:
            f = bottom_fit[1] + bottom_fit[0]*i
    return f

X,Y = extract_data('cl.txt')
top_fit2 = np.polyfit(X[2:],Y[2:],2)
top_fit1 = np.polyfit(X[3:],Y[3:],1)
bottom_fit1 = np.polyfit(X[:3],Y[:3],1)
interp1 = scipy.interpolate.interp1d(X,Y)

def A_cl_func(spin):
    try:
        cl = interp1(spin)
    except:
        if spin < 2200:
            cl = bottom_fit1[0]*spin + bottom_fit1[1]
        elif spin > 3900:
            cl =  top_fit1[0]*spin + top_fit1[1]
    if 4000 > spin > 3450:
        cl = top_fit2[0]*spin*spin + top_fit2[1]*spin + top_fit2[2]
    return cl

