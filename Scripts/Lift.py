from scipy.optimize import minimize
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

## Function to pull data and parse it properly ##
def extract_data(filename):
    reynolds = []
    drag =[]
    filename = "../Data/Lift/" + filename
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

## Input function, this is the function to be optimised to fit the data ## 
def f(spin, reynolds,a,b,c,d):
    return a + reynolds*spin*b  + c*np.exp((spin*reynolds)*(spin*reynolds)*(spin*reynolds)*d)

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
    params[0]=a
    params[1]=b
    params[2]=c
    params[3]=d
    return diff 

def initialise_function(a=0.1,b=0.0000000005,c=0.00000000000000000001,d=0.0000000000000000000000000001):
    minimize(fun=fun,x0 = 0.1,args=(0.000000001,0.1,0.0000000000000000000000000001),method='TNC')
    print('The Lift function has been determined.')

## Creates a mesh and plots the calculated function on top of the data ##
def plot_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ys=r1000,zs=d1000,xs=[1000]*7)
    ax.scatter(ys=r1400,zs=d1400,xs=[1400]*7)
    ax.scatter(ys=r1800,zs=d1800,xs=[1800]*7)
    ax.scatter(ys=r2200,zs=d2200,xs=[2200]*7)
    ax.scatter(ys=r2600,zs=d2600,xs=[2600]*7)
    ax.scatter(ys=r3000,zs=d3000,xs=[3000]*7)
    ax.scatter(ys=r3400,zs=d3400,xs=[3400]*7)
    x = np.linspace(1000,3400,100)
    y = np.linspace(70000,110000,100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    X, Y = np.meshgrid(x, y)
    Z = f(spin=X, reynolds=Y,a=params[0],b=params[1],c=params[2],d=params[3])
    ax.contour(X, Y, Z,100)
    plt.show()


def lift_function(angular_velocity, velocity):
    return float(f(spin=angular_velocity, reynolds=velocity,a=params[0],b=params[1],c=params[2],d=params[3]))

initialise_function()
