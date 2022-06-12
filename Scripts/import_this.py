import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize, curve_fit
import scipy as sp 
from scipy.interpolate import griddata, interp2d,Rbf,NearestNDInterpolator as nd, RectBivariateSpline as rbs
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import itertools
import plotly
import plotly.graph_objs as go

def extract_data(filename):
    velocity = []
    Cd =[]
    filename = 'C:\\Users\\bradl\\OneDrive\\Documents\\Useful\\Uni\\3rdYear\\Project\\Data\\Bearman\\Cl\\' + filename
    data_file = open(filename)
    data_file = data_file.read().split('\n')
    for elements in data_file:
        if len(elements.split(',')) == 2:
            velocity.append(float(elements.split(',')[0]))
            Cd.append(float(elements.split(',')[1]))
    return velocity, Cd

## Pulls the data from the text file ##
r136,d136 = extract_data('13.7.txt')
r216,d216 = extract_data('21.6.txt')
r299,d299 = extract_data('29.9.txt')
r384,d384 = extract_data('38.4.txt')
r459,d459 = extract_data('46.9.txt')
r552,d552 = extract_data('55.2.txt')
r631,d631 = extract_data('63.1.txt')
r719,d719 = extract_data('71.9.txt')
r802,d802 = extract_data('80.2.txt')
r889,d889 = extract_data('89.1.txt')

## Combine data ##
vel = [13.6]*6 + [21.6]*6 + [29.9]*6 + [38.4]*6 + [46.9]*6 + [55.2]*6 + [63.1]*6 + [71.9]*6 + [80.2]*6 + [89.1]*6
Spin = r136 + r216 + r299 + r384 + r459 + r552 + r631 + r719 + r802 + r889
Drag = d136 + d216 + d299 + d384 + d459 + d552 + d631 + d719 + d802 + d889

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z,rcond=-1)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


f = polyfit2d(np.array(vel),np.array(Spin),np.array(Drag),order = 2)
x = np.linspace(0,100,20)
y = np.linspace(0,8000,20)
xx,yy = np.meshgrid(x,y)
zz = polyval2d(xx,yy,f)

def BearmanClVelSpin(velocity,spin):
    return polyval2d(float(velocity),float(spin),f)

def plot_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ys=r136,zs=d136,xs=[13.6]*6)
    #ax.plot_surface(xx,yy,zz)
    ax.scatter(ys=r216,zs=d216,xs=[21.6]*6)
    ax.scatter(ys=r299,zs=d299,xs=[29.9]*6)
    ax.scatter(ys=r384,zs=d384,xs=[38.4]*6)
    ax.scatter(ys=r459,zs=d459,xs=[46.9]*6)
    ax.scatter(ys=r552,zs=d552,xs=[55.2]*6)
    ax.scatter(ys=r631,zs=d631,xs=[63.1]*6)
    ax.scatter(ys=r719,zs=d719,xs=[71.9]*6)
    ax.scatter(ys=r802,zs=d802,xs=[80.2]*6)
    ax.scatter(ys=r889,zs=d889,xs=[89.1]*6)
    plt.show()