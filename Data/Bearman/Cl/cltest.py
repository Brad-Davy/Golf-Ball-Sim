import matplotlib.pyplot as plt 
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize, curve_fit
import scipy as sp 
from scipy.interpolate import griddata, interp2d,Rbf,NearestNDInterpolator as nd, RectBivariateSpline as rbs
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
import itertools
import plotly
import plotly.graph_objs as go
import chart_studio.plotly as py
from import_this import surface

def extract_data(filename):
    velocity = []
    Cd =[]
    filename = 'C:\\Users\\bradl\\OneDrive\\Documents\\Useful\\Uni\\3rdYear\\Project\\Data\\Bearman\\Cd\\' + filename
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
r459,d459 = extract_data('45.9.txt')
r552,d552 = extract_data('55.2.txt')
r631,d631 = extract_data('63.1.txt')
r719,d719 = extract_data('71.9.txt')
r802,d802 = extract_data('80.2.txt')
r889,d889 = extract_data('88.9.txt')

## Combine data ##
vel = [13.6]*7 + [21.6]*7 + [29.9]*7 + [38.4]*7 + [45.9]*7 + [55.2]*7 + [63.1]*7 + [71.9]*7 + [80.2]*7 + [88.9]*7 
#Spin = r136 + r216 + r299 + r384 + r459 + r552 + r631 + r719 + r802 + r889 
Spin = r136*10
Velocity = [13.6,21.6,29.9,38.4,45.9,55.2,63.1,71.9,80.2,88.9]
Drag = d136 + d216 + d299 + d384 + d459 + d552 + d631 + d719 + d802 + d889 
r136.sort()
print(r136)
data_dict = {}

for d in range(len(vel)):
    data_dict.update({str(vel[d])+','+str(Spin[d]):Drag[d]})


def find_nearest_spin(array, value):
    if value < array[0]:
        return array[0],array[1]
    elif array[0] < value < array[1]:
        return array[0],array[1]
    elif array[1] < value < array[2]:
        return array[1],array[2]
    elif array[2] < value < array[3]:
        return array[2],array[3]
    elif array[3] < value < array[4]:
        return array[3],array[4]
    elif array[4] < value < array[5]:
        return array[4],array[5]
    elif array[5] < value < array[6]:
        return array[5],array[6]
    elif value > array[6]:
        return array[5],array[6]

def find_nearest_vel(array, value):
    if value < array[0]:
        return array[0],array[1]
    elif array[0] < value < array[1]:
        return array[0],array[1]
    elif array[1] < value < array[2]:
        return array[1],array[2]
    elif array[2] < value < array[3]:
        return array[2],array[3]
    elif array[3] < value < array[4]:
        return array[3],array[4]
    elif array[4] < value < array[5]:
        return array[4],array[5]
    elif array[5] < value < array[6]:
        return array[5],array[6]
    elif array[6] < value < array[7]:
        return array[6],array[7]
    elif array[7] < value < array[8]:
        return array[7],array[8]
    elif array[8] < value < array[9]:
        return array[8],array[9]
    elif value > array[9]:
        return array[8],array[9]



def return_nearest(x,y):
    x1,x2 = find_nearest_spin(r136,x)
    y1,y2 = find_nearest_vel(Velocity,y)
    #print(x1,y1,x2,y2)
    return x1,y1,x2,y2

def Data(x,y):
    x1,x2 = find_nearest_spin(r136,x)
    y1,y2 = find_nearest_vel(Velocity,y)
    f11 = data_dict[str(y1)+','+str(x1)]
    f12 = data_dict[str(y1)+','+str(x2)]
    f21 = data_dict[str(y2)+','+str(x1)]
    f22 = data_dict[str(y2)+','+str(x2)]
    return f11,f12,f21,f22

def interpolate(x,y):
    x1,y1,x2,y2 = return_nearest(x,y)
    f11,f12,f21,f22 = Data(x,y)
    return (1/((x1-x2)*(y1-y2)))*(f11*(abs(x2-x))*(abs(y2-y)) + f12*(abs(x-x1))*(abs(y2-y)) + f21*(abs(x2-x))*(abs(y-y1)) + f22*(abs(x-x1))*(abs(y-y1)))

X = np.linspace(vel[0]+1,vel[-1]-1,100)
Y = np.linspace(Spin[0]+1,Spin[-1]-1,100)
xx,yy = np.meshgrid(X,Y)
zz = []

for i in range(len(X)):
    for j in range(len(Y)):
        zz.append(interpolate(Y[i],X[j]))

zz = np.array(zz)
zz = zz.reshape(len(X),len(Y))
xx1,yy1,zz1 = surface()

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
v136,l136 = extract_data('13.7.txt')
v216,l216 = extract_data('21.6.txt')
v299,l299 = extract_data('29.9.txt')
v384,l384 = extract_data('38.4.txt')
v459,l459 = extract_data('46.9.txt')
v552,l552 = extract_data('55.2.txt')
v631,l631 = extract_data('63.1.txt')
v719,l719 = extract_data('71.9.txt')
v802,l802 = extract_data('80.2.txt')
v889,l889 = extract_data('89.1.txt')


def plot_surface():
    fig = plt.figure(figsize=(14,6))
    gs = matplotlib.gridspec.GridSpec(1, 2,
         wspace=0.05, hspace=0.0, top=0.9, bottom=0.05, left=0.0, right=0.95) 
    ax = fig.add_subplot(gs[0,0], projection='3d')
    ax2 = fig.add_subplot(gs[0,1], projection='3d')
    ax.plot_surface(X = xx,Y = yy,Z=zz, cmap = cm.plasma, alpha=0.5)
    ax2.plot_surface(X = xx1,Y = yy1,Z=zz1, cmap = cm.plasma, alpha=0.5)
    ax.set_xlabel('Velocity / m/s',fontname="Arial", fontsize=14)
    ax.set_ylabel('Spin / rpm',fontname="Arial", fontsize=14)
    ax.set_zlabel('Cd',fontname="Arial", fontsize=14)
    ax2.set_xlabel('Velocity / m/s',fontname="Arial", fontsize=14)
    ax2.set_ylabel('Spin / rpm',fontname="Arial", fontsize=14)
    ax2.set_zlabel('Cl',fontname="Arial", fontsize=14)
    ax2.scatter(ys=v136,zs=l136,xs=[13.6]*6, s = 100)
    ax2.scatter(ys=v216,zs=l216,xs=[21.6]*6, s = 100)
    ax2.scatter(ys=v299,zs=l299,xs=[29.9]*6, s = 100)
    ax2.scatter(ys=v384,zs=l384,xs=[38.4]*6, s = 100)
    ax2.scatter(ys=v459,zs=l459,xs=[46.9]*6, s = 100)
    ax2.scatter(ys=v552,zs=l552,xs=[55.2]*6, s = 100)
    ax2.scatter(ys=v631,zs=l631,xs=[63.1]*6, s = 100)
    ax2.scatter(ys=v719,zs=l719,xs=[71.9]*6, s = 100)
    ax2.scatter(ys=v802,zs=l802,xs=[80.2]*6, s = 100)
    ax2.scatter(ys=v889,zs=l889,xs=[89.1]*6, s = 100)
    ax.scatter(ys=r136,zs=d136,xs=[13.6]*7, s = 100)
    ax.scatter(ys=r216,zs=d216,xs=[21.6]*7, s = 100)
    ax.scatter(ys=r299,zs=d299,xs=[29.9]*7, s = 100)
    ax.scatter(ys=r384,zs=d384,xs=[38.4]*7, s = 100)
    ax.scatter(ys=r459,zs=d459,xs=[45.9]*7, s = 100)
    ax.scatter(ys=r552,zs=d552,xs=[55.2]*7, s = 100)
    ax.scatter(ys=r631,zs=d631,xs=[63.1]*7, s = 100)
    ax.scatter(ys=r719,zs=d719,xs=[71.9]*7, s = 100)
    ax.scatter(ys=r802,zs=d802,xs=[80.2]*7, s = 100)
    ax.scatter(ys=r889,zs=d889,xs=[88.9]*7, s = 100)

    fig.savefig('CdClDataBearmanInterpolation.eps', format='eps', dpi=2000)
    plt.show()
plot_surface()