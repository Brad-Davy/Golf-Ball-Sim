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
from Bearman_CL import BearmanClVelSpin

def extract_data(filename):
    velocity = []
    Cd =[]
    filename = '../Data/Bearman/Cl/' + filename
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
vel = [13.6]*6 + [21.6]*6 + [29.9]*6 + [38.4]*6 + [45.9]*6 + [55.2]*6 + [63.1]*6 + [71.9]*6 + [80.2]*6 + [88.9]*6
Spin = r136*10
Velocity = [13.6,21.6,29.9,38.4,45.9,55.2,63.1,71.9,80.2,88.9]
Drag = d136 + d216 + d299 + d384 + d459 + d552 + d631 + d719 + d802 + d889 
r136.sort()
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
    elif value > array[5]:
        return array[4],array[5]

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
    return x1,y1,x2,y2

def Data(x,y):
    x1,x2 = find_nearest_spin(r136,x)
    y1,y2 = find_nearest_vel(Velocity,y)
    f11 = data_dict[str(y1)+','+str(x1)]
    f12 = data_dict[str(y1)+','+str(x2)]
    f21 = data_dict[str(y2)+','+str(x1)]
    f22 = data_dict[str(y2)+','+str(x2)]
    return f11,f12,f21,f22

def interpolate_cl(x,y):
    if x > 6200:
        return BearmanClVelSpin(y,x)
    x1,y1,x2,y2 = return_nearest(x,y)
    f11,f12,f21,f22 = Data(x,y)
    return (1/((x1-x2)*(y1-y2)))*(f11*(abs(x2-x))*(abs(y2-y)) + f12*(abs(x-x1))*(abs(y2-y)) + f21*(abs(x2-x))*(abs(y-y1)) + f22*(abs(x-x1))*(abs(y-y1)))
