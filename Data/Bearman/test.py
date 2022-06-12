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
    #print(str(x)+'  :  '+'    X1:'+str(x1)+'     X2:'+str(x2))
    f11,f12,f21,f22 = Data(x,y)
    #print(str(x1)+'  :  '+'    X1:'+str(y1)+'     X2:'+str(f11))
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


def plot_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ys=r136,zs=d136,xs=[13.6]*6)
    #ax.plot_surface(xx,yy,zz)
    ax.scatter(ys=r216,zs=d216,xs=[21.6]*6)
    ax.scatter(ys=r299,zs=d299,xs=[29.9]*6)
    ax.scatter(ys=r384,zs=d384,xs=[38.4]*6)
    ax.scatter(ys=r459,zs=d459,xs=[45.9]*6)
    ax.scatter(ys=r552,zs=d552,xs=[55.2]*6)
    ax.scatter(ys=r631,zs=d631,xs=[63.1]*6)
    ax.scatter(ys=r719,zs=d719,xs=[71.9]*6)
    ax.scatter(ys=r802,zs=d802,xs=[80.2]*6)
    ax.scatter(ys=r889,zs=d889,xs=[88.9]*6)
    plt.show()
    
def plotly_plot():
    trace = go.Surface(x = xx, y = yy, z =zz )
    data = [trace]
    layout = go.Layout(title = '3D Surface plot')
    fig = go.Figure(data = data)
    fig.add_trace(
        go.Scatter3d(
        x = [21.6]*6, y = r216, z = d216,mode = 'markers', marker = dict(
        size = 10,
        )
    ))
    fig.add_trace(
        go.Scatter3d(
        x = [13.6]*6, y = r136, z = d136,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [29.9]*6, y = r299, z = d299,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [38.4]*6, y = r384, z = d384,mode = 'markers', marker = dict(
        size = 10,

        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [45.9]*6, y = r459, z = d459,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [55.2]*6, y = r552, z = d552,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [63.1]*6, y = r631, z = d631,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [71.9]*6, y = r719, z = d719,mode = 'markers', marker = dict(
        size = 10,
        )
    ))
    fig.add_trace(
        go.Scatter3d(
        x = [88.9]*6, y = r889, z = d889,mode = 'markers', marker = dict(
        size = 10,
        )
    ))

    fig.add_trace(
        go.Scatter3d(
        x = [45.9]*6, y = r459, z = d459,mode = 'markers', marker = dict(
        size = 10,
        )
    ))
    layout = go.Layout(title = '3D Scatter plot')
    fig.write_html('first_figure.html', auto_open=True)
xxx,yyy = extract_data('CdVelocity.txt')
zzz = np.polyfit(xxx, yyy, 12)
fff = np.poly1d(zzz)
YY = []
XX = []
plotly_plot()
bottomfit = np.polyfit(xxx[8:],yyy[8:],1)

def Bearman_Cd_func(i):
    if i > 69:
        new_y = bottomfit[0]*i + bottomfit[1]
    if i < 15:
        new_y = 0.5271980521935997
    elif 14<i<70:
        new_y = fff(i)
    return new_y