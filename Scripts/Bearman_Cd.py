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
    filename = '../Data/Bearman/cd/' + filename
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
r100 = r889
d100 = []
for elements in d889:
    d100.append(elements*0.9)
## Combine data ##
vel = [13.6]*7 + [21.6]*7 + [29.9]*7 + [38.4]*7 + [45.9]*7 + [55.2]*7 + [63.1]*7 + [71.9]*7 + [80.2]*7 + [88.9]*7 + [100]*7
Spin = r136 + r216 + r299 + r384 + r459 + r552 + r631 + r719 + r802 + r889 + r100
Drag = d136 + d216 + d299 + d384 + d459 + d552 + d631 + d719 + d802 + d889 + d100

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

def BearmanCdVelSpin(velocity,spin):
    return polyval2d(float(velocity),float(spin),f)

def plot_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ys=r136,zs=d136,xs=[13.6]*7)
    #ax.plot_surface(xx,yy,zz)
    ax.scatter(ys=r216,zs=d216,xs=[21.6]*7)
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
        x = [21.6]*7, y = r216, z = d216,mode = 'markers', marker = dict(
        size = 10,
        )
    ))
    fig.add_trace(
        go.Scatter3d(
        x = [13.6]*7, y = r136, z = d136,mode = 'markers', marker = dict(
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


bottomfit = np.polyfit(xxx[8:],yyy[8:],1)

def Bearman_Cd_func(i):
    if i > 69:
        new_y = bottomfit[0]*i + bottomfit[1]
    if i < 15:
        new_y = 0.5271980521935997
    elif 14<i<70:
        new_y = fff(i)
    return new_y
