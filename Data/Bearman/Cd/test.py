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
from matplotlib import cm
from import_this import return_this
import matplotlib
data_dict = {}
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
r100 = r889
d100 = []
for elements in d889:
    d100.append(elements*0.9)
## Combine data ##
velocity = [13.6,21.6,29.9,38.4,45.9,55.2,63.1,71.9,80.2,88.9]
vel = [13.6]*7 + [21.6]*7 + [29.9]*7 + [38.4]*7 + [45.9]*7 + [55.2]*7 + [63.1]*7 + [71.9]*7 + [80.2]*7 + [88.9]*7 
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
def pull_fit_value(velo,spin):
    velo = float(velo)
    spin = float(spin)
    z = polyval2d(np.array([velo]),np.array([spin]),f)
    return float(z)

for d in range(len(vel)):
    data_dict.update({str(vel[d])+','+str(Spin[d]):Drag[d]})

def pull_real_value(veloc,spinn):
    return float(data_dict[str(veloc)+','+str(spinn)])

d = []
for element in data_dict:
    diff = pull_real_value(float(element.split(',')[0]),float(element.split(',')[1])) - pull_fit_value(float(element.split(',')[0]),float(element.split(',')[1]))
    diff = diff**2
    d.append(diff)
print(sum(d))




X = np.linspace(vel[0]+1,vel[-1]-1,100)
Y = np.linspace(Spin[0]+1,Spin[-1]-1,100)
xx,yy = np.meshgrid(X,Y)
zz = polyval2d(xx,yy,f)
xx1,yy1,zz1 = return_this()


def BearmanCdVelSpin(velocity,spin):
    return polyval2d(float(velocity),float(spin),f)


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
    gs = matplotlib.gridspec.GridSpec(1, 2,wspace=0.05, hspace=0.0, top=0.9, bottom=0.05, left=0.0, right=0.95) 
    ax = fig.add_subplot(gs[0,0], projection='3d')
    ax2 = fig.add_subplot(gs[0,1], projection='3d')
    ax2.set_xlabel('Velocity / m/s',fontname="Arial", fontsize=14)
    ax2.set_ylabel('Spin / rpm',fontname="Arial", fontsize=14)
    ax2.set_zlabel('Cl',fontname="Arial", fontsize=14)
    ax2.plot_surface(X = xx1,Y = yy1,Z=zz1, cmap = cm.plasma, alpha=0.01)
    ax.plot_surface(X = xx,Y = yy,Z=zz, cmap = cm.plasma, alpha=0.011)
    ax.set_xlabel('Velocity / m/s',fontname="Arial", fontsize=14)
    ax.set_ylabel('Spin / rpm',fontname="Arial", fontsize=14)
    ax.set_zlabel('Cd',fontname="Arial", fontsize=14)
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
    fig.savefig('CdClDataBearmanPolynomialFit.eps', format='eps', dpi=2000)
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
#plotly_plot()