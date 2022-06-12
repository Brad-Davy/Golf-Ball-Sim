from Intergration_Methods_for_sweep import runge_kutta_4 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rk4 = runge_kutta_4()
X = []
Y = []

def sweep(velocity,windangle,windspeed=5,low_angle = 3,high_angle=40):
    X = []
    Y = []
    for angle in np.linspace(low_angle,high_angle,400):
        x_pos,y_pos,z_pos,xv,yv,zv,t,cd,cl,run = rk4.intergrate(total_velocity=velocity,angle=angle,WindVelocity=windspeed,WindAngle=windangle)
        #print(angle,x_pos[-1])
        X.append(angle)
        Y.append(x_pos[-1])
    return X[Y.index(max(Y))]
#l = sweep(55)

def single_plot(velocity,angle):
    fig = plt.figure()

    #ax = fig.add_subplot(111, projection='3d')
    x_pos,y_pos,z_pos,xv,yv,zv,t,cd,cl,run = rk4.intergrate(total_velocity=velocity,angle=angle,dt = 0.01,WindAngle=90,WindVelocity=10)
    plt.plot(t,w,color = 'black')
    plt.grid()
    plt.xlabel('Time / s')
    plt.ylabel('Wind Speed / m/s')

    #ax.plot(xs=z_pos, ys=x_pos, zs = y_pos,label = '90$^\circ$')
    #x_pos,y_pos,z_pos,xv,yv,zv,t,cd,cl,run = rk4.intergrate(total_velocity=velocity,angle=angle,dt = 0.001,WindAngle=0,WindVelocity=0)
    #ax.plot(xs=z_pos, ys=x_pos, zs = y_pos,label = '0$^\circ$')
    #ax.set_xlabel('Distance / m',fontname="Arial", fontsize=14)
    #ax.set_zlabel('Height / m',fontname="Arial", fontsize=14)
    #ax.legend()
    plt.show()
    fig.savefig('WindPlot.eps', format='eps', dpi=2000)
#single_plot(65,15)

def angle_comparison():
    fig = plt.figure(figsize=(10,7))
    for angle in range (10,15,1):
        X = []
        Y = []
        for velocity in range(45,70,4):
            x,y,z,xv,yv,zv,t,cd,cl,r = rk4.intergrate(total_velocity=velocity,angle=angle,dt = 0.001)
            X.append(t)
            Y.append(x[-1])
        plt.plot(X,Y,label = str(angle))
    #plt.style.use('seaborn')
    plt.grid()
    plt.xlabel('Velocity at launch / m/s',fontname="Arial", fontsize=14)
    plt.ylabel('Distance / m',fontname="Arial", fontsize=14)
    i = [46.76852415687393, 49.66315984953509, 51.16791694198636, 52.09299111635119, 55.474348841929796, 55.45124358127222, 55.474348841929796, 56.38182414953207, 56.77310202160234, 58.1863018194853, 58.78952187226031, 60.387239849880594, 61.890917257254415, 63.07263257453067, 63.396106223736666, 66.31341530302443, 67.09802244881215, 67.07513312517005, 67.07513312517005, 68.32681203546551] 
    j = [116.20769029449485, 129.00164492180915, 134.75719679203965, 136.45845017685238, 152.23927341513024, 152.2394708679372, 152.23927341513024, 164.8365650467072, 161.8421932290789, 169.30768640762858, 174.21636318878507, 181.0393449334698, 184.65845743233612, 191.2713493904098, 191.2685850511123, 203.20776647707396, 201.2782576474094, 201.70574297448917, 201.70574297448917, 209.59990619696046]
    plt.scatter(i,j,label = 'Williams')
    plt.legend()
    plt.show()
    #fig.savefig('AngleComparison.eps', format='eps', dpi=2000)
#angle_comparison()

def slice_plot(spinangle):
    x_pos,y_pos,z_pos,xv,yv,zv,t,cd,cl,run = rk4.intergrate(total_velocity=55,angle=10,dt = 0.001,SpinAngle = spinangle)
    return x_pos,y_pos,z_pos

def slice_plot1():
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('equal')
    ax.set_xlim3d(0,300)
    ax.set_ylim3d(-10,10)
    ax.set_zlim3d(0,50)
    x,y,z = slice_plot(-20)
    print(x[-1])
    ax.plot(xs = x,ys = z,zs = y,label = '-20')
    x,y,z = slice_plot(-10)
    print(x[-1])
    ax.plot(xs = x,ys = z,zs = y,label = '-10')
    x,y,z = slice_plot(0)
    print(x[-1])
    ax.plot(xs = x,ys = z,zs = y,label = '-5')
    x,y,z = slice_plot(10)
    print(x[-1])
    ax.plot(xs = x,ys = z,zs = y,label = '0')
    x,y,z = slice_plot(20)
    print(x[-1])
    ax.plot(xs = x,ys = z,zs = y,label = '20')
    plt.legend()
    fig.savefig('SliceComparison.eps', format='eps', dpi=2000)
    plt.show()


def optimum_loft(windspeed):
    loft_data = open('LoftData.txt','a')
    angle = []
    velocity = []
    for j in range(30,61,1):
        A = sweep(velocity = j,windangle=windspeed)
        angle.append(A)
        velocity.append(j)
        print(j,A)
    loft_data.close()
    plt.plot(velocity,angle)
    plt.show()
optimum_loft(270)



def plot_optimum_loft():
    fig = plt.figure()
    X = []
    Y = []
    loft_data = open('LoftData.txt').read().split('\n')
    for lines in loft_data:
        try:
            X.append(float(lines.split(' ')[0]))
            Y.append(float(lines.split(' ')[1]))
        except:
            pass
    ## 10 degree ##
    xx = [30.0, 31.0, 32.0, 33.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 55.0, 56.0, 57.0, 58.0] 
    yy = [18.49367088607595, 17.848101265822784, 17.632911392405063, 16.9873417721519, 16.126582278481013, 15.69620253164557, 15.265822784810126, 14.835443037974684, 14.620253164556962, 14.18987341772152, 13.759493670886076, 13.544303797468356, 13.113924050632912, 12.898734177215191, 12.468354430379748, 12.037974683544304, 11.822784810126583, 11.39240506329114, 11.177215189873419, 10.962025316455698, 10.531645569620252, 10.316455696202532, 10.10126582278481, 9.670886075949367, 9.240506329113924, 9.025316455696203, 8.810126582278482]
    x = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0]
    y = [19.164556962025316, 18.759493670886076, 18.354430379746837, 17.949367088607595, 17.341772151898734, 16.936708860759495, 16.531645569620252, 16.126582278481013, 15.721518987341772, 15.40506329113924, 14.91139240506329, 14.746835443037973, 14.417721518987342, 14.088607594936708, 13.759493670886076, 13.430379746835442, 12.936708860759493, 12.60759493670886, 12.443037974683545, 12.11392405063291, 11.784810126582279, 11.455696202531644, 11.126582278481013, 10.962025316455696, 10.632911392405063, 10.30379746835443, 10.139240506329113, 9.974683544303797, 9.645569620253164, 9.481012658227847, 9.151898734177216]
    plt.plot(x,y,color = 'black',label = '$0^\degree$')
    plt.plot(xx,yy,color = 'red', label = '$10^\degree$')
    plt.plot(X,Y)
    summ = 0
    for i in range(len(X)):
        summ = summ + abs(y[i]-Y[i])
    print(summ/len(X))
    plt.grid()
    plt.ylabel('Loft / $\degree$')
    plt.xlabel('Club head velocity / $m/s$')
    plt.legend()
    plt.show()
    fig.savefig('SliceComparison.eps', format='eps', dpi=2000)
#plot_optimum_loft()

#optimum_loft()
def gamma_plot(gamma):
    n = []
    T = []
    percent25 = []
    t = 0
    gamma = gamma
    m = 0.0459
    r = 0.0213
    n_0 = 2500
    dt = 0.01
    while True:
        n.append(n_0*np.exp(gamma*t*5/2*m*r))
        percent25.append(n_0*0.75)
        t = t +dt
        T.append(t)
        if t > 10:
            break
    plt.plot(T,n,label = str(gamma))
