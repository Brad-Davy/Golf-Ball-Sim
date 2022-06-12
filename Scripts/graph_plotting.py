import numpy as np 
import matplotlib.pyplot as plt 
from Intergration_Methods_for_sweep import runge_kutta_4 as RK4
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
Xx = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0]
Yy = [19.164556962025316, 18.759493670886076, 18.354430379746837, 17.949367088607595, 17.341772151898734, 16.936708860759495, 16.531645569620252, 16.126582278481013, 15.721518987341772, 15.40506329113924, 14.91139240506329, 14.746835443037973, 14.417721518987342, 14.088607594936708, 13.759493670886076, 13.430379746835442, 12.936708860759493, 12.60759493670886, 12.443037974683545, 12.11392405063291, 11.784810126582279, 11.455696202531644, 11.126582278481013, 10.962025316455696, 10.632911392405063, 10.30379746835443, 10.139240506329113, 9.974683544303797, 9.645569620253164, 9.481012658227847, 9.151898734177216]

def sweep(velocity,windspeed):
    rk4 = RK4()
    X = []
    J = []
    for j in range(3,60):
        x_pos, y_pos, _, _, _,_, _,_, _,_ = rk4.intergrate(total_velocity = velocity, angle = j, WindVelocity = windspeed)
        X.append(x_pos[-1])
        J.append(j)
    return X,J
'''
fig = plt.figure()
plt.xlabel('Angle / Degrees')
plt.ylabel('Distance / m')
x,y = sweep(55,12)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '12 m/s')
x,y = sweep(55,10)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '10 m/s')
x,y = sweep(55,5)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '5 m/s')
x,y = sweep(55,0)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '0 m/s')
x,y = sweep(55,-5)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '-5 m/s')
x,y = sweep(55,-10)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '-10 m/s')
x,y = sweep(55,-12)
print(max(x),y[x.index(max(x))])
plt.plot(y,x,label = '-12 m/s')
plt.legend()
plt.grid()
fig.savefig('WindComparison.eps', format='eps', dpi=2000)
plt.show()

fig1 = plt.figure()
plt.xlabel('Wind Speed / m/s')
plt.ylabel('Optimum angle / Degrees')
plt.scatter([-12,-10,-5,0,5,10,12],[14.5,14,11,10,7,6,5])
plt.grid()
fig1.savefig('WindComparisonPlot.eps', format='eps', dpi=2000)
plt.show()
'''











def plot_optimum_loft():
    x = []
    y = []
    loft_data = open('LoftData.txt').read().split('\n')
    for lines in loft_data:
        print(lines)
        try:
            x.append(float(lines.split(' ')[0]))
            y.append(float(lines.split(' ')[1]))
        except:
            pass
    fig = plt.figure()
    plt.plot(x,y,label = 'Model', color = 'black')
    plt.plot(X,Y,label = 'Penner', color = 'red')
    plt.legend()
    plt.xlabel('Velocity of Club head / m/s')
    plt.ylabel('Angle / Degrees')
    plt.grid()
    fig.savefig('OptimumAngle.eps', format='eps', dpi=2000)
    plt.show()


def plot_cd(velocity,angle, torque_gamma):
    rk4 = RK4()
    x_pos, y_pos, z_pos, x_vel, y_vel, z_vel ,total_velocity , CD, CL, run = rk4.intergrate(total_velocity=velocity,angle=angle, torque_gamma=torque_gamma)
    del x_pos[0]
    del CD[0]
    return x_pos,CD

def plot_cl(velocity,angle,torque_gamma,dynamic_air):
    rk4 = RK4()
    x_pos, y_pos, z_pos, x_vel, y_vel, z_vel ,total_velocity , CD, CL, run = rk4.intergrate(total_velocity=velocity,angle=angle,torque_gamma=torque_gamma, dynamic_air=dynamic_air)
    for item in y_pos: 
        if item < 1: 
           index = y_pos.index(item)
           print(y_pos[index])
           y_pos.remove(item)
           del x_pos[index]
    return x_pos,y_pos

def generic_plots():
    fig = plt.figure(figsize = (14,4))
    plt.gca().set_aspect('equal', adjustable='box')
    x,y = plot_cl(55,10,12.5, dynamic_air=True)
    plt.plot(x,y,'r--',label = '$Dynamic$')#,'r--',label = 'Cd')
    x,y = plot_cl(55,10,12.5, dynamic_air = False)
    plt.plot(x,y,'b--',label = '$Fixed$')#,'r--',label = 'Cd')
    #x,y = plot_cl(55,10,30)
    #plt.plot(x,y,color = 'blue',label = '$\gamma_b = -30$')
    #x,y = plot_cd(55,10,30)
    #plt.plot(x,y,'b--',label = '$\gamma_b = -30$')#,'b--', label = 'Cl')
    plt.xlabel('Distance / m')
    plt.ylabel('Height / m')
    plt.grid()
    plt.legend()
    fig.savefig('10GammaTrajectoryPlot.eps', format='eps', dpi=2000)
    plt.show()

#fig = plt.figure(figsize=(14,6))
#gs = matplotlib.gridspec.GridSpec(1, 2,wspace=0.05, hspace=0.0, top=0.9, bottom=0.05, left=0.0, right=0.95) 
#ax = fig.add_subplot(gs[0,0], projection='3d')
#ax2 = fig.add_subplot(gs[0,1], projection='3d')
##
def big_3d_plot(velocity,color, type):
    rk4 = RK4()
    for x in range(3,60):
        angle_array = []
        x_pos, y_pos, z_pos, x_vel, y_vel, z_vel ,total_velocity , CD, CL, run = rk4.intergrate(total_velocity=velocity,angle=x)
        for j in range(len(x_pos)):
            angle_array.append(x)
        if type == True:
            ax.plot(xs=angle_array,ys=x_pos,zs=y_pos)#,color = color)
            ax.set_xlabel('Angle / Degrees')
            ax.set_ylabel('Distance / m')
            ax.set_zlabel('Height / m')
        if type == False:
            ax2.plot(xs=angle_array,ys=x_pos,zs=y_pos)#,color = color)
            ax2.set_xlabel('Angle / Degrees')
            ax2.set_ylabel('Distance / m')
            ax2.set_zlabel('Height / m')

#big_3d_plot(35,'r',True)
#big_3d_plot(65,'r',False)
#plt.show()
#fig.savefig('3dsweep.eps', format='eps', dpi=2000)

def air_dens():
    rho = []
    height = []
    rho1 =[]
    for j in range(100):
        height.append(j)
        p0  = 101325
        T0  = 288.15
        g   = 9.81
        L   = 0.0065
        R   = 8.31447
        M   = 0.0289654
        T   = T0 - L*j
        p   = p0 * ( (1-(L*j/T0) )**((g*M)/(R*L)) )
        r = (p*M)/(R*T)
        rho.append(r)
        P = p0 * np.exp(-(M*g*j)/(R*T))
        rho1.append((P*M)/(R*T))



    fig = plt.figure()
    plt.plot(height,rho1, color='black')
    plt.xlabel('Height (z) / m')
    plt.ylabel('Air density / $kg/m^3$')
    plt.grid()
    fig.savefig('Airdensity.eps', format='eps', dpi=2000)
    plt.show()
    
ten = '''30 19.69172932330827
31 18.857142857142854        
32 18.57894736842105
33 18.30075187969925
34 18.115288220551378        
35 17.466165413533833        
36 17.18796992481203
37 17.00250626566416
38 16.44611528822055
39 16.260651629072683        
40 15.889724310776941        
41 15.611528822055137        
42 15.240601503759398        
43 14.869674185463658
44 14.498746867167918
45 14.220551378446114        
46 13.849624060150376        
47 13.571428571428571        
48 13.200501253132831        
49 12.922305764411027        
50 12.644110275689222        
51 12.273182957393484        
52 11.99498746867168
53 11.809523809523808        
54 11.531328320802004        
55 11.2531328320802
56 11.06766917293233
57 10.882205513784461        
58 10.511278195488721        
59 10.418546365914786        
60 10.140350877192983'''


five = '''30 18.115288220551378
31 17.55889724310777
32 17.280701754385966
33 16.724310776942353
34 16.167919799498748
35 15.704260651629072
36 15.333333333333332
37 15.147869674185463
38 14.776942355889723
39 14.31328320802005
40 13.94235588972431
41 13.664160401002505
42 13.385964912280702
43 13.015037593984962
44 12.551378446115288
45 12.273182957393484
46 12.087719298245613
47 11.62406015037594
48 11.43859649122807
49 10.974937343358395
50 10.789473684210526
51 10.418546365914786
52 10.325814536340852
53 9.954887218045112
54 9.676691729323307
55 9.583959899749374
56 9.305764411027567
57 9.027568922305765
58 8.93483709273183
59 8.656641604010025
60 8.471177944862156'''

minus_five = '''30 17.651629072681704
31 17.00250626566416
32 16.538847117794486
33 15.982456140350877
34 15.611528822055137
35 15.147869674185463
36 14.776942355889723
37 14.406015037593985
38 13.94235588972431
39 13.571428571428571
40 13.200501253132831
41 12.922305764411027
42 12.551378446115288
43 12.180451127819548
44 11.99498746867168
45 11.62406015037594
46 11.2531328320802
47 11.06766917293233
48 10.696741854636592
49 10.511278195488721
50 10.233082706766917
51 9.954887218045112
52 9.676691729323307
53 9.491228070175438
54 9.305764411027567
55 9.027568922305765
56 8.842105263157894
57 8.563909774436091
58 8.37844611528822
59 8.285714285714285
60 8.100250626566416'''


minus_ten = '''30 18.115288220551378   
31 17.55889724310777    
32 17.280701754385966   
33 16.724310776942353   
34 16.167919799498748   
35 15.704260651629072        
36 15.333333333333332        
37 15.147869674185463        
38 14.776942355889723        
39 14.31328320802005
40 13.94235588972431
41 13.664160401002505        
42 13.385964912280702        
43 13.015037593984962
44 12.551378446115288
45 12.273182957393484        
46 12.087719298245613        
47 11.62406015037594
48 11.43859649122807
49 10.974937343358395        
50 10.789473684210526        
51 10.418546365914786        
52 10.325814536340852        
53 9.954887218045112
54 9.676691729323307
55 9.583959899749374
56 9.305764411027567
57 9.027568922305765
58 8.93483709273183
59 8.656641604010025
60 8.471177944862156'''


def pull_data(string):
    X = []
    Y = []
    for elements in string.split('\n'):
        X.append(float(elements.split(' ')[0]))
        Y.append(float(elements.split(' ')[1]))
    return X,Y




def plot_that_graph(string,label):
    x,y = pull_data(string)
    plt.plot(x,y,label = label)


fig = plt.figure()
plot_that_graph(ten,'0$^\circ$')
plot_that_graph(five,'90$^\circ$')
plt.plot(Xx,Yy, label = 'No Wind')
plot_that_graph(minus_five,'180$^\circ$')
plot_that_graph(minus_ten,'270$^\circ$')
plt.xlabel('Club Head Velocity / m/s')
plt.ylabel('Loft / $^\circ$')
plt.legend()
plt.grid()
fig.savefig('WindAngleComaprison.eps', format='eps', dpi=2000)
plt.show()