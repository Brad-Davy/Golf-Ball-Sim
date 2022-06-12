from Intergration_Methods_for_sweep import runge_kutta_4
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
from scipy.optimize import curve_fit

rk4 = runge_kutta_4()
X = []
xx = []
xxx = []
yyy = []
#x,y,xv,yv,t = rk4.intergrate(total_velocity=55,angle=3,dt = 0.001)
#plt.plot(x,y)

for t in range(0,90,1):
        x,y,xv,yv,k = rk4.intergrate(total_velocity=55,angle=t,dt = 0.001)
        if x[-1] < 0:
                X.append(0)
        else:
                X.append(x[-1])
        print(t,x[-1])
        xx.append(t)

#plt.scatter([30,40,50,60],[83.198,116.02,137.4,154])
plt.plot(xx,X,color = 'black')
plt.grid()
plt.xlabel('Angle/degrees')
plt.ylabel('Max distnace / M')
plt.style.use('seaborn')
plt.legend()
plt.show()