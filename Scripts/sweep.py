## sweeps function and plots max distance for each func1tion ##
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
from Intergration_Methods_for_sweep import runge_kutta_4
rk4 = runge_kutta_4()
distances = []
plot_X = []

def sweep(velocity):
    for x in range(301):
        x = x/10
        x_pos, y_pos, xv, yv,tot = rk4.intergrate(total_velocity = velocity ,angle = x)
        distances.append(x_pos[-1])
        print(x,x_pos[-1])
        plot_X.append(x)
    return distances

distances = sweep(45)
distances = np.array(distances)
print(distances.argmax()/10,distances[distances.argmax()])
plt.grid()
plt.xlabel('Angle/degrees')
plt.ylabel('Max distnace / M')
plt.style.use('seaborn')
plt.plot(plot_X,distances, color = 'black')
plt.show()