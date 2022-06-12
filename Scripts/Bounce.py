## This script will deal with how the ball bounces ##
import matplotlib.pyplot as plt
import numpy as np 
from Intergration_Methods import runge_kutta_4 as RK4

def reduce_kinetic(init_kinetic, efficency):
    print (init_kinetic)
    print (init_kinetic*efficency)
    return init_kinetic*efficency

def velocity(kinetic_energy):
    return np.sqrt(2/0.043)

def total_vel(x,y):
    return np.sqrt(x**2+y**2)

def kinetic_energy_converter(vel):
    return 0.5*0.043*vel**2

rk4 = RK4()
y_array = []
x_array = []

def bounce(n,vel_array = [0,0],angle = 45,efficency = 0.5,r = 0.023,spin = 100, dx=1, dy=1):
    vel_array[0] = abs(vel_array[0])
    vel_array[1] = abs(vel_array[1])
    for t in range(n):
        efficency = 0.510 - 0.0375*abs(vel_array[0]) + 0.000903*abs((vel_array[0])**2)
        if vel_array[0] > 20:
            efficency = 0.1
        vbry = vel_array[0]*efficency
        vbrx = (5*vel_array[1] - 2*r*spin)/7
        angle = np.arctan(dy/dx)
        x,y,xv,yv,spin = rk4.intergrate(total_velocity = (vbry**2 + vbrx**2)**0.5, angle = angle,dt = 0.001, bounce = True)
        if t == 0:
            y_array = y
            x_array = x
            vel_array[0] = vbry
            vel_array[1] = vbrx
        else:
            y_array = y_array + y
            for i in range(len(x)):
                x[i] = x_array[-1] + abs(x[i])
            x_array = x_array + x
            vel_array[0] = vbry
            vel_array[1] = vbrx
    return x_array, y_array, vel_array[0]

def add_arrays(bounce_array,shot_array):
    for i in range(len(bounce_array)):
        bounce_array[i] = shot_array[-1] + abs(bounce_array[i])
    return shot_array+bounce_array


