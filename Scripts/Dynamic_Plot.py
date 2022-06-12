## This script will make a dynamic plot of the golf ball in 3D ##
import Components as cp
import matplotlib.pyplot as plt
import numpy as np
from Intergration_Methods_for_sweep import runge_kutta_2 as RK2, runge_kutta_4 as RK4
import plotly
import plotly.graph_objs as go
from Bounce import bounce, add_arrays

def main(total_velocity = 55, angle = 1, gravity = 9.81, WindAngle = 0, WindVelocity = 0,slice_angle = 0):
    rk4 = RK4()
    x_pos, y_pos, z_pos, x_vel, y_vel, z_vel ,total_velocity , CD, CL, run = rk4.intergrate(dt = 0.05, gravity = gravity, total_velocity = float(total_velocity), angle = float(angle), WindAngle=WindAngle, WindVelocity=WindVelocity,SpinAngle=5)
    print(x_pos,y_pos,z_pos)
    #fig = go.Figure(data=[go.Scatter3d(x=x_pos, y=z_pos, z=y_pos,mode='markers')])
    #layout = go.Layout(title = '3D Scatter plot')
    #fig.update_layout(scene_aspectmode='data')
    #fig.write_html('firstfigure.html', auto_open=True)
    # Helix equation
    for position in z_pos:
        if position < 0.001:
            z_pos[z_pos.index(position)] = 0

    fig = go.Figure(data=[go.Scatter3d(x=x_pos, y=z_pos, z=y_pos, mode='markers')])
    fig.show()
    max_y = max(y_pos)
    max_x = max(x_pos)
    return max_x, max_y