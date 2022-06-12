import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x_pos = []
y_pos = []
global iteration 
iteration = 0

"Need to assume that the force is only present for some time i.e. the connection time between the club and the ball. Lets assume 0.01 seconds. Ask Tim"

def get_components(force = 50, angle = 30):
    x = force*math.cos(angle)
    y = force*math.sin(angle)
    return x,y


def get_acceleration(mass = 0.043, force = 100, angle = 45):
    x_force, y_force = get_components(force = force, angle=angle)
    x_acc = x_force/mass
    y_acc = y_force/mass
    return x_acc , y_acc



def get_velocity(force = 100, angle = 45, time = 0.05):
    x,y = get_acceleration(force = force,angle = angle)
    x_vel = x*time
    y_vel = y*time
    total_vel = np.sqrt((x_vel**2 + y_vel**2))
    return x_vel , y_vel, total_vel

def plot_static_graph(input_force  = 100, input_angle = 0.5, input_gravity = 9.81):
    global iteration
    iteration = iteration+1
    y_position, x_position = analytical_with_out_drag(force = input_force, angle = input_angle, gravity = input_gravity)
    max_height = max(x_position)
    distance  = max(y_position )
    fig_name = 'fig'+str(iteration)
    fig_name = plt.figure()
    fig_name.suptitle('Ball Trajectory')
    ax_name = 'ax'+str(iteration)
    ax_name = fig_name.add_subplot(111, projection='3d')
    return ax_name.plot(xs = x_position,zs = y_position, ys = np.linspace(0,0,len(y_position))), max_height, distance


def analytical_with_out_drag(force, angle, gravity,dt = 0.01):
    x_vel , y_vel, velocity = get_velocity(force=force,angle=angle)
    flight_time = (y_vel*2)/9.81
    run = True
    time = 0 
    y_pos = []
    x_pos = []
    while run == True:
        y = velocity*np.cos(angle)*time-(0.5*gravity*time**2)
        if y < 0:
            run = False
            y=0
        y_pos.append(y)
        x_pos.append(time*velocity*np.sin(angle))
        time = time + dt
    return y_pos, x_pos    
    
def analytical_with_linear_drag(force = 0,angle = 0,gravity = 9.81,dt = 0.001):
    # Still needs work, need to double check all of the math in this function ##
    #x_vel , y_vel, velocity = get_velocity(force=force,angle=angle)
    x_vel = 50
    y_vel = 50
    run = True
    time = 0 
    mass = 0.043
    density = 1.225
    area = np.pi*0.02**2
    Cd = 10
    gamma = 0.001
    vt = mass*gravity/gamma
    y_pos = []
    x_pos = []
    while run == True:
        y = vt/gravity * (y_vel+vt)*(1-np.exp(-gravity*time/vt))-vt*time
        if y < 0:
            run = False
            y=0
        y_pos.append(y)
        x_pos.append((x_vel*vt/gravity)*(1-np.exp(-gravity*time/vt)))
        time = time + dt 
    return x_pos, y_pos    



def plot_static_graph_with_drag(input_force  = 100, input_angle = 0.5, input_gravity = 9.81):
    global iteration
    iteration = iteration+1
    x_position, y_position = analytical_with_linear_drag(force = input_force, angle = input_angle, gravity = input_gravity)
    max_height = max(x_position)
    distance  = max(y_position )
    fig_name = 'fig'+str(iteration)
    fig_name = plt.figure()
    fig_name.suptitle('Ball Trajectory')
    ax_name = 'ax'+str(iteration)
    ax_name = fig_name.add_subplot(111, projection='3d')
    return ax_name.plot(xs = x_position,zs = y_position, ys = np.linspace(0,0,len(y_position))), max_height, distance


def magnus_force(spin,velocity):
    pass

def rk4():
    mass = 0.043
    gamma = 0.001
    gravity = 9.81
    dt = 0.001
    x_pos = [0]
    y_pos = [0]
    x_vel = [50]
    y_vel = [50]

    for i in range(20000):
        if y_pos[i-1] < 0:
            break
        else:
            k1x = x_vel[i-1] 
            k1y = y_vel[i-1]
            x_acc = -gamma*k1x/mass
            y_acc = (1/mass)*(-gamma*k1y-mass*gravity)
            k2x = k1x + 0.5*dt*x_acc
            k2y = k1y + 0.5*dt*y_acc

            x_acc1 = -gamma*k2x/mass
            y_acc1 = (1/mass)*(-gamma*k2y-mass*gravity)
            k3x = x_vel[i-1] + 0.5*dt*x_acc1
            k3y = y_vel[i-1] + 0.5*dt*y_acc1
            
            x_acc2 = -gamma*k3x/mass
            y_acc2 = (1/mass)*(-gamma*k3y-mass*gravity)
            k4x = x_vel[i-1] + dt*x_acc2
            k4y = y_vel[i-1] + dt*y_acc2


            xh_acc2 = -gamma*k4x/mass
            yh_acc2 = (1/mass)*(-gamma*k4y-mass*gravity)

            velocity_x1 = x_vel[i-1] + dt*(x_acc+2*x_acc1+2*x_acc2+xh_acc2)*(1/6)
            velocity_y1 = y_vel[i-1] + dt*(y_acc+2*y_acc1+2*y_acc2+yh_acc2)*(1/6)
            x = x_pos[i-1] + dt*(k1x+2*k2x+2*k3x+k4x)*(1/6)
            y = y_pos[i-1] + dt*(k1y+2*k2y+2*k3y+k4y)*(1/6)

            x_pos.append(x)
            y_pos.append(y)
            x_vel.append(velocity_x1)
            y_vel.append(velocity_y1)
    return x_pos, y_pos

def intergrate_linear_drag(dt = 0.001, gravity = 9.81, force = 100, angle = 45):
        #Determine the intergartion of the function and return two arrays ##
        init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 100
        iterations = T/dt
        gamma = 0.001
        t = 0
        mass = 0.043
        x_pos = [0]
        y_pos = [0]
        y_vel = [init_vely]
        x_vel = [init_velx]
        for i in range(int(iterations)):
            if y_pos[i-1] < 0:
                break
            else:
                x_acc = -gamma*x_vel[i-1]/mass
                y_acc = (1/mass)*(-gamma*y_vel[i-1]-mass*gravity)
                velocity_x = x_vel[i-1]+dt*x_acc
                velocity_y = y_vel[i-1]+dt*y_acc
                x = x_pos[i-1] + dt*velocity_x
                y = y_pos[i-1] + dt*velocity_y
                x_pos.append(x)
                y_pos.append(y)
                x_vel.append(velocity_x)
                y_vel.append(velocity_y)
                t = t + dt
        return x_pos, y_pos