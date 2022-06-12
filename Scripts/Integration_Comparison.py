## Test Integration methods ##
import numpy as np 
import matplotlib.pyplot as plt
def analytical_with_linear_drag(force = 0,angle = 0,gravity = 9.81,dt = 0.001):
    # Still needs work, need to double check all of the math in this function ##
    #x_vel , y_vel, velocity = get_velocity(force=force,angle=angle)
    x_vel = 50
    y_vel = 50
    run = True
    time = 0 
    mass = 0.043
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
        if y_pos[i] < 0:
            break
        else:
            k1x = x_vel[i] 
            k1y = y_vel[i]
            x_acc = -gamma*k1x/mass
            y_acc = (1/mass)*(-gamma*k1y-mass*gravity)
            k2x = k1x + 0.5*dt*x_acc
            k2y = k1y + 0.5*dt*y_acc

            x_acc1 = -gamma*k2x/mass
            y_acc1 = (1/mass)*(-gamma*k2y-mass*gravity)
            k3x = x_vel[i] + 0.5*dt*x_acc1
            k3y = y_vel[i] + 0.5*dt*y_acc1
            
            x_acc2 = -gamma*k3x/mass
            y_acc2 = (1/mass)*(-gamma*k3y-mass*gravity)
            k4x = x_vel[i] + dt*x_acc2
            k4y = y_vel[i] + dt*y_acc2


            xh_acc2 = -gamma*k4x/mass
            yh_acc2 = (1/mass)*(-gamma*k4y-mass*gravity)

            velocity_x1 = x_vel[i] + dt*(x_acc+2*x_acc1+2*x_acc2+xh_acc2)*(1/6)
            velocity_y1 = y_vel[i] + dt*(y_acc+2*y_acc1+2*y_acc2+yh_acc2)*(1/6)
            x = x_pos[i] + dt*(k1x+2*k2x+2*k3x+k4x)*(1/6)
            y = y_pos[i] + dt*(k1y+2*k2y+2*k3y+k4y)*(1/6)
            
            x_pos.append(x)
            y_pos.append(y)
            x_vel.append(velocity_x1)
            y_vel.append(velocity_y1)
    y_pos[-1] = 0
    return x_pos, y_pos

def intergrate_linear_drag(dt = 0.001, gravity = 9.81, force = 100, angle = 45):
        #Determine the intergartion of the function and return two arrays ##
        #init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        x_vel = [50]
        y_vel = [50]
        T = 100
        iterations = T/dt
        gamma = 0.001
        t = 0
        mass = 0.043
        x_pos = [0]
        y_pos = [0]
        #y_vel = [init_vely]
        #x_vel = [init_velx]
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
        y_pos[-1] = 0
        return x_pos[::2], y_pos[::2]

def intergrate_no_drag( dt = 0.001, gravity = 9.81, force = 100, angle = 45):
        #Determine the intergartion of the function and return two arrays ##
        #init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 1000
        iterations = T/dt
        t = 0
        gamma = 0.001
        mass = 0.043
        x_pos = [0]
        y_pos = [0]
        x_vel = [50]
        y_vel = [50]
        #y_vel = [init_vely]
        #x_vel = [init_velx]
        for i in range(int(iterations)):
            if y_pos[i-1] < 0:
                break
            else:
                x_acc = -gamma*x_vel[i-1]/mass
                y_acc = (1/mass)*(-gamma*y_vel[i-1]-mass*gravity)
                vxh = x_vel[i-1] + 0.5*dt*x_acc
                vyh = y_vel[i-1] + 0.5*dt*y_acc
                xh_acc = -gamma*vxh/mass
                yh_acc = (1/mass)*(-gamma*vyh-mass*gravity)
                velocity_x = x_vel[i-1] + dt*xh_acc
                velocity_y = y_vel[i-1] + dt*yh_acc
                x = x_pos[i-1] + dt*vxh
                y = y_pos[i-1] + dt*vyh
                x_pos.append(x)
                y_pos.append(y)
                x_vel.append(velocity_x)
                y_vel.append(velocity_y)
                t = t + dt
        y_pos[-1] = 0
        return x_pos[::2], y_pos[::2]

def new_rk4( dt = 0.001, gravity = 9.81, force = 100, angle = 45, rho = 1.225):

        #Determine the intergartion of the function and return two arrays ##
        

        # Set appropriate constants for the integrator #
        t = 0
        T = []
        mass = 0.043
        gamma = 0.001
        x_pos = [0]
        y_pos = [0]
        x_vel = [50]
        y_vel = [50]
        
        # Begin loop for the runge kutta method #
        for i in range(20000):
            t = t + dt
            # Breaks loop if the Y values goes past 0 as this is when the ball bounces #
            if y_pos[i-1] < 0:
                break
            else:

                # Determines the K1 constant of the runge kutta method #
                k1x = x_vel[i-1] 
                k1y = y_vel[i-1]
                l1x = k1x*dt
                l1y = k1y*dt
                x_acc = -gamma*k1x/mass
                y_acc = (1/mass)*(-gamma*k1y-mass*gravity)

                # Determines the K2 constant of the runge kutta method #
                k2x = k1x + 0.5*dt*x_acc
                k2y = k1y + 0.5*dt*y_acc
                l2x = k2x*dt
                l2y = k2y*dt
                x_acc1 = -gamma*k2x/mass
                y_acc1 = (1/mass)*(-gamma*k2y-mass*gravity)

                # Determines the K3 constant of the runge kutta method #
                k3x = x_vel[i-1] + 0.5*dt*x_acc1
                k3y = y_vel[i-1] + 0.5*dt*y_acc1
                l3x = k3x*dt
                l3y = k3y*dt
                x_acc2 = -gamma*k3x/mass
                y_acc2 = (1/mass)*(-gamma*k3y-mass*gravity)

                # Determines the K4 constant of the runge kutta method #
                k4x = x_vel[i-1] + dt*x_acc2
                k4y = y_vel[i-1] + dt*y_acc2
                l4x = k4x*dt 
                l4y = k4y*dt
                xh_acc2 = -gamma*k4x/mass
                yh_acc2 = (1/mass)*(-gamma*k4y-mass*gravity)

                # Update position and velocity #
                velocity_x1 = x_vel[i-1] + dt*(x_acc+2*x_acc1+2*x_acc2+xh_acc2)*(1/6)
                velocity_y1 = y_vel[i-1] + dt*(y_acc+2*y_acc1+2*y_acc2+yh_acc2)*(1/6)
                x = x_pos[i-1] + (l1x+2*l2x+2*l3x+l4x)*(1/6)
                y = y_pos[i-1] + (l1y+2*l2y+2*l3y+l4y)*(1/6)

                # Append to both arrays #
                x_pos.append(x)
                y_pos.append(y)
                x_vel.append(velocity_x1)
                y_vel.append(velocity_y1)

                # Increments the time step #
                T.append(t)
        # Return 4 arrays including the x position of the ball, the y position of the ball and the velocity at each time step #
        y_pos[-1] = 0
        return x_pos[::2], y_pos[::2]
        
def error(x,y,x1,y1):
    xerr = []
    yerr = []
    if len(x) == len(x1):
        for i in range(len(x)):
            df = abs(x[i] - x1[i])
            xerr.append(df*df)
    else:
        print ('X array length dosnt match')

    if len(y) == len(y1):
        for k in range(len(y)):
            df = abs(y[k] - y1[k])
            yerr.append(df*df)
    else:
        print ('Y array length dosnt match')
    xerr = np.sum(np.array(xerr))
    yerr = np.sum(np.array(yerr))

    return xerr,yerr
print('Analytical')
analx,analy = analytical_with_linear_drag()
plt.plot(analx,analy,label = 'Analytical')
print ('x:  ',len(analx))
print('y:  ',len(analy),'\n')


print('Euler')
x,y = intergrate_linear_drag()
plt.plot(x,y,label = 'Euler')
tempx = analx
tempy = analy
del(tempx[-1])
del(tempy[-1])
print ('x:  ',len(x))
print('y:  ',len(y))
xerr, yerr = error(tempx,tempy,x,y)
print('Euler:  ','X:',xerr,'   Y:',yerr,'\n')

analx,analy = analytical_with_linear_drag()

print('Rk4')
x_pos,y_pos = rk4()
plt.plot(x_pos,y_pos,label = 'Rk4')
print('x:  ',len(x_pos))
print('y:  ',len(y_pos))
xerr, yerr = error(analx,analy,x_pos,y_pos)
print('RK4:  ','X:',xerr,'   Y:',yerr,'\n')

print('Rk2')
x_pos,y_pos = intergrate_no_drag()
plt.plot(x_pos,y_pos,label = 'Rk2')
print('x:  ',len(x_pos))
print('y:  ',len(y_pos))
xerr, yerr = error(analx,analy,x_pos,y_pos)
print('Rk4:  ','X:',xerr,'   Y:',yerr,'\n')

print('New Rk4')
x_pos,y_pos = rk4()
plt.plot(x_pos,y_pos,label = 'new Rk4')
print('x:  ',len(x_pos))
print('y:  ',len(y_pos))
xerr, yerr = error(analx,analy,x_pos,y_pos)
print('New RK4:  ','X:',xerr,'   Y:',yerr,'\n')

plt.legend()
plt.grid()
plt.show()