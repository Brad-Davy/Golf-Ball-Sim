## This script will run the intergration method ##
from Components import get_velocity , analytical_with_linear_drag as ald 
import matplotlib.pyplot as plt 
import numpy as np 
import Lift
from Lift import lift_function
from Drag import drag_function, test_drag_fucntion
from LinearRegression import LRLift_function
from Strike import Impulse
from aokicd import A_cd_func
import math

class Euler:
    def what_am_i(self):
        print ('This is the euler method of intergration.')

    def intergrate_linear_drag(self, dt = 0.05, gravity = 9.81, force = 100, angle = 45):
        #Determine the intergartion of the function and return two arrays ##
        init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 100
        iterations = T/dt
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
        return x_pos, y_pos , x_vel, y_vel


class runge_kutta_2:
    def what_am_i(self):
        print ('This is the 2nd order Runge-Kutta method of intergration.')

    def intergrate_no_drag(self, dt = 0.05, gravity = 9.81, force = 100, angle = 45):
        #Determine the intergartion of the function and return two arrays ##
        init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 1000
        iterations = T/dt
        t = 0
        mass = 0.043
        x_pos = [0]
        y_pos = [0]
        y_vel = [init_vely]
        x_vel = [init_velx]
        gamma = 0.002
        for i in range(int(iterations)):
            if y_pos[i-1] < 0:
                break
            else:
                x_acc = 0
                y_acc = (1/mass)*(-mass*gravity)
                vxh = x_vel[i-1] + 0.5*dt*x_acc
                vyh = y_vel[i-1] + 0.5*dt*y_acc
                xh_acc = 0
                yh_acc = (1/mass)*(-mass*gravity)
                velocity_x = x_vel[i-1] + dt*xh_acc
                velocity_y = y_vel[i-1] + dt*yh_acc
                x = x_pos[i-1] + dt*vxh
                y = y_pos[i-1] + dt*vyh
                x_pos.append(x)
                y_pos.append(y)
                x_vel.append(velocity_x)
                y_vel.append(velocity_y)
                t = t + dt
        return x_pos, y_pos , x_vel, y_vel

    def intergrate_linear_drag(self, dt = 0.05, gravity = 9.81, force = 100, angle = 45, gamma = 0.001):
        #Determine the intergartion of the function and return two arrays ##
        init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 1000
        iterations = T/dt
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
        return x_pos, y_pos , x_vel, y_vel
    
    def intergrate_quadratic_drag(self, dt = 0.05, gravity = 9.81, force = 100, angle = 45, gamma = 0.001):
        #Determine the intergartion of the function and return two arrays ##
        init_velx, init_vely, total_vel = get_velocity(force=force, angle = angle, time = 0.05)
        T = 100
        iterations = T/dt
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
                vxh = x_vel[i-1] + 0.5*dt*x_acc
                vyh = y_vel[i-1] + 0.5*dt*y_acc
                xh_acc = -gamma*(vxh*vxh)/mass
                yh_acc = (1/mass)*(-gamma*vyh*vyh-mass*gravity)
                velocity_x = x_vel[i-1] + dt*xh_acc
                velocity_y = y_vel[i-1] + dt*yh_acc
                x = x_pos[i-1] + dt*vxh
                y = y_pos[i-1] + dt*vyh
                x_pos.append(x)
                y_pos.append(y)
                x_vel.append(velocity_x)
                y_vel.append(velocity_y)
                t = t + dt
        return x_pos, y_pos , x_vel, y_vel


class force:
    def force(self):
        forces = {'gravity':9.81,'gamma':0.0001}

class runge_kutta_4:

    #def Xforce(self, drag = True, Lift = True, wind = True, velocity, gamma,mass):
     #   if drag == True and Lift == True and wind == True:
      #      return -gamma*velocity/mass

    def air_density(self,y_cord):
        p0 = 101325
        T0 = 288.15
        g = 9.81
        L = 0.0065
        R = 8.31447
        M = 0.0289654
        T = T0 - L*y_cord
        p = p0 * ( (1-(L*y_cord/T0) )**((g*M)/(R*L)) )
        rho = (p*M)/(R*T)
        return rho

    def wind_velocity(self,y_cord,angle,v1):
        if v1 == 0:
            wind_x = 0
            wind_z = 0
        else:
            v1 = v1
            z1 = 2
            angle = np.radians(angle)
            n = 0.37-(0.0881*np.log(v1))
            v2 = v1*(y_cord/z1)**n
            wind_x = np.cos(angle)*v2
            wind_z = np.sin(angle)*v2
        return wind_x, wind_z

    def what_am_i(self):
        print ('This is the 4th order Runge-Kutta method of intergration.')

    def determine_launch(self,velocity,loft):
        im = Impulse
        velocity, angle, spin =  im.determine_angle_from_club(self,velocity_of_club = velocity,loft = loft)
        return velocity, angle, spin

    def intergrate(self, dt = 0.05, gravity = 9.81, total_velocity = 55, angle = 45, rho = 1.225, wind_speed = 5, bounce = False):
        angle = np.degrees(angle)
        #Determines the launch conditions based on the loft anf velocity of the club head #
        if bounce == False:
            total_velocity, angle, init_omega = self.determine_launch(total_velocity, angle)
        else:
            init_omega = 300
            pass
        angle = np.radians(angle)
        #Determine the intergartion of the function and return two arrays #
        init_velx = np.cos(angle)*total_velocity 
        init_vely = np.sin(angle)*total_velocity
        # Set appropriate constants for the integrator #
        t = 0
        kin_visc_air = 0.00000148
        T = []
        mass = 0.0459
        r = 0.0213
        A = np.pi*r*r
        Cd = 0.1
        Cl = 0.1
        gamma = 0.5*rho*A*Cd
        lift_gamma = 0.5*rho*A*Cl
        torque_gamma = 7.5
        x_pos = [0]
        y_pos = [0]
        x_vel = [init_velx]
        y_vel = [init_vely]
        
        # Begin loop for the runge kutta method #
        for i in range(20000):
            # Breaks loop if the Y values goes past 0 as this is when the ball bounces #
            if y_pos[i-1] < 0:
                break
            else:
                # Calculates the angular veloicty for that specific time step #
                omega = init_omega*np.exp(-torque_gamma*t*5/2*mass*r)
                total_velocity = (x_vel[i-1]**2+y_vel[i-1]**2)**0.5
                reynolds_number = (total_velocity*A)/kin_visc_air
                #Cl = lift_function(omega,reynolds_number)
                #Cl = LRLift_function(reynolds_number,omega)
                #Cd = drag_function(omega,reynolds_number)
                rho = self.air_density(y_pos[i-1])
                #Cd = test_drag_fucntion(reynolds_number)
                Cd = 0.005*x_vel[i-1] + math.exp(0.64288716)*math.exp(-0.10546236*x_vel[i-1])
                cl = 0.2
                gamma = 0.5*rho*A*Cd
                lift_gamma = 0.5*rho*A*Cl
                # Determien the wind speed #
                winx,winy = self.wind_velocity(y_pos[i-1],45,float(wind_speed))
                relative_velocityx = x_vel[i-1] - winx

                # Determines the K1 constant of the runge kutta method #
                k1x = relative_velocityx 
                k1y = y_vel[i-1]
                l1x = k1x*dt
                l1y = k1y*dt
                x_acc = -gamma*k1x*k1x/mass
                y_acc = (1/mass)*(-gamma*k1y*k1y-mass*gravity+(lift_gamma*k1y*k1y))


                # Determines the K2 constant of the runge kutta method #
                k2x = k1x + 0.5*dt*x_acc
                k2y = k1y + 0.5*dt*y_acc
                l2x = k2x*dt
                l2y = k2y*dt
                x_acc1 = -gamma*k2x*k2x/mass
                y_acc1 = (1/mass)*(-gamma*k2y*k2y-mass*gravity+lift_gamma*k2y*k2y)

                # Determines the K3 constant of the runge kutta method #
                k3x = relative_velocityx + 0.5*dt*x_acc1
                k3y = y_vel[i-1] + 0.5*dt*y_acc1
                l3x = k3x*dt
                l3y = k3y*dt
                x_acc2 = -gamma*k3x*k3x/mass
                y_acc2 = (1/mass)*(-gamma*k3y*k3y-mass*gravity+lift_gamma*k3y*k3y)

                # Determines the K4 constant of the runge kutta method #
                k4x = relative_velocityx + dt*x_acc2
                k4y = y_vel[i-1] + dt*y_acc2
                l4x = k4x*dt 
                l4y = k4y*dt
                xh_acc2 = -gamma*k4x*k4x/mass
                yh_acc2 = (1/mass)*(-gamma*k4y*k4y-mass*gravity+lift_gamma*k4y*k4y)

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
                t = t + dt
                T.append(t)
        # Return 4 arrays including the x position of the ball, the y position of the ball and the velocity at each time step #
        return x_pos, y_pos, x_vel, y_vel, omega
