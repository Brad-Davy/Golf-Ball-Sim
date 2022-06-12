## This script will run the intergration method ##
from Components import get_velocity , analytical_with_linear_drag as ald 
import matplotlib.pyplot as plt 
import numpy as np 
import math
from Strike import Impulse
from aokicd import A_cd_func, A_cl_func
from Bearman_Cd import BearmanCdVelSpin, Bearman_Cd_func 
from Bearman_CL import BearmanClVelSpin
from Interpolation import interpolate
from interpolate_cl import interpolate_cl
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


class runge_kutta_4:

    def run_off(self,x_pos, y_pos, x_vel, y_vel,spin):
        g    = 9.81
        mu   = 1
        r    =  0.0213
        spin = spin*2*np.pi*r
        vbqy = abs(y_vel[-1])
        vbqx = abs(x_vel[-1])
        ef   = 0.510 - 0.0375*vbqy + 0.000903*(vbqy**2)
        vbry = vbqy*ef
        vbrx = (5*vbqx-2*r*spin)/7
        b1   = (2/g)*vbrx*vbry
        bt   = abs(2*b1)
        rt   = abs((vbrx**2)/(2*mu*g))
        run = bt + rt
        return run

    def diff(self,Z,W):
        if W > Z:
           return True, -abs(abs(Z)-abs(W))
        if Z > W:
         return False, abs(abs(Z)-abs(W))
        elif Z == 0:
           return True, 0

    def DragForce(self,XVelocity,YVelocity,ZVelocity,spin,rho,Area,WindVelocity,WindAngle):
        wx = WindVelocity*np.cos(np.radians(WindAngle))
        wz = WindVelocity*np.sin(np.radians(WindAngle))
        _,Zvelocity = self.diff(ZVelocity,wz)
        XVelocity = XVelocity - wx
        Velocity = ((XVelocity)**2 + YVelocity**2)**0.5
        phi = 0
        theta    = np.radians(270) - np.arctan(YVelocity/XVelocity) 
        Cd       = interpolate(spin,Velocity)
        Fd       = 0.5*rho*Area*Cd*Velocity**2
        Fdz      = -Fd*np.sin(theta)*np.sin(phi) - 0.5*rho*Area*Cd*Zvelocity**2
        Fdx      = -Fd*np.sin(theta)*np.cos(phi)
        Fdy      = -Fd*np.cos(theta)
        return Fdx,Fdy,Fdz,Cd

    def LiftForce(self,XVelocity,YVelocity,ZVelocity,spin,rho,Area,SpinAngle,WindVelocity = 0,WindAngle = 0):
        wx = WindVelocity#*np.cos(np.radians(float(WindAngle)))
        wz = 0#WindVelocity*np.sin(np.radians(float(WindAngle)))
        Velocity = ((XVelocity - wx)**2 + YVelocity**2 + (ZVelocity - wz)**2)**0.5
        angle    = np.arctan(YVelocity/(XVelocity - wx))
        Cl       = interpolate_cl(spin,Velocity)#BearmanClVelSpin(Velocity,spin)
        Fl       = 0.5*rho*Area*Cl*Velocity**2
        Flx      = Fl*np.sin(np.radians(360)-angle)*np.sin(np.radians(270-SpinAngle))
        Fly      = Fl*np.cos(np.radians(360)-angle)
        Flz      = Fl*np.sin(np.radians(360)-angle)*np.cos(np.radians(270-SpinAngle))
        return Flx,Fly,Flz,Cl

    def air_density(self,y_cord,dynamic):
        if dynamic == True:
            p0  = 101325
            T0  = 288.15
            g   = 9.81
            L   = 0.0065
            R   = 8.31447
            M   = 0.0289654
            T   = T0 - L*y_cord
            p   = p0 * ( (1-(L*y_cord/T0) )**((g*M)/(R*L)) )
            rho = (p*M)/(R*T)
            return rho
        elif dynamic == False:
            return 1.225

    def determine_launch(self,velocity,loft):
        im = Impulse
        velocity, angle, spin =  im.determine_angle_from_club(self,velocity_of_club = velocity,loft = loft)
        return velocity, angle, spin
    
    def determine_landing(self,x_pos,y_pos):
        y1 = y_pos[-2]
        y2 = y_pos[-1]
        x1 = x_pos[-2]
        x2 = x_pos[-1]
        Y = [y1,y2]
        X = [x1,x2]
        fit = np.polyfit(X,Y,1)
        fit = abs(fit[1])/abs(fit[0])
        return fit , 0
    
    def wind_speed(self,y_cord,windspeed):
        if y_cord <= 0:
            return 0
        else:
            if windspeed == 0:
                return 0
            else:
                n  = 0.37-(0.0881*np.log(abs(windspeed)))
                v2 = windspeed*(y_cord/2)**n
                return v2


    def intergrate(self, dt = 0.05, gravity = 9.81, total_velocity = 55, angle = 30, WindVelocity = 0, WindAngle = 0, SpinAngle = 0, torque_gamma = 12.5,dynamic_air=True):
        #Determine the intergartion of the function and return two arrays #
        total_velocity, angle, init_omega = self.determine_launch(total_velocity, angle)
        init_velx = np.cos(np.radians(angle))*total_velocity 
        init_vely = np.sin(np.radians(angle))*total_velocity
        np.seterr('raise')
        # Set appropriate constants for the integrator #
        t            = 0
        T            = []
        CD           = [0]
        CL           = [0]
        mass         = 0.0459
        r            = 0.0213
        A            = (np.pi*(2*r)**2)/4
        x_pos        = [0]
        y_pos        = [0]
        z_pos        = [0]
        x_vel        = [init_velx]
        y_vel        = [init_vely]
        z_vel        = [0]
        W = []
        StartWindVelocity = WindVelocity
        
        # Begin loop for the runge kutta method #
        for i in range(20000):
            # Breaks loop if the Y values goes past 0 as this is when the ball bounces #
            if y_pos[i-1] < 0:
                break
            else:
                # Determine the spin of the ball and the air density at that paticular y cordinate #
                spin = init_omega*np.exp(-torque_gamma*t*5/2*mass*r)
                rho  = self.air_density(y_pos[i],dynamic_air)
                WindVelocity = self.wind_speed(y_pos[i],StartWindVelocity)
                WindVelocity = abs(WindVelocity*np.sin(np.pi*t/10))
                W.append(WindVelocity)


                # Determines the K1 constant of the runge kutta method #
                k1x         = x_vel[i]
                k1y         = y_vel[i]
                k1z         = z_vel[i]
                fdx,fdy,fdz,cd1 = self.DragForce(XVelocity = k1x, YVelocity = k1y, ZVelocity = k1z, rho = rho, Area = A, spin = spin, WindVelocity = WindVelocity, WindAngle = WindAngle)
                flx,fly,flz,cl1 = self.LiftForce(XVelocity = k1x, YVelocity = k1y, ZVelocity = k1z ,rho = rho, Area = A, spin = spin, SpinAngle = SpinAngle, WindVelocity = WindVelocity, WindAngle = WindAngle)
                l1x         = k1x*dt
                l1y         = k1y*dt
                l1z         = k1z*dt
                x_acc       = (-flx-fdx)*(1/mass)
                y_acc       = (1/mass)*(fly-fdy)-gravity
                z_acc       = (1/mass)*(flz-fdz)



                # Determines the K2 constant of the runge kutta method #
                k2x         = k1x + 0.5*dt*x_acc
                k2y         = k1y + 0.5*dt*y_acc
                k2z         = k1z + 0.5*dt*z_acc
                l2x         = k2x*dt
                l2y         = k2y*dt
                l2z         = k2z*dt
                fdx,fdy,fdz,cd2 = self.DragForce(XVelocity = k2x, YVelocity = k2y, ZVelocity = k2z, rho = rho, Area = A, spin = spin, WindVelocity = WindVelocity, WindAngle = WindAngle)
                flx,fly,flz,cl2 = self.LiftForce(XVelocity = k2x, YVelocity = k2y, ZVelocity = k2z, rho = rho, Area = A, spin = spin, SpinAngle = SpinAngle, WindVelocity = WindVelocity, WindAngle = WindAngle)
                x_acc1      = (-flx-fdx)*(1/mass)
                y_acc1      = (1/mass)*(fly-fdy)-gravity
                z_acc1      = (1/mass)*(flz-fdz)

                # Determines the K3 constant of the runge kutta method #
                k3x         = x_vel[i] + 0.5*dt*x_acc1
                k3y         = y_vel[i] + 0.5*dt*y_acc1
                k3z         = z_vel[i] + 0.5*dt*z_acc1
                l3x         = k3x*dt
                l3y         = k3y*dt
                l3z         = k3z*dt
                fdx,fdy,fdz,cd3 = self.DragForce(XVelocity = k3x, YVelocity = k3y, ZVelocity = k3z, rho = rho, Area = A, spin = spin, WindVelocity = WindVelocity, WindAngle = WindAngle)
                flx,fly,flz,cl3 = self.LiftForce(XVelocity = k3x, YVelocity = k3y, ZVelocity = k3z, rho = rho, Area = A, spin = spin, SpinAngle = SpinAngle, WindVelocity = WindVelocity, WindAngle = WindAngle)
                x_acc2      = (-flx-fdx)*(1/mass)
                y_acc2      = (1/mass)*(fly-fdy)-gravity
                z_acc2      = (1/mass)*(flz-fdz)

                # Determines the K4 constant of the runge kutta method #
                k4x     = x_vel[i] + dt*x_acc2
                k4y     = y_vel[i] + dt*y_acc2
                k4z     = z_vel[i] + dt*z_acc2
                l4x     = k4x*dt 
                l4y     = k4y*dt
                l4z     = k4z*dt
                fdx,fdy,fdz,cd4 = self.DragForce(XVelocity = k4x, YVelocity = k4y, ZVelocity = k4z, rho = rho, Area = A, spin = spin, WindVelocity = WindVelocity, WindAngle = WindAngle)
                flx,fly,flz,cl4 = self.LiftForce(XVelocity = k4x, YVelocity = k4y, ZVelocity = k4z, rho = rho, Area = A, spin = spin, SpinAngle = SpinAngle, WindVelocity = WindVelocity, WindAngle = WindAngle)
                xh_acc2 = (-flx-fdx)*(1/mass)
                yh_acc2 = (1/mass)*(fly-fdy)-gravity
                zh_acc2      = (1/mass)*(flz-fdz)

                # Update position and velocity #
                velocity_x1 = x_vel[i] + dt*(x_acc+2*x_acc1+2*x_acc2+xh_acc2)*(1/6)
                velocity_y1 = y_vel[i] + dt*(y_acc+2*y_acc1+2*y_acc2+yh_acc2)*(1/6)
                velocity_z1 = z_vel[i] + dt*(z_acc+2*z_acc1+2*z_acc2+zh_acc2)*(1/6)
                x           = x_pos[i] + (l1x+2*l2x+2*l3x+l4x)*(1/6)
                y           = y_pos[i] + (l1y+2*l2y+2*l3y+l4y)*(1/6)
                z           = z_pos[i] + (l1z+2*l2z+2*l3z+l4z)*(1/6)

                # Append to both arrays #
                x_pos.append(x)
                y_pos.append(y)
                z_pos.append(z)
                x_vel.append(velocity_x1)
                y_vel.append(velocity_y1)
                z_vel.append(velocity_z1)
                CD.append((cd1+2*cd2+2*cd3+cd4)*(1/6))
                CL.append((cl1+2*cl2+2*cl3+cl4)*(1/6))

                # Increments the time step #
                t = t + dt
                T.append(t)

        #Determine exact landing position#
        x,y = self.determine_landing(x_pos,y_pos)
        x_pos[-1] = x
        y_pos[-1] = y
        run = self.run_off(x_pos, y_pos, x_vel, y_vel,spin)
        # Return 4 arrays including the x position of the ball, the y position of the ball and the velocity at each time step #
        return x_pos, y_pos, z_pos, x_vel, y_vel, z_vel ,total_velocity , CD, CL, run