## This script will determine the spin/force/angle that the golf ball is hit at ##
import numpy as np 
class Impulse:
    def determine_angle_from_club(self,velocity_of_club = 45, loft = 45):
    ## Angles of the club are from 25 to 58 degreese typically, this function determines the angle of the ball after colliding with the club ##

        # Due to the flex of the shaft of the club the loft of the club when striking the ball is slightly greater ##
        loft = loft + 2
        loft = np.radians(loft)   
        ## Angles of the club are from 25 to 58 degreese typically, this function determines the angle of the ball after colliding with the club ##
        # Constants #
        m = 0.0459
        r = 0.0213
        I = 0.4*m*r*r
        M = 0.2
        
        # e accounts for the compression of the ball #
        e = 0.86 - 0.0029*velocity_of_club*np.cos(loft)

        # Components of velocity normal and parallel to the face of the club #
        vbfn = (1+e)*velocity_of_club*np.cos(loft)/(1+(m/M))
        vbfp = -(velocity_of_club*np.sin(loft))/(1 + m/M + m*r**2/I)

        # Determines the velocity angle and spin of the ball # 
        Vb = (vbfn**2+vbfp**2)**0.5
        angle = loft + np.arctan(vbfp/vbfn)
        spin = -(m*vbfp*r*M)/(I*r)

        # returns the values #
        return Vb, np.degrees(angle), spin
