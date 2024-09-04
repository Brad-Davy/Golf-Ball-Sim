## This script will contain the GUI ##
from tkinter import *
import tkinter as tk
from math import pi
import matplotlib.pyplot as plt
from Components import plot_static_graph , plot_static_graph_with_drag as gwd
import Dynamic_Plot
from Intergration_Methods_for_sweep import Euler, runge_kutta_2 as RK2, runge_kutta_4 as RK4
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
#from ttkthemes import ThemedStyle 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from Bounce import bounce, add_arrays
import tkinter.ttk as ttk
import threading 

global change
change = False
height = 1000
width = 1200

LARGE_FONT= ("Verdana", 12)


f = Figure(figsize=(22,7), dpi=100)
#f.subplots_adjust(left = 0.05,right = 0.96, bottom = 0.04,top = 0.95)
gs = matplotlib.gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.25) # 2x2 grid
a = f.add_subplot(gs[0, 0:2])
Vax = f.add_subplot(gs[0, 2] ) 
Cax = f.add_subplot(gs[1, 2]) 
ax = f.add_subplot(gs[1, 0:2])
ax.set_aspect('equal', adjustable='box')
a.set_aspect('equal', adjustable='box')
rect = f.patch
rect.set_facecolor('lightgray')




def animate(i):
    pullData  = open('sampleText.txt','r').read()
    dataArray = pullData.split('\n')
    xar       = []
    yar       = []
    zar       = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y,z = eachLine.split(',')
            xar.append(float(x))
            yar.append(float(y))
            zar.append(float(z))
    if dataArray[0] == str(0):
        a.clear()
    a.plot(xar,yar,'r--')
    pullData  = open('COEFS.txt','r').read()
    dataArray = pullData.split('\n')
    cd_array       = []
    cl_array       = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            cd_array.append(float(x))
            cl_array.append(float(y))
    Cax.clear()
    Cax.plot(xar[1:],cd_array[1:],'r--',label = 'Cd')
    Cax.plot(xar[1:],cl_array[1:],'b--',label = 'Cl')
    Cax.legend()

    pullData  = open('Velocity.txt','r').read()
    dataArray = pullData.split('\n')
    x_velo       = []
    y_velo       = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            x_velo.append(float(x))
            y_velo.append(float(y))
    Vax.clear()
    Vax.plot(xar,x_velo,'r--',label = 'X')
    Vax.plot(xar,y_velo,'b--',label = 'Y')
    Vax.legend()

    ax.clear()
    ax.plot(xar,zar,'r--')
    a.set_ylim(0,80)
    a.set_xlim(0,300)
    a.set_title('Trajectory')
    a.set_xlabel('Distance / M')
    a.set_ylabel('Height / M')
    ax.set_ylim(-40,40)
    ax.set_xlim(0,300)
    ax.set_title('Trajectory')
    ax.set_xlabel('Distance / M')
    ax.set_ylabel('Height / M')
    Cax.set_title('Coeficents')
    Cax.set_xlabel('Distance / M')
    Cax.set_ylabel('Cd   and   Cl')
    Vax.set_title('Velocitys')
    Vax.set_xlabel('Distance / M')
    Vax.set_ylabel('Velocity')
    


class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.configure(background = 'gray')
        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        menubar = tk.Menu(container)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Graph", command = lambda: print("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        planetsmenu = tk.Menu(menubar,tearoff=0)
        planetsmenu.add_command(label='Earth',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Moon',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Jupiter',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Mars',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Saturn',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Uranus',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Mercury',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Venus',command=lambda: print("Not supported just yet!"))
        planetsmenu.add_command(label='Neptune',command=lambda: print("Not supported just yet!"))
        menubar.add_cascade(label="Planets", menu=planetsmenu)

        densitymenu = tk.Menu(menubar,tearoff=0)
        densitymenu.add_command(label='Air',command=lambda: print("Not supported just yet!"))
        densitymenu.add_command(label='Water',command=lambda: print("Not supported just yet!"))
        densitymenu.add_command(label='Helium',command=lambda: print("Not supported just yet!"))
        densitymenu.add_command(label='Mercury',command=lambda: print("Not supported just yet!"))
        densitymenu.add_command(label='Space',command=lambda: print("Not supported just yet!"))
        menubar.add_cascade(label="Materials", menu=densitymenu)
    

        tk.Tk.config(self, menu=menubar)
        self.frames = {}

        for F in (StartPage, Simulator, Settings):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        tk.Frame.configure(self,background = 'lightgray')
        
        tk.Label(self, text = 'Golf Simulator', font = ('Times',34), bg  = 'light green').place(x=320,y=20)
        start_button = tk.Button(self, text="Start", command=lambda: controller.show_frame(Simulator))
        start_button.place(x=430,y=320)



class Simulator(tk.Frame):
   
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background = 'lightgray')
   

        ## Input entry boxes ##
        self.force_entry=tk.Entry(self,width=4, font = ('times',16), bd = 3, justify = 'center')
        self.force_entry.place(x=85,y=65)     
        self.angle_entry=tk.Entry(self,width=4, font = ('times',16), bd = 3, justify = 'center')
        self.angle_entry.place(x=85,y=115)

        self.keep_plot = IntVar()
        Checkbutton(self, text = 'Keep Plot' ,variable = self.keep_plot, bg  = 'lightgrey').place (x=810,y=150)
        

        # Sets input values to begin #
        self.force_entry.insert(0,'65')
        self.angle_entry.insert(0,'45')

        ## Output entry boxes ##
        self.max_height=tk.Entry(self,width=6,cursor='none', font = ('times',16), bd = 3, justify = 'center')
        self.max_height.place(x=425,y=115) 
        self.max_distance=tk.Entry(self,width=6,cursor='none', font = ('times',16), bd = 3, justify = 'center')
        self.max_distance.place(x=425,y=65)

        # Buttons #

        # Makes a dynamic plot #
        dynamic_plot_button = Button(self,text = 'Dynamic \n Plot', bg = 'light green' , command = self.dynamic_plot)
        dynamic_plot_button.config(font = ('Times',16), relief = 'raised')
        dynamic_plot_button.place(x=800,y=60)

        # Labels/text #
        Label(self,text = 'Velocity', font = ('times',16), bg  = 'lightgrey').place(x=3,y=67)
        Label(self,text = 'Angle', font = ('times',16), bg  = 'lightgrey').place(x=5,y=117)
        Label(self,text = 'Height: ', font = ("times",16), bg  = 'lightgrey').place(x=339,y=67)
        Label(self,text = 'Distance: ', font = ('times', 16), bg  = 'lightgrey').place(x=325,y=117)
        Label(self,text = 'Simulator', font = ('Times',28), bg  = 'lightgrey').place(x=10,y=10)
        Label(self,text = ' Slice \n Angle', font = ('Times',14), bg  = ('lightgrey')).place(x=1505,y=147)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas._tkcanvas.place(x=-300,y=200)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()

        # Plots graph #
        plot_button = Button(self,text = 'Plot', command = self.run, bg = 'light green')
        plot_button.config(font = ('Times',16), relief = 'raised')
        plot_button.place(x=830,y=180)

        # Input boxes #
        gamma_entry = self.gamma_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.gamma_entry.place(x=1575,y=55)
        gravity_entry = self.gravity_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.gravity_entry.place(x=1375,y=163)
        mass_entry = self.mass_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.mass_entry.place(x=1575,y=110)
        wind_speed_entry = self.wind_speed_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.wind_speed_entry.place(x=1375,y=55)
        wind_angle_entry = self.wind_angle_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.wind_angle_entry.place(x=1375,y=110)
        slice_angle_entry = self.slice_angle_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.slice_angle_entry.place(x=1575,y=163)

        # Settin original value for input boxes #
        self.gamma_entry.insert(0,'1.225')
        self.gravity_entry.insert(0,'9.81')
        self.mass_entry.insert(0,'0.0459')
        self.wind_speed_entry.insert(0,'10')
        self.wind_angle_entry.insert(0,'0')
        self.slice_angle_entry.insert(0,'0')

        Label(self,text = 'Denisty', font = ('Times',14), bg  = 'lightgrey').place (x=1510,y=53)
        Label(self,text = 'Mass', font = ('Times',14), bg  = 'lightgrey').place (x=1515,y=111)
        Label(self,text = 'Gravity', font = ('Times',14), bg  = 'lightgrey').place (x=1310,y=162)
        Label(self,text = ' Wind \n Speed', font = ('Times',14), bg  = 'lightgrey').place (x=1310,y=50)
        Label(self,text = ' Wind \n Angle', font = ('Times',14), bg  = 'lightgrey').place (x=1310,y=106)
        Label(self, text = 'Settings', font = ('Times',28), bg  = 'lightgrey').place(x=1410,y=5)

        set_button = Button(self, text = 'Set', command = self.set_parameters,width = 8)
        set_button.config(font=('Times',16), bg = 'light green')
        set_button.place(x=1580,y=200)


    def set_parameters(self):
        f = open('run_file.txt','w')
        f.write(self.gamma_entry.get())
        f.write(' '+self.gravity_entry.get())      
        f.write(' '+'1')  
        f.write(' '+self.wind_speed_entry.get())
        f.write(' '+self.wind_angle_entry.get())


    def dynamic_plot(self):
        velocity   = self.force_entry.get()
        angle   = float(self.angle_entry.get())
        f       = open('run_file.txt')
        f       = f.read()
        f       = f.split()
        gravity = float(f[1])
        wind_speed = float(f[3])
        wind_angle = float(f[4])
        if int(f[2]) == 1:
            drag = True
        if int(f[2]) == 0:
            drag = True    
        angle = np.radians(angle)
        height, distance = Dynamic_Plot.main(total_velocity=velocity, angle=float(np.degrees(angle)), gravity=gravity, WindAngle=wind_angle, WindVelocity=wind_speed)
        self.max_height.delete(0,'end')
        self.max_distance.delete(0,'end')
        self.max_height.insert(0,str(float('%.2f'%(height))))
        self.max_distance.insert(0,str(float('%.2f'%(distance))))

    def run(self):
        rk4         = RK4()
        force       = self.force_entry.get()
        slice_angle = float(self.slice_angle_entry.get())
        angle       = float(self.angle_entry.get())
        f           = open('run_file.txt')
        f           = f.read()
        f           = f.split()
        gravity     = float(f[1])
        drag = True
        wind_speed = float(f[3])
        wind_angle = float(f[4])
        if drag == True:
            x_pos, y_pos, z_pos, xv, yv, zv, omega, Cd, Cl, run = rk4.intergrate(dt = 0.05, gravity = gravity, total_velocity = int(force), angle = angle,WindVelocity=wind_speed,WindAngle = wind_angle,SpinAngle = slice_angle)
            distance = x_pos[-1]
            height = max(y_pos)
        if drag == False:
            x_pos, y_pos, xv, yv = rk4.intergrate(dt = 0.05, gravity = gravity, total_velocity = int(force), angle = angle,wind_speed=wind_speed)
            bx,by,vbrx = bounce(5)
            x_pos = add_arrays(bx,x_pos)
            y_pos = y_pos + by
        f = open('sampleText.txt','w')
        C = open('COEFS.txt','w')
        V = open('Velocity.txt','w')
        for w in range(len(xv)):
            V.write(str(xv[w])+','+str(yv[w]))
            V.write('\n')
        V.close()
        for q in range(len(Cd)):
            C.write(str(Cd[q])+','+str(Cl[q]))
            C.write('\n')
        C.close()
        f.write(str(self.keep_plot.get()))
        f.write('\n')
        for x in range(len(x_pos)):
            f.write(str(x_pos[x])+','+str(y_pos[x])+','+str(z_pos[x]))
            f.write('\n')
        #f.write(str(x_pos[-1]+run)+','+'2,2')
        #f.write('\n')
        f.close()


        self.max_height.delete(0,'end')
        self.max_distance.delete(0,'end')
        self.max_height.insert(0,str(float('%.2f'%(distance))))
        self.max_distance.insert(0,str(float('%.2f'%(height))))
        plt.show()

    def quit_app(self):
        pass



class Settings(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background = 'lightgray')
        self.drag = True
        self.drag_colour = 'green'
        self.wind = True
        self.wind_colour = 'green'
        self.lift = True
        self.lift_colour = 'green'

        settings_button = Button(self, text = 'Settings', command = lambda: controller.show_frame(Settings),bg = 'light blue')
        settings_button.config(font = ('Times',16), relief = 'sunken')
        settings_button.place(x=810,y=10)

        simulator_button = Button(self, text = 'Simulator', command = lambda: controller.show_frame(Simulator),bg = 'light blue')
        simulator_button.config(font = ('Times',16))
        simulator_button.place(x=708,y=10)

        # Choosing integration method #
        euler_variable = IntVar()
        euler_method   = Checkbutton(self, text = 'Euler' ,variable = euler_variable, bg  = 'lightgrey').place (x=830,y=670)
        rk2_variable   = IntVar()
        rk2_method     = Checkbutton(self, text = 'RK2' ,variable = rk2_variable, bg  = 'lightgrey').place (x=830,y=650)
        rk4_variable   = IntVar()
        rk4_method     = Checkbutton(self, text = 'RK4' ,variable = rk4_variable, bg  = 'lightgrey').place (x=830,y=630)


        # Input boxes #
        gamma_entry = self.gamma_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.gamma_entry.place(x=75,y=85)
        gravity_entry = self.gravity_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.gravity_entry.place(x=75,y=140)
        mass_entry = self.mass_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.mass_entry.place(x=75,y=195)
        wind_speed_entry = self.wind_speed_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.wind_speed_entry.place(x=75,y=250)
        wind_angle_entry = self.wind_angle_entry=tk.Entry(self,width=6, font = ('times',16), bd = 3, justify = 'center')
        self.wind_angle_entry.place(x=75,y=305)

        # Settin original value for input boxes #
        self.gamma_entry.insert(0,'1.125')
        self.gravity_entry.insert(0,'9.81')
        self.mass_entry.insert(0,'0.0459')
        self.wind_speed_entry.insert(0,'10')
        self.wind_angle_entry.insert(0,'0')


        # Indicators for drag, wind, e.c.t #
        drag_indicator = self.drag_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.drag_colour)
        self.drag_indicator.place(x=420,y=85)
        wind_indicator = self.wind_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.drag_colour)
        self.wind_indicator.place(x=420,y=145)
        lift_indicator = self.lift_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.drag_colour)
        self.lift_indicator.place(x=420,y=205)


        Label(self,text = 'Denisty', font = ('Times',14), bg  = 'lightgrey').place (x=10,y=83)
        Label(self,text = 'Mass', font = ('Times',14), bg  = 'lightgrey').place (x=15,y=191)
        Label(self,text = 'Gravity', font = ('Times',14), bg  = 'lightgrey').place (x=10,y=137)
        Label(self,text = ' Wind \n Speed', font = ('Times',14), bg  = 'lightgrey').place (x=10,y=245)
        Label(self,text = ' Wind \n Angle', font = ('Times',14), bg  = 'lightgrey').place (x=10,y=299)
        Label(self, text = 'Settings', font = ('Times',28), bg  = 'lightgrey').place(x=10,y=10)

        set_button = Button(self, text = 'Set', command = self.set_parameters)
        set_button.config(font=('Times',16), bg = 'light green')
        set_button.place(x=280,y=280)

        drag_toggle = Button(self,text = 'Drag', command = self.toggle_drag, bg = 'light blue')
        drag_toggle.config(font = ('Times',16))
        drag_toggle.place(x = 340, y = 80)

        wind_toggle = Button(self,text = 'Wind', command = self.toggle_wind, bg = 'light blue')
        wind_toggle.config(font = ('Times',16))
        wind_toggle.place(x = 340, y = 140)

        lift_toggle = Button(self,text = 'Lift', command = self.toggle_lift, bg = 'light blue')
        lift_toggle.config(font = ('Times',16))
        lift_toggle.place(x = 340, y = 200)

    def set_parameters(self):
        f = open('run_file.txt','w')
        f.write(self.gamma_entry.get())
        f.write(' '+self.gravity_entry.get())
        if self.drag == True:
            f.write(' '+'1')
        if self.drag == False:
            f.write(' '+'0')
        f.write(' '+self.wind_speed_entry.get())
        f.write(' '+self.wind_angle_entry.get())


    def toggle_drag(self):
        if self.drag == True:
            self.drag_colour = 'red'
            self.drag = False
            drag_indicator = self.drag_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.drag_colour)
            self.drag_indicator.place(x=420,y=85) 
            

        elif self.drag == False:
            self.drag_colour = 'green'
            self.drag = True
            drag_indicator = self.drag_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.drag_colour)
            self.drag_indicator.place(x=420,y=85)


    def toggle_wind(self):
        if self.wind == True:
            self.wind_colour = 'red'
            self.wind = False
            wind_indicator = self.wind_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.wind_colour)
            self.wind_indicator.place(x=420,y=145) 
            

        elif self.wind == False:
            self.wind_colour = 'green'
            self.wind = True
            wind_indicator = self.wind_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.wind_colour)
            self.wind_indicator.place(x=420,y=145)


    def toggle_lift(self):
        if self.lift == True:
            self.lift_colour = 'red'
            self.lift = False
            lift_indicator = self.lift_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.lift_colour)
            self.lift_indicator.place(x=420,y=205) 
            

        elif self.lift == False:
            self.lift_colour = 'green'
            self.lift = True
            lift_indicator = self.lift_indicator=tk.Entry(self,width=3,cursor='none', font = ('times',16), bd = 3, justify = 'center', bg = self.lift_colour)
            self.lift_indicator.place(x=420,y=205)


## Creates the window ##
def RUNGUI():
    app = GUI()
    ani = animation.FuncAnimation(f, animate, interval=1000)
    app.geometry(str(width)+"x"+str(height))
    app.title('Golf Simulator')
    app.mainloop()
GUI_thread = threading.Thread(target = RUNGUI())
GUI_thread.start()
