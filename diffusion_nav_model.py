# CSCI 4314: Final Project
# Diffusion Navigation Culture

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import matplotlib.animation as animation
import random

#fix d a and b parameters 
#play with ratio of chemotaxis and just random walk (random mix)

#explore speed:
#slower speed for chemotaxis

#fix wind

#set time 

# fix D,B,A
# fix path
# fix diffusion properties
# play with ratio between chemotaxis and random walk
# play with difference in speed (ratio)
# fix wind speed
# better/more efficient search
# run till convergence, max time
# maybe use chemotaxis at higher concentrations
class Source:
    def __init__(self):
        pass
    
    def get_interpolated_data(self, t_array, n_sources):
        source_x_array, source_y_array, wind_x, wind_y = self.get_data(30) # wind_x and wind_y must have same length
        pathXY = np.asarray([source_x_array, source_y_array])
        
        # compute step lengths from data
        step_lengths = np.sqrt(np.sum(np.power(np.diff(pathXY, axis=1), 2), axis=0))
        step_lengths = np.insert(step_lengths, 0, 0)
        
        # interpolate movement between data points
        cumulative_length = np.cumsum(step_lengths)
        final_step_locs = np.linspace(0, cumulative_length[-1], n_sources)

        # # target should stop moving!
        # if(n_sources < len(t_array)): # should target just stop moving but still have an odor?
        #     repeat_final_position = np.repeat(cumulative_length[-1], repeats=(len(t_array)-n_sources))
        #     final_step_locs = np.append(final_step_locs, repeat_final_position)
        #     n_sources = len(t_array)

        f = interpolate.interp1d(cumulative_length, pathXY) # location (x,y) as a funciton of arclength (distance traveled along curve)
        final_pathXY= f(final_step_locs) # compute smooth curve of source's path s
        
        # interpolated movement data
        source_x_array = final_pathXY[0] 
        source_y_array = final_pathXY[1]

        # now for wind (interpolate)
        f_windx = interpolate.interp1d(np.linspace(1, len(wind_x), len(wind_x)), wind_x)
        f_windy = interpolate.interp1d(np.linspace(1, len(wind_y), len(wind_y)), wind_y)
        final_wind_steps = np.linspace(1, len(wind_x), n_sources)
        wind_x = f_windx(final_wind_steps)
        wind_y = f_windy(final_wind_steps)
        return source_x_array, source_y_array, wind_x, wind_y

    def get_data(self, n):
        wind_x = np.zeros(n)
        wind_y = wind_x
        source_y_array = np.linspace(0,50,n)
        source_x_array = 10*np.sin(np.linspace(-math.pi, math.pi, n))
        return source_x_array, source_y_array, wind_x, wind_y

class Human(Source):
    def get_data(self, n):
        wind_x = np.random.uniform(-0.05,0.05,size=n)
        wind_y = np.random.uniform(-0.05,0.05,size=n)
        source_y_array = np.linspace(0,25,n)
        source_x_array = 10*np.sin(np.linspace(-math.pi, math.pi, n))
        return source_x_array, source_y_array, wind_x, wind_y


class Agent:
    def __init__(self, x=0, y=0, dx=0.5, mem_internal=25, v_multiplier=5):
        self.pos_x = 0
        self.pos_y = 0
        self.v = dx*v_multiplier

        self.curr_px = 0
        self.curr_py = 0
        self.prev_px = 0 
        self.prev_py = 0
        self.prev_theta = 0
        self.curr_c_agent = 0
        self.sigma02 = np.pi
        self.theta = 0
    
    # default agent moves up concentration gradient at speed v
    # this agent works perfectly when wind = 0, but kind of sucks otherwise, also pretty slow.
    def get_agent_position(self, c, x, y, dx, dy):
        fy,fx = np.gradient(c,dx,dx) # fy before fx
        tt = np.floor(x/dx) == np.floor(self.pos_x/dx)
        ttt = np.floor(y/dy) == np.floor(self.pos_y/dy)
        indt = tt&ttt # where is agent
        norm =np.linalg.norm([fx[indt], fy[indt]])
        if norm>0: #0.01
            self.prev_px = self.curr_px
            self.prev_py = self.curr_py
            self.curr_px = fx[indt][0]/norm
            self.curr_py = fy[indt][0]/norm
            self.curr_c_agent  = c[indt]
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.pos_x = self.pos_x + self.v*self.curr_px
        self.pos_y = self.pos_y + self.v*self.curr_py
        return self.pos_x, self.pos_y

class Curtis(Agent):
    #my attempt at a correlated random walk
    #taken from a portion of Orit Peleg's starter code
    def __init__(self, strat_probs=[0.5,0.5], **kwds):
        """
        :param strat_probs: list of length two for probability for each of two strategies ['chemotaxis','crw']
        """
        super().__init__(**kwds)
        self.strat_probs = strat_probs
        self.strat_change = False
    
    def get_agent_position(self, c, x, y, dx, dy):
        strat = np.random.choice(['chemotaxis', 'crw'], p=self.strat_probs)
        if strat == 'chemotaxis':
            return self.chemotaxis(c,x,y,dx,dy)
        elif strat == 'crw':
            return self.crw(c,x,y,dx,dy)
        else:
            raise ValueError(f"No strategy called {strat}")

    #this is currently a biased random walk, my mistake
    #we need to change this to a crw

    def brw(self, c, x, y, dx, dy):

        #Changed to have to move in either direction
        choice = np.floor(random.uniform(0,1) * 2) + 1
        if choice == 1:
            self.theta = self.prev_theta + self.sigma02 * random.uniform(0,1)
        elif choice == 2:
            self.theta = self.prev_theta - self.sigma02 * random.uniform(0,1)
        else:
            raise ValueError('no such choice')
        self.pos_x = self.pos_x + self.v * np.cos(self.theta)
        self.pos_y = self.pos_y + self.v * np.sin(self.theta)
        return self.pos_x, self.pos_y

    def crw(self, c, x, y, dx, dy):

        #Changed to have to move in either direction
        self.theta = self.prev_theta
        choice = np.floor(random.uniform(0,1) * 2) + 1
        if choice == 1:
            self.theta = self.theta + self.sigma02 * random.uniform(0,1)
        elif choice == 2:
            self.theta = self.theta - self.sigma02 * random.uniform(0,1)
        else:
            raise ValueError('no such choice')
        self.pos_x = self.pos_x + self.v * np.cos(self.theta)
        self.pos_y = self.pos_y + self.v * np.sin(self.theta)
        return self.pos_x, self.pos_y

    def chemotaxis(self, c, x, y, dx, dy):
        fy,fx = np.gradient(c,dx,dx) # fy before fx
        tt = np.floor(x/dx) == np.floor(self.pos_x/dx)
        ttt = np.floor(y/dy) == np.floor(self.pos_y/dy)
        indt = tt&ttt # where is agent
        norm =np.linalg.norm([fx[indt], fy[indt]])
        if norm>0: #0.01
            self.prev_px = self.curr_px
            self.prev_py = self.curr_py
            self.curr_px = fx[indt][0]/norm
            self.curr_py = fy[indt][0]/norm
            self.curr_c_agent  = c[indt]
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.pos_x = self.pos_x + self.v*self.curr_px
        self.pos_y = self.pos_y + self.v*self.curr_py
        self.prev_theta = np.arctan2(self.curr_px, self.curr_py)
        self.theta = self.prev_theta

        return self.pos_x, self.pos_y

class Ian(Agent):
    pass # maybe add an agent with momentum

class DiffusionModel:
    # used for saving figure
    fig = None
    ims = None
    
    def __init__(self, A=2, B=200, D=0.1, dt=1, endtime=275, source=Source(), agent=Agent(), agent_start=20):
        self.A = A # A is how much substance you have a time 0
        self.B = B # 200 #20 # B is the decay rate
        self.D = D # D is diffusion rate
        self.dt = dt # timestep
        self.endtime = endtime
        self.t_array = np.arange(start=0, stop=self.endtime, step=self.dt) # consider converting this to linspace
        
        self.source = source
        self.agent = agent
        self.agent_start = agent_start 

    def save(self, save_name, dpi=300):
        if self.fig is None or self.ims is None:
            print("error while saving")
        else:
            print(f"saving animation as {save_name}")
            im_ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, repeat_delay=3000)
            Writer = animation.writers['ffmpeg'] # ['pillow'] can write gifs
            writer = Writer(fps=15, metadata=dict(artist='Me'))
            im_ani.save(save_name, writer=writer, dpi=dpi)


    def source_propagation(self, save_movie=True, include_agent=True, save_name='animation.mp4', n_sources=250):
        # get source movement data and wind data
        # data has been smoothed (via 'linear' interpolation)
        source_x_array, source_y_array, wind_x, wind_y = self.source.get_interpolated_data(self.t_array, n_sources)
        # n_sources = len(self.t_array) # should we do it like this? or would it be better if we allowed for n_sources and then let time keep going by
        
        # delta x and delta y
        # make smaller for smoother grid
        dx = 0.5
        dy = 0.5

        source_x_array = np.floor(source_x_array/dx) # does this NEED to be floored? @ian: maybe un-floor this
        source_y_array = np.floor(source_y_array/dy)

        extra_space = 20
        min_x = np.min(source_x_array) - dx*extra_space; max_x = np.max(source_x_array)+ dx*extra_space
        min_y = np.min(source_y_array) - dy*extra_space; max_y = np.max(source_y_array)+ dy*extra_space

        # print(n_sources) # how many points are there?

        x,y = np.meshgrid(np.arange(min_x, max_x, dx), np.arange(min_y, max_y, dy))

        c = np.zeros(x.shape)
        self.fig = plt.figure(figsize=(5,6))
        self.ims = []
        source_activity = np.zeros(n_sources)
        if include_agent:
            agent_path_x = []
            agent_path_y = []
        
        for t_i in range(0,len(self.t_array)):
            if(t_i%25 == 0):
                print(f"{t_i*100/len(self.t_array)}% done.")
            t = self.t_array[t_i]

            # # what if len(t_array) > n_sources???? we might need to do a min max dealio or just extend source x array to be length of t_array
            if t_i <= n_sources:
                source_activity[:t_i] = self.t_array[:t_i]
            else:
                source_activity[:n_sources] = self.t_array[:n_sources]

            # move agent step here (we need option to not include in movie)
            if include_agent and t>self.agent_start:
                a_x, a_y = self.agent.get_agent_position(c, x, y, dx, dy)
                agent_path_x.append(a_x)
                agent_path_y.append(a_y)

            if save_movie: # ADD ANOTHER OPTION TO WATCH IN REAL TIME vs just save
                plot = plt.pcolormesh(x,y,c, vmin=0, vmax=5)
                title = plt.text(0, max_y+5, f"t={t_i+1}", size=20, horizontalalignment='center', verticalalignment='baseline')
                if include_agent and t>self.agent_start:
                    agent_point = plt.scatter(a_x, a_y, c='black', s=15)
                    agent_path, = plt.plot(agent_path_x, agent_path_y, color='red', linewidth=1)
                    self.ims.append([plot, title, agent_point, agent_path])
                else:
                    self.ims.append([plot, title])
                # we need an evaluation here of how close to the target we are, like do we stop?!
                # also a cost function
                # plt.draw()
                # plt.pause(0.00001)
                # plt.clf()
            c = np.zeros(x.shape)
            for source_i in range(0,n_sources):
                if source_activity[source_i] > 0:
                    curr_t = t - source_activity[source_i]
                    curr_c = ((self.A/(curr_t**0.5))*np.exp(-1*(np.power((x-source_x_array[source_i]-(wind_x[source_i]*curr_t)),2)+
                    np.power((y-source_y_array[source_i])-wind_y[source_i]*curr_t,2))/(4*self.D*curr_t)))*(0.5**(curr_t/self.B))
                    c = c + curr_c
        
        if save_movie:
            self.fig.colorbar(plot)
            plt.xlabel('x')
            plt.ylabel('y')
            self.save(save_name, dpi=200)
        print("done!")

model = DiffusionModel(source=Source(), agent=Curtis(v_multiplier=0.75, strat_probs=[0.9,.1]), endtime=300)
model.source_propagation(save_name='animation_test.mp4', n_sources=250) # n_sources could be smaller
# the source simulation doesn't actually change so maybe we could do multiple agents on same simulation