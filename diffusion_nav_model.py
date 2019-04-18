# CSCI 4314: Final Project
# Diffusion Navigation Culture

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import matplotlib.animation as animation
import random


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

        if(n_sources < len(t_array)): # should target just stop moving but still have an odor?
            repeat_final_position = np.repeat(cumulative_length[-1], repeats=(len(t_array)-n_sources))
            final_step_locs = np.append(final_step_locs, repeat_final_position)
            n_sources = len(t_array)

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
        source_y_array = np.linspace(0,50,n)
        source_x_array = 10*np.sin(np.linspace(-math.pi, math.pi, n))
        return source_x_array, source_y_array, wind_x, wind_y


class Agent:
    def __init__(self, x=0, y=0, dx=0.5, mem_internal=25, v_multiplier=5):
        self.pos_x = 0
        self.pos_y = 0
        self.v = dx*v_multiplier

        # self.position_mem_interval = mem_internal
        # self.prev_x = np.zeros(self.position_mem_interval)
        # self.prev_y = np.zeros(self.position_mem_interval)
        # self.curr_c = np.zeros(self.position_mem_interval)
        # self.dir_bias_coef = dx  
        self.curr_px = 0
        self.curr_py = 0
        self.prev_px = 0 
        self.prev_py = 0
        self.prev_theta = 0
        self.curr_c_agent = 0
        self.sigma02 = np.pi
        self.theta = np.arctan2(0, 0)
    
    # default agent moves up concentration gradient at speed v
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
    #taken partly from hw#2 and orit's starter code

    #i think this is a CRW? at every step

    def get_agent_position(self, c, x, y, dx, dy, t_i):

        #need to change this to a UNIFORM distribution
        choice = random.randint(1,3)
        # print(choice)
        #NEED TO ADD function to switch between two strategies
        #unsure why rand(1,1)
        if choice == 1:
            # print('ENTER')
            self.theta = self.prev_theta + self.sigma02 * random.uniform(0,1)
            # print(self.theta)
        elif choice == 2:
            self.theta = self.prev_theta - self.sigma02 * random.uniform(0,1)
        # else:
        #     pass
        # print(self.theta)
        # print(np.sin(self.theta))
        self.pos_x = self.pos_x + self.v * np.cos(self.theta)
        # print(self.pos_x)
        self.pos_y = self.pos_y + self.v * np.sin(self.theta)
        # print(self.pos_y)
        return self.pos_x, self.pos_y

    #We need to ask Orit how she was using this in her original code
    # def CRW(N, realizations, NreS, v, sigma02, x_initial, y_initial, theta_initial):
    #     x = np.zeros([realizations, n])
    #     y = np.zeros([realizations, n])
    #     theta = zeros([realizations, n])

    #     #no idea what NreS is?
    #     x[:,0] = x_initial
    #     y[:,0] = y_initial
    #     theta[:,0] = theta_initial

    #     for realization_i in range(realizations):
    #         for step_i in range(1, N):
    #             if step_i%NreS == 0:



class Ian(Agent):
    pass # maybe add an agent with momentum

class DiffusionModel:
    # used for saving figure
    fig = None
    ims = None
    
    def __init__(self, A=2, B=200, D=0.1, dt=1, endtime=275, source=Source(), agent=Agent(), agent_start=20):
        self.A = A # SOMETHING ABOUT DIFFUSION; i think its concentration at sourcec
        self.B = B #200 #20 
        self.D = D
        self.dt = dt # timestep
        self.endtime = endtime
        self.t_array = np.arange(start=0, stop=self.endtime, step= self.dt) # consider converting this to linspace

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
        n_sources = len(self.t_array) # should we do it like this? or would it be better if we allowed for n_sources and then let time keep going by
        
        # delta x and delta y
        dx = 0.5
        dy = 0.5

        source_x_array = np.floor(source_x_array/dx) # does this NEED to be floored? @ian: maybe un-floor this
        source_y_array = np.floor(source_y_array/dy)

        extra_space = 20
        min_x = np.min(source_x_array) - dx*extra_space; max_x = np.max(source_x_array)+ dx*extra_space
        min_y = np.min(source_y_array) - dy*extra_space; max_y = np.max(source_y_array)+ dy*extra_space

        # print(n_sources) # how many points are there?

        x,y = np.meshgrid(np.arange(min_x, max_x, dx), np.arange(min_y, max_y, dy))

        # strategy usage
        percentage_steps_per_strategy1 = 1 #0.5 #(accurate and slow)
        elements_per_strat1 = math.floor(percentage_steps_per_strategy1*len(self.t_array))
        elements_per_strat2 = len(self.t_array) - elements_per_strat1
        strategy_array = [np.ones(elements_per_strat1), 2*np.ones(elements_per_strat2)]
        strategy_array = np.random.permutation(strategy_array) # use strategy randomly (according to proportions)


        agent_start = 20 # start agent at t = 20

        c = np.zeros(x.shape)
        self.fig = plt.figure(figsize=(10,12))
        self.ims = []
        for t_i in range(1,len(self.t_array)):
            if(t_i%25 == 0):
                print(f"{t_i*100/len(self.t_array)}% done.")
            t = self.t_array[t_i]
            source_activity = np.zeros(n_sources) 
            # what if len(t_array) > n_sources???? we might need to do a min max dealio or just extend source x array to be length of t_array
            source_activity[:t_i] = self.t_array[:t_i]
            if include_agent and t > agent_start:
                # find current up the gradient direction
                pass
            if save_movie: # ADD ANOTHER OPTION TO WATCH IN REAL TIME vs just save
                plot = plt.pcolormesh(x,y,c, vmin=0, vmax=5)
                title = plt.text(0,max_y+5,f"t={t_i}",size=20,horizontalalignment='center',verticalalignment='baseline')
                if include_agent and t>agent_start:
                    # print(t_i)
                    a_x,a_y = self.agent.get_agent_position(c, x, y, dx, dy, t_i)
                    agent_plot = plt.scatter(a_x,a_y, c='black')
                    self.ims.append([plot,title,agent_plot])
                self.ims.append([plot,title])
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
        self.fig.colorbar(plot)
        plt.xlabel('x')
        plt.ylabel('y')
        self.save(save_name, dpi=200)

model = DiffusionModel(source=Source(), agent=Curtis(v_multiplier=3), endtime=275)
model.source_propagation(save_name='animation_test.mp4', n_sources=250)
