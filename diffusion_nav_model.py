# CSCI 4314: Final Project
# Diffusion Navigation Culture

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

class Source:
    def __init__(self, iterations, step):
        x = np.arange(0, step*iterations, step)
        self.trajectory = np.asarray([x, x])
    def next_position(self, i):
        return self.trajectory[i]

class Human(Source):
    pass

class Dog:
    location = None
    def __init__(self):
        pass

class DiffusionModel:
    def __init__(self, iterations, step):
        self.iterations = iterations
        self.step = step
        self.source = Human(self.iterations, self.step)
        self.agent = Dog()
        self.grid = None


def get_data(n):
    wind_x = np.random.uniform(-0.05,0.05,size=n)
    wind_y = np.random.uniform(-0.05,0.05,size=n)
    source_y_array = np.linspace(0,50,n)
    source_x_array = 10*np.sin(np.linspace(-math.pi, math.pi, n))
    return source_x_array, source_y_array, wind_x, wind_y

def source_propagation(A, B, D, t_array, save_movie=True, include_agent=True):
    # get source movement data and wind data
    source_x_array, source_y_array, wind_x, wind_y = get_data(30) # wind_x and wind_y must have same length
    pathXY = np.asarray([source_x_array, source_y_array])
    
    # compute step lengths from data
    step_lengths = np.sqrt(np.sum(np.power(np.diff(pathXY, axis=1), 2), axis=0))
    step_lengths = np.insert(step_lengths, 0, 0)
    
    # interpolate movement between data spointss
    n_sources = 250
    cumulative_length = np.cumsum(step_lengths)
    final_step_locs = np.linspace(0, cumulative_length[-1], n_sources)
    if(n_sources < len(t_array)):
        repeat_final_position = np.repeat(cumulative_length[-1], repeats=(len(t_array)-n_sources))
        final_step_locs = np.append(final_step_locs, repeat_final_position)
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
    # data has been smoothed (via 'linear' interpolation)

    # delta x and delta y
    dx = 0.5
    dy = 0.5

    source_x_array = np.floor(source_x_array/dx) # does this NEED to be floored? @ian: maybe un-floor this
    source_y_array = np.floor(source_y_array/dy)

    min_x = np.min(source_x_array) - dx*100; max_x = np.max(source_x_array)+ dx*100
    min_y = np.min(source_y_array) - dy*100; max_y = np.max(source_y_array)+ dy*100

    print(n_sources)

    x,y = np.meshgrid(np.arange(min_x, max_x, dx), np.arange(min_y, max_y, dy))

    # strategy usage
    percentage_steps_per_strategy1 = 1 #0.5 #(accurate and slow)
    elements_per_strat1 = math.floor(percentage_steps_per_strategy1*len(t_array))
    elements_per_strat2 = len(t_array) - elements_per_strat1
    strategy_array = [np.ones(elements_per_strat1), 2*np.ones(elements_per_strat2)]
    strategy_array = np.random.permutation(strategy_array) # use strategy randomly (according to proportions)


    agent_start = 20 # start agent at t = 20

    c = np.zeros(x.shape)
    for t_i in range(1,len(t_array)):
        t = t_array[t_i]

        source_activity = np.zeros(n_sources) 
        # what if len(t_array) > n_sources???? we might need to do a min max dealio or just extend source x array to be length of t_array
        source_activity[:t_i] = t_array[:t_i]
        if include_agent and t > agent_start:
            # find current up the gradient direction
            pass
        if save_movie:
            plt.pcolormesh(x,y,c, vmin=0, vmax=3.5)
            plt.colorbar()
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
        c = np.zeros(x.shape)
        for source_i in range(0,n_sources):
            if source_activity[source_i] > 0:
                curr_t = t - source_activity[source_i]
                curr_c = ((A/(curr_t**0.5))*np.exp(-1*(np.power((x-source_x_array[source_i]-(wind_x[source_i]*curr_t)),2)+
                np.power((y-source_y_array[source_i])-wind_y[source_i]*curr_t,2))/(4*D*curr_t)))*(0.5**(curr_t/B))
                c = c + curr_c

A = 2 # SOMETHING ABOUT DIFFUSION
B=200 #200 #20 
D=0.1
dt = 1 # timestep
endtime = 275
t_array = np.arange(start=0, stop=endtime, step= dt) # consider converting this to linspace
source_propagation(A,B,D,t_array)
