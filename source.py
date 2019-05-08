import numpy as np
from scipy import interpolate

class Source:
    def __init__(self):
        pass
    
    def get_interpolated_data(self, t_array, n_sources):
        source_x_array, source_y_array, wind_x, wind_y = self.get_data(100) # wind_x and wind_y must have same length
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
        source_x_array = 10*np.sin(np.linspace(-np.pi, np.pi, n))
        return source_x_array, source_y_array, wind_x, wind_y

class Human(Source):
    def get_data(self, n):
        wind_x = np.random.uniform(-0.05,0.05,size=n)
        wind_y = np.random.uniform(-0.05,0.05,size=n)
        source_y_array = np.linspace(0,50,n)
        source_x_array = 10*np.sin(np.linspace(-np.pi, np.pi, n))
        return source_x_array, source_y_array, wind_x, wind_y

class Tricky(Source):
    def get_data(self, n):
        wind_x = np.random.uniform(-0.05,0.05,size=n)
        wind_y = np.random.uniform(-0.05,0.05,size=n)
        first_half_x = np.zeros(int(n/2))
        second_half_x = np.linspace(0,30,n-int(n/2))
        first_half_y = np.linspace(0,30,int(n/2))
        second_half_y = 30*np.ones(n-int(n/2))
        source_y_array = np.concatenate((first_half_y, second_half_y))
        source_x_array = np.concatenate((first_half_x, second_half_x))
        return source_x_array, source_y_array, wind_x, wind_y
