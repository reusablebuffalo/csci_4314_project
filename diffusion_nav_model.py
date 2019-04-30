# CSCI 4314: Final Project
# Diffusion Navigation Simulation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from source import Source, Human
from agent import Agent, Intermittent

class DiffusionModel:
    # used for saving figure
    fig = None
    ims = None
    # delta x and delta y
    # make smaller for smoother grid
    dx = 0.5
    dy = 0.5
    max_y = 115
    
    def __init__(self, A=2, B=400, D=0.1, dt=1, endtime=275, source=Source(), agent=Agent(), agent_start=20):
        self.A = A # A is how much substance you have a time 0
        self.B = B # 200 #20 # B is the decay rate
        self.D = D # D is diffusion rate # 0.1 default
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
            writer = Writer(fps=24, metadata=dict(artist='Me'))
            im_ani.save(save_name, writer=writer, dpi=dpi)


    def source_propagation(self, n_sources=250):
        # get source movement data and wind data
        # data has been smoothed (via 'linear' interpolation)
        source_x_array, source_y_array, wind_x, wind_y = self.source.get_interpolated_data(self.t_array, n_sources)
        # n_sources = len(self.t_array) # should we do it like this? or would it be better if we allowed for n_sources and then let time keep going by
        
        source_x_array = np.floor(source_x_array/self.dx) # does this NEED to be floored? @ian: maybe un-floor this
        source_y_array = np.floor(source_y_array/self.dy)

        extra_space = 20
        min_x = np.min(source_x_array) - self.dx*extra_space; max_x = np.max(source_x_array)+ self.dx*extra_space
        min_y = np.min(source_y_array) - self.dy*extra_space; self.max_y = np.max(source_y_array)+ self.dy*extra_space

        # print(n_sources) # how many points are there?

        x,y = np.meshgrid(np.arange(min_x, max_x, self.dx), np.arange(min_y, self.max_y, self.dy))

        c = np.zeros(x.shape)
        source_activity = np.zeros(n_sources)
        s_x = list(source_x_array)
        s_y = list(source_y_array)
        cs = [c]
        for t_i in range(0,len(self.t_array)):
            if(t_i%25 == 0):
                print(f"Source propagation: {t_i*100/len(self.t_array)}% done.")
            t = self.t_array[t_i]

            if t_i < n_sources:
                source_activity[:t_i] = self.t_array[:t_i]
            else:
                source_activity[:n_sources] = self.t_array[:n_sources]
                s_x.append(source_x_array[-1])
                s_y.append(source_y_array[-1])
            c = np.zeros(x.shape)
            for source_i in range(0,n_sources):
                if source_activity[source_i] > 0:
                    curr_t = t - source_activity[source_i]
                    curr_c = ((self.A/(curr_t**0.5))*np.exp(-1*(np.power((x-source_x_array[source_i]-(wind_x[source_i]*curr_t)),2)+
                    np.power((y-source_y_array[source_i])-wind_y[source_i]*curr_t,2))/(4*self.D*curr_t)))*(0.5**(curr_t/self.B))
                    c = c + curr_c
            cs.append(c)
        print("saving")
        np.savez(f"diffusion.npz", x=x, y=y, cs=np.asarray(cs), s_x=s_x, s_y=s_y)
    
    @staticmethod
    def dist(x1,y1,x2,y2):
        return np.floor(np.sqrt((x1-x2)**2 + (y1-y2)**2))

    def run_simulation(self, save_movie=True, include_agent=True, save_name='animation.mp4', n_sources=250, new_source_prop=False):
        try:
            if new_source_prop:
                raise IOError
            data = np.load(f"diffusion.npz")
            print("No Source Simulation Necessary")
        except IOError as error:
            print("Propagating Source")
            self.source_propagation(n_sources=n_sources)
            data = np.load(f"diffusion.npz")
        finally:
            x = data['x']
            y = data['y']
            cs = data['cs']
            if cs.shape[0] < len(self.t_array):
                raise Exception("need new_source_prop")
            s_x = data['s_x']
            s_y = data['s_y']

            self.fig = plt.figure(figsize=(10,12))
            self.ims = []            
            if include_agent:
                agent_path_x = []
                agent_path_y = []
            d = None
            t = None
            for t_i in range(0,len(self.t_array)):
                if(t_i%25 == 0):
                    print(f"Agent Tracking: {t_i*100/len(self.t_array)}% done.")
                    print(d,t)
                t = self.t_array[t_i]
                c = cs[t_i]
                # move agent step here (we need option to not include in movie)
                if include_agent and t>self.agent_start:
                    a_x, a_y = self.agent.get_agent_position(c, x, y, self.dx, self.dy)
                    agent_path_x.append(a_x)
                    agent_path_y.append(a_y)
                    d = self.dist(a_x,a_y,s_x[t_i],s_y[t_i])
                    if d < 5:
                        break
                if save_movie: # ADD ANOTHER OPTION TO WATCH IN REAL TIME vs just save
                    plot = plt.pcolormesh(x,y,c, vmin=0, vmax=4)
                    if include_agent and t>self.agent_start:
                        title = plt.text(0, self.max_y, f"t={t_i}, d={d}", size=20, horizontalalignment='center', verticalalignment='baseline')
                        agent_point = plt.scatter(a_x, a_y, c='black', s=15)
                        agent_path, = plt.plot(agent_path_x, agent_path_y, color='red', linewidth=1)
                        self.ims.append([plot, title, agent_point, agent_path])
                    else:
                        title = plt.text(0, self.max_y, f"t={t_i+1}", size=20, horizontalalignment='center', verticalalignment='baseline')
                        self.ims.append([plot, title])
            if save_movie:
                self.fig.colorbar(plot)
                plt.xlabel('x')
                plt.ylabel('y')
                if include_agent:
                    self.fig.legend([agent_path],[f"p(chemo)={self.agent.strat_probs[0]}\np(crw)={self.agent.strat_probs[1]}\np(brw)={self.agent.strat_probs[2]}\nv={self.agent.v}"],loc='lower right')
                self.save(save_name, dpi=200)
            # print("done!")
            return d, t

min_i, min_d = 10000, 10000
# for i in np.linspace(0,1,20):
#     model = DiffusionModel(source=Source(), agent=Curtis(v_multiplier={'chemotaxis':1,'crw':2, 'brw':2}, strat_probs=[i,1-i,0]), endtime=400)
#     d,t = model.run_simulation(save_name='animation_test.mp4', new_source_prop=False, save_movie=False) # n_sources could be smaller
#     if d < min_d:
#         min_i = i
#         min_d = d 
# print(min_i, min_d)

model = DiffusionModel(source=Human(), agent=Intermittent(v_multiplier={'chemotaxis':1,'crw':4, 'brw':6}, strat_probs=[0.85,0.15,0]), endtime=250)
d,t = model.run_simulation(save_name='animation_test.mp4', new_source_prop=False, save_movie=True, n_sources=250) # n_sources could be smaller
print(d,t)

# 0.01 for wind, brw: 3, 0.9, 0.1s