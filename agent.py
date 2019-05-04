import numpy as np
import random

class Agent:
    def __init__(self, x=0, y=0, dx=0.5, mem_internal=10, v_multiplier={}, discount=1):
        self.pos_x = 0
        self.pos_y = 0
        self.v = v_multiplier
        self.step = dx
        self.mem_internal = mem_internal
        self.discount = discount
        
        self.prev_px = []
        self.prev_py = []
        
        self.curr_px = 0
        self.curr_py = 0
        self.bias_theta = np.pi/2
        self.curr_c_agent = 0
        self.sigma02 = np.pi/16 # np.pi
        self.theta = np.pi*(3/4)
    
    def get_v(self, strategy):
        return self.v.get(strategy,1)*self.step

    def get_agent_position(self, c, x, y, dx, dy):
        return self.pos_x, self.pos_y
    

class Intermittent(Agent):
    #my attempt at a correlated random walk
    #taken from a portion of Orit Peleg's starter code
    def __init__(self, strat_probs=[0.25,0.25,0.25,0.25], **kwds):
        """
        :param strat_probs: list of length two for probability for each of two strategies ['chemotaxis','crw']
        """
        super().__init__(**kwds)
        self.strat_probs = strat_probs
    
    def get_agent_position(self, c, x, y, dx, dy):
        strat = np.random.choice(['chemotaxis', 'crw','brw','chemoment'], p=self.strat_probs)

        fy,fx = np.gradient(c,dy,dx) # fy before fx
        tt = np.floor(x/dx) == np.floor(self.pos_x/dx); ttt = np.floor(y/dy) == np.floor(self.pos_y/dy)
        indt = tt&ttt # where is agent
        fx = fx[indt]; fy = fy[indt]
        self.curr_c_agent  = c[indt]
        if self.curr_c_agent < 0.01 or len(self.curr_c_agent)==0: # threshold for chemotaxis
            self.theta = self.theta + np.pi/2
            strat = 'crw'
        elif len(self.prev_px) < 5:
            strat = 'chemotaxis'

        px, py = None, None
        if strat == 'chemotaxis':
            px, py = self.chemotaxis(fx, fy)
        elif strat == 'crw':
            px, py = self.crw()
        elif strat == 'brw':
            px, py = self.brw()
        elif strat == 'chemoment':
            px, py = self.chemoment(fx, fy)
        else:
            raise ValueError(f"No strategy called {strat}")
        
        self.pos_x = self.pos_x + self.get_v(strat) * px
        self.pos_y = self.pos_y + self.get_v(strat) * py
        return self.pos_x, self.pos_y

    def brw(self):
        #Changed to have to move in either direction
        choice = np.floor(random.uniform(0,1) * 2) + 1
        if choice == 1:
            self.theta = self.bias_theta + self.sigma02 * random.uniform(0,1)
        elif choice == 2:
            self.theta = self.bias_theta - self.sigma02 * random.uniform(0,1)
        else:
            raise ValueError('no such choice')
        self.curr_px = np.cos(self.theta)
        self.curr_py = np.sin(self.theta)
        self.prev_px.append(self.curr_px)
        self.prev_py.append(self.curr_py)
        return self.curr_px, self.curr_py

    def crw(self):
        choice = np.floor(random.uniform(0,1) * 2) + 1
        if choice == 1:
            self.theta = self.theta + self.sigma02 * random.uniform(0,1)
        elif choice == 2:
            self.theta = self.theta - self.sigma02 * random.uniform(0,1)
        else:
            raise ValueError('no such choice')
        self.curr_px = np.cos(self.theta)
        self.curr_py = np.sin(self.theta)
        self.prev_px.append(self.curr_px)
        self.prev_py.append(self.curr_py)

        return self.curr_px, self.curr_py

    def chemotaxis(self, fx, fy):
        norm = np.linalg.norm([fx, fy])
        if norm>0: #0.01
            self.curr_px = fx[0]/norm
            self.curr_py = fy[0]/norm
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.prev_px.append(self.curr_px)
        self.prev_py.append(self.curr_py)
        self.theta = np.arctan2(self.curr_py, self.curr_px)
        return self.curr_px, self.curr_py
    
    def chemoment(self, fx, fy):
        norm = np.linalg.norm([fx, fy])
        if norm>0: #0.01
            self.curr_px = fx[0]/norm
            self.curr_py = fy[0]/norm
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.prev_px.append(self.curr_px)
        self.prev_py.append(self.curr_py)
        
        mem_px = self.prev_px[-self.mem_internal:]
        mem_py = self.prev_py[-self.mem_internal:]
        discounts = np.power(self.discount*np.ones(len(mem_px)),np.arange(len(mem_px)))
        discounts = np.flip(discounts)
        mem_px = np.mean(discounts*mem_px)
        mem_py = np.mean(discounts*mem_py)
        norm2 = np.linalg.norm([mem_px,mem_py])
        if norm2>0:
            mem_px = mem_px/norm2
            mem_py = mem_py/norm2
        else:
            mem_px = 0
            mem_py = 0
        
        self.theta = np.arctan2(mem_py, mem_px)
        return mem_px, mem_py

class Ian(Agent):
    pass # maybe add an agent with momentum