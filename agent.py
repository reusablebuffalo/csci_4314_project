import numpy as np
import random

class Agent:
    def __init__(self, x=0, y=0, dx=0.5, mem_internal=10, v_multiplier={}):
        self.pos_x = 0
        self.pos_y = 0
        self.v = v_multiplier
        
        self.prev_px = []
        self.prev_py = []
        
        self.curr_px = 0
        self.curr_py = 0
        self.bias_theta = np.pi/2
        self.curr_c_agent = 0
        self.sigma02 = np.pi/32 # np.pi
        self.theta = np.pi*(4.0/3)
    
    def get_v(self, strategy):
        return self.v.get(strategy,1)

    # default agent moves up concentration gradient at speed v
    # this agent works perfectly when wind = 0, but kind of sucks otherwise, also pretty slow.
    def get_agent_position(self, c, x, y, dx, dy):
        fy,fx = np.gradient(c,dx,dx) # fy before fx
        tt = np.floor(x/dx) == np.floor(self.pos_x/dx)
        ttt = np.floor(y/dy) == np.floor(self.pos_y/dy)
        indt = tt&ttt # where is agent
        norm =np.linalg.norm([fx[indt], fy[indt]])
        if norm>0: #0.01
            self.curr_px = fx[indt][0]/norm
            self.curr_py = fy[indt][0]/norm
            self.curr_c_agent  = c[indt]
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.pos_x = self.pos_x + self.get_v('default')*self.curr_px
        self.pos_y = self.pos_y + self.get_v('default')*self.curr_py
        return self.pos_x, self.pos_y
    

class Intermittent(Agent):
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
        strat = np.random.choice(['chemotaxis', 'crw','brw'], p=self.strat_probs)
        if strat == 'chemotaxis':
            return self.chemotaxis(c,x,y,dx,dy)
        elif strat == 'crw':
            return self.crw(c,x,y,dx,dy)
        elif strat == 'brw':
            return self.brw(c,x,y,dx,dy)
        elif strat == 'chemoment':
            return self.chemoment(c,x,y,dx,dy)
        else:
            raise ValueError(f"No strategy called {strat}")

    #this is currently a biased random walk, my mistake
    #we need to change this to a crw

    def brw(self, c, x, y, dx, dy):
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
        self.pos_x = self.pos_x + self.get_v('brw') * self.curr_px
        self.pos_y = self.pos_y + self.get_v('brw') * self.curr_py
        return self.pos_x, self.pos_y

    def crw(self, c, x, y, dx, dy):

        #Changed to have to move in either direction
        choice = np.floor(random.uniform(0,1) * 2) + 1
        if choice == 1:
            self.theta = self.theta + self.sigma02 * random.uniform(0,1)
        elif choice == 2:
            self.theta = self.theta - self.sigma02 * random.uniform(0,1)
        else:
            raise ValueError('no such choice')
        self.curr_px = np.cos(self.theta)
        self.curr_py = np.sin(self.theta)
        self.pos_x = self.pos_x + self.get_v('crw') * self.curr_px
        self.pos_y = self.pos_y + self.get_v('crw') * self.curr_py
        return self.pos_x, self.pos_y

    def chemotaxis(self, c, x, y, dx, dy):
        fy,fx = np.gradient(c,dx,dx) # fy before fx
        tt = np.floor(x/dx) == np.floor(self.pos_x/dx)
        ttt = np.floor(y/dy) == np.floor(self.pos_y/dy)
        indt = tt&ttt # where is agent
        norm =np.linalg.norm([fx[indt], fy[indt]])
        if norm>0: #0.01
            self.curr_px = fx[indt][0]/norm
            self.curr_py = fy[indt][0]/norm
            self.curr_c_agent  = c[indt]
        else:
            self.curr_px = 0 
            self.curr_py = 0
        self.pos_x = self.pos_x + self.get_v('chemotaxis')*self.curr_px
        self.pos_y = self.pos_y + self.get_v('chemotaxis')*self.curr_py
        self.theta = np.arctan2(self.curr_px, self.curr_py)

        return self.pos_x, self.pos_y
    
    def chemoment(self, c, x, y, dx, dy):
        pass

class Ian(Agent):
    pass # maybe add an agent with momentum