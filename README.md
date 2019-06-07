# Always Follow Your (Dog’s) Nose: Searching for an Optimal Strategy for Tracking a Moving Target by Airborne Odor Trail
### Course Project for CSCI 4314: MolecularBio/ Dynamic Models in Biology
#### Authors: Ian Smith and Curtis Lin

*Abstract and Introduction are excerpts from project [paper](./CSCI_Project_Write_Up.pdf)*

##### Abstract
Strategies (in dogs) for finding a moving target using only olfactory information have not been closely examined. 
While previous studies have discovered optimal strategies for finding a hidden non-moving target by odor trails, we simulate an airborne odor plume of a moving target.
We discover a successful intermittent strategy composed of correlated random walks, a form of momentum, and chemotaxis for finding such a target. 
This discovered strategy was largely successful on our modeled environments and the global behavior of the agent following this strategy was similar to the observed behavior of real-life odor-tracking species, like rats and dogs.

#### Introduction
Dogs are very good at tracking prey over long distances (many miles) and they do so by sensing a combination of airborne and depositional odorants.
Airborne odorants diffuse in the air as animals move through an environment and are often modeled according to Gaussian dispersion models and are easily affected by turbulence.
Depositional odorants on the other hand are left behind on the ground as prey (or search and rescue targets) travel through a landscape.
These depositional odorants can be resuspended into the air and detected when disturbed (by wind or other animals). 
In tracking prey (like rabbits) or searching for humans (in the case of search and rescue missions), dogs rely on both of these odorants.

Little is known about the navigational strategies that dogs use in these moving target scenarios, so the goal of this project is to gain some insight into how dogs perform so well at this task. 
In this project, we want to find an optimal strategy for tracking a moving target, prey or human (to be rescued), where the only informational cues are airborne odorants. 
For the purpose of this paper, we will disregard depositional odorants as they are harder to simulate and are dependent on traits of the target/prey because they are subject to variations in shedding patterns and other biophysical features.
The strategies considered will be motivated by what is known about dog olfactory tracking (observed in experiment and anecdotally) and our strategies will be evaluated in simulations.

#### Repository
This repository contains the code for all the simulations that were run during this project.
[`agent.py`](./agent.py) contains all the code for the various agent/dog search strategies.
Given a current position and the concentration of odor at the position the dog makes a decision of where to step.
[`source.py`](./source.py) contains the source class, which determines the path of the source/target as well as what wind that source's odorants are subject to.
[`diffusion_nav_model`](./diffusion_nav_model.py) contains all the code that runs the simulation itself. This involves computing the position of the source/target and agent/dog in every time step and then simulating the diffusion of the source's odor in the environment according to physical equations.
This class also handles creation of `.mp4` animations for watching the simulation.
See a sample successful run of the simulation [here](https://www.youtube.com/watch?v=b6kb-MZIhnY&feature=youtu.be).

##### [Read the full paper here.](./CSCI_Project_Write_Up.pdf)

