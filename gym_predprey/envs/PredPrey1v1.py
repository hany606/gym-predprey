# This file is written with taking as a reference: https://github.com/snolfi/longdpole/blob/master/longdpole/envs/longdpole_env.py

import os
import math
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding

import ErPredprey
from gym_predprey.envs import renderWorld

# Env that has dictionary of observations for multi-agent traiing
class PredPrey1v1(gym.Env, MultiAgentEnv):
    def __init__(self):

        self.nrobots = 2
        self.env = ErPredprey.PyErProblem()
        self.action_space = spaces.Box(low=np.array([-1 for i in range(self.env.noutputs*self.nrobots)]),
                                high=np.array([1 for i in range(self.env.noutputs*self.nrobots)]),
                                dtype=np.float32)
        self.observation_space = None
        # print(self.env.ninputs) #24 = 8 (ir) + 9 (camera) + 5 (ground) + 1 (bias) + 1 (time)
        self.ob = np.arange(self.env.ninputs*self.nrobots, dtype=np.float32)
        self.ac = np.arange(self.env.noutputs*self.nrobots, dtype=np.float32)
        self.done = np.arange(1, dtype=np.intc)
        self.objs = np.arange(2000, dtype=np.float64) 

        self.env.copyObs(self.ob)
        self.env.copyAct(self.ac)
        self.env.copyDone(self.done)
        self.seed()

        self.env.copyDobj(self.objs)

        self.num_steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.env.reset() 
        return self.ob


    def step(self, action):
        self.num_steps += 1
        # self.ac[0] = action[0]
        self.ac = action
        reward = self.env.step()
        return self.ob, reward, self.done, {}

    def render(self, mode='human'):
        self.env.render()
        info = f'Step: {self.num_steps}'
        # print(info)
        renderWorld.update(self.objs, info, self.ob, self.ac, None)


# Env that has all observations in one list (train as single agent -> one network but for multi-agent)
class PredPrey1v1Super:
    pass

# Env to train only the predetor
class PredPrey1v1Pred:
    pass

# Env to train only the prey
class PredPrey1v1Prey:
    pass