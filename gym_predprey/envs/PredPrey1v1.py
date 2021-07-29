# This file is written with taking as a reference: https://github.com/snolfi/longdpole/blob/master/longdpole/envs/longdpole_env.py
# Predetor, Prey -> Predetor is red, Prey is green

import sys
sys.path
sys.path.append('/home/hany606/repos/research/Drones-PEG-Bachelor-Thesis-2022/2D/gym-predprey/predpreylib')
# print(sys.path)

import os
import math
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding

import ErPredprey

class Behavior: # For only prey for now, we need to make it configured for the predator also :TODO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if(isinstance(action, dict)):
            action[1] = [0, 0]
        else:
            action[2:] = [0, 0]
        return action
        
    def cos_1D(self, action, time, observation):
        freq = 5#self.kwargs["freq"]
        amplitude = 1#self.kwargs["amplitude"]
        if(isinstance(action, dict)):
            action[1] = [0, 0]
        else:
            sin_wave = amplitude*np.cos(time/freq)
            action[2:] = [sin_wave, sin_wave]
        return action


# TODO: make behavior to take from a network and infer the observations to get the actions (behavior)

# TODO: Check decorators and lambda functions to do that in a good way
# class Behavior:
#     @staticmethod
#     def fixed_prey(action, time):
#         if(isinstance(action, dict)):
#             action[1] = [0, 0]
#         else:
#             action[2:] = [0, 0]
#         return action
#     @staticmethod
#     def cos(freq=20, amplitude=1):
#         def cos_1D(action, time):
#             if(isinstance(action, dict)):
#                 action[1] = [0, 0]
#             else:
#                 sin_wave = amplitude*np.cos(time/freq)
#                 action[2:] = [sin_wave, sin_wave]
#             return action
    
# This is exactly the same as the one used with the evorobotpy2
class PredPreyEvorobot(gym.Env):
    def __init__(self, max_num_steps=1000, pred_behavior=None, prey_behavior=None, pred_policy=None, prey_policy=None):
        self.nrobots = 2
        self.env = ErPredprey.PyErProblem()
        # print(self.env.ninputs) #24 = 8 (ir) + 9 (camera) + 5 (ground) + 1 (bias) + 1 (time)
        self.ob = np.arange(self.env.ninputs * self.nrobots, dtype=np.float32)
        self.ac = np.arange(self.env.noutputs * self.nrobots, dtype=np.float32)
        self.done = np.arange(1, dtype=np.int32)
        self.objs = np.arange(max_num_steps, dtype=np.float64)
        self.objs[0] = -1

        self.env.copyObs(self.ob)
        self.env.copyAct(self.ac)
        self.env.copyDone(self.done)
        self.env.copyDobj(self.objs)
        self.seed()

        self.action_space      = spaces.Box(low=np.array([-1 for _ in range(self.env.noutputs)]),
                                            high=np.array([1 for _ in range(self.env.noutputs)]),
                                            dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0 for _ in range(self.env.ninputs)]),
                                            high=np.array([1000 for _ in range(self.env.ninputs)]),
                                            dtype=np.float64)


        self.num_steps = 0
        self.max_num_steps = max_num_steps
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy
        self.prey_policy = prey_policy
        self.caught = False
        self.steps_done = False

    def reinit(self, max_num_steps=1000, prey_behavior=None):
        self.max_num_steps = max_num_steps
        self.prey_behavior = prey_behavior

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.env.reset()
        self.num_steps = 0
        ob = self._process_observation()
        return ob

    # get action from the network
    # process the action to be passed to the environment
    def _process_action(self, action):
        self.ac = deepcopy(action)
        self.ac = np.array(self.ac, dtype=np.float32)
        if(self.prey_behavior is not None):
            # print(self.ac)
            # print("----")
            self.ac = self.prey_behavior(self.ac, self.num_steps, self.ob)
            # print(self.ac)
        if(self.pred_policy is not None):
            # print("policy pred")
            self.ac[:2] = self.pred_policy.compute_action(self.ob)
        if(self.prey_policy is not None):
            # print("policy prey")
            self.ac[2:] = self.prey_policy.compute_action(self.ob)
        # self.ac = self.ac if self.prey_behavior is None else self.prey_behavior(self.ac, self.num_steps, self.ob)

        # print(self.ac)
        self.env.copyAct(self.ac)

    def _process_observation(self):
        self.env.copyObs(self.ob)
        return self.ob

    def _process_reward(self, ob, returned_reward, done):
        return returned_reward

    def _process_done(self):
        self.env.copyDone(self.done)
        self.caught = self.done[0]
        self.steps_done = self.num_steps > self.max_num_steps
        # print(self.steps_done, self.caught)
        return True if self.caught or self.steps_done else False

    def _process_info(self):
        return {}

    def step(self, action):
        self.num_steps += 1
        self._process_action(action)        # self.ac has changed
        reward = self.env.step()
        ob = self._process_observation()    # self.ob has changed
        if(ob.shape != self.observation_space.shape):
            raise ValueError("Observation space is incorrect")
        done = self._process_done()         # self.done has changed
        reward = self._process_reward(ob, reward, self.done[0])
        info = self._process_info()
        return ob, reward, done, info

    def render(self, mode='human'):
        from gym_predprey.envs import renderWorld
        self.env.render()
        info = f'Step: {self.num_steps}'
        renderWorld.update(deepcopy(self.objs), info, deepcopy(self.ob), deepcopy(self.ac), None)

# Env that has dictionary of observations for multi-agent training
# This is made in context of multi-agent system
class PredPrey1v1(PredPreyEvorobot, gym.Env, MultiAgentEnv):
    def __init__(self, **kwargs):
        PredPreyEvorobot.__init__(self, **kwargs)
        self.action_space      = spaces.Dict({i: spaces.Box(low=np.array([-1 for _ in range(self.env.noutputs)]),
                                                 high=np.array([1 for _ in range(self.env.noutputs)]),
                                                 dtype=np.float32) for i in range(self.nrobots)})
        self.observation_space = spaces.Dict({i: spaces.Box(low=np.array([0     for _ in range(self.env.ninputs)]),
                                                 high=np.array([1000 for _ in range(self.env.ninputs)]),
                                                 dtype=np.float64) for i in range(self.nrobots)})
        self.caught = False

    def _process_action(self, action):
        action = np.array([action[0], action[1]]).flatten()
        PredPreyEvorobot._process_action(self, action)

    def _process_observation(self):
        ob = PredPreyEvorobot._process_observation(self)
        ninputs = self.env.ninputs
        nrobots = self.nrobots
        return {i: ob[i*ninputs: i*ninputs + ninputs] for i in range(nrobots)}

    # Adapted from: https://github.com/openai/multiagent-competition/blob/master/gym-compete/gym_compete/new_envs/you_shall_not_pass.py
    # Two teams: predetor and prey
    # Prey needs to run away and the predetor needs to catch it
    # Rewards:
    #   If predetor caught the prey:
    #       It is finished and predetor gets +1000 and prey -1000
    #   If the predetor did not catch the prey:
    #       The prey gets +10 and the predetor -10
    #   if the episode finished the prey get +1000 and predetor -1000

    # OpenAI human blocker reward function
    #     Some Walker reaches end:
    #         walker which did touchdown: +1000
    #         all blockers: -1000
    #     No Walker reaches end:
    #         all walkers: -1000
    #         if blocker is standing:
    #             blocker gets +1000
    #         else:
    #             blocker gets 0
    def _process_reward(self, ob, _, done):
        # if the predetor(pursuer) caught the prey(evader) then the predetor takes good reward and done
        # if the predetor couldn't catch the prey then it will take negative reward
        prey_reward = 10
        predetor_reward = -10
        # dist = np.linalg.norm(ob[0] - ob[1]) # this was made if the agent returned x and y positions
        # eps = 200
        # print(f"distance: {dist}")
        # if (dist < eps):
        if(done["__all__"]):   # if the predetor caught the prey
            # self.caught = True
            prey_reward = -1000
            predetor_reward = 1000
        if(self.num_steps >= self.max_num_steps):
            prey_reward = 1000
            predetor_reward = -1000
        # "predetor": 0, "prey": 1
        return {0:predetor_reward, 1:prey_reward}

    def _process_done(self):
        self.caught = PredPreyEvorobot._process_done(self)
        bool_val = True if(self.caught or self.num_steps >= self.max_num_steps) else False
        done = {i: bool_val for i in range(self.nrobots)}
        done["__all__"] = True if True in done.values() else False
        return done

    def _process_info(self):
        return {i: {} for i in range(self.nrobots)}

# Env that has all observations in one list (train as single agent -> one network but for multi-agent)
class PredPrey1v1Super(PredPreyEvorobot, gym.Env):
    def __init__(self, **kwargs):
        PredPreyEvorobot.__init__(self, **kwargs)
        self.action_space      = spaces.Box(low=np.array([-1 for _ in range(self.env.noutputs * self.nrobots)]),
                                            high=np.array([1 for _ in range(self.env.noutputs * self.nrobots)]),
                                            dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.env.ninputs * self.nrobots)]),
                                            high=np.array([1000 for _ in range(self.env.ninputs * self.nrobots)]),
                                            dtype=np.float64)

    # # TODO: think about it more and read about it
    # def _process_reward(self, ob, returned_reward, done):
    #     prey_reward = 1
    #     predetor_reward = -1
    #     # dist = np.linalg.norm(ob[0] - ob[1]) # this was made if the agent returned x and y positions
    #     # eps = 200
    #     # print(f"distance: {dist}")
    #     # if (dist < eps):
    #     if(done):   # if the predetor caught the prey
    #         # self.caught = True
    #         prey_reward = -10
    #         predetor_reward = 10
    #     if(self.num_steps >= self.max_num_steps):
    #         prey_reward = 10
    #         predetor_reward = -10            
    #     return predetor_reward#[prey_reward, predetor_reward]

class PredPrey1v1Pred(PredPreyEvorobot, gym.Env):
    def __init__(self, **kwargs):
        PredPreyEvorobot.__init__(self, **kwargs)
        print(f"Predator ---- Prey policy: {True if self.prey_policy is not None else False}")
        self.action_space      = spaces.Box(low=np.array([-1 for _ in range(self.env.noutputs)]),
                                            high=np.array([1 for _ in range(self.env.noutputs)]),
                                            dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.env.ninputs)]),
        #                                     high=np.array([1000 for _ in range(self.env.ninputs)]),
        #                                     dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.env.ninputs*self.nrobots)]),
                                            high=np.array([1000 for _ in range(self.env.ninputs*self.nrobots)]),
                                            dtype=np.float64)


    def _process_action(self, action):
        if(self.prey_behavior is None and self.prey_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        action = np.array([action, [0,0]]).flatten()
        PredPreyEvorobot._process_action(self, action)

    # if we removed this override and changed the observation space, the agent will know the other's observations
    def _process_observation(self):
        # Knows nothing from the observations from the other agent info
        ob = PredPreyEvorobot._process_observation(self)
        # print(self.env.ninputs)
        # return ob[:24]#self.env.ninputs]
        return ob

    # TODO: think about it more and read about it
    def _process_reward(self, ob, returned_reward, done):
        # Dense reward based on catching without taking into consideration the distance between them
        prey_reward = 1
        predetor_reward = -1
        # dist = np.linalg.norm(ob[0] - ob[1]) # this was made if the agent returned x and y positions
        # eps = 200
        # print(f"distance: {dist}")
        # if (dist < eps):
        if(self.caught):   # if the predetor caught the prey
            # self.caught = True
            prey_reward = -10
            predetor_reward = 10
        if(self.steps_done):
            prey_reward = 10
            predetor_reward = -10            
        return predetor_reward
        
# Env to train only the prey -> Pred is following a policy or behavior
class PredPrey1v1Prey(PredPreyEvorobot, gym.Env):
    def __init__(self, **kwargs):
        PredPreyEvorobot.__init__(self, **kwargs)
        print(f"Prey ---- Pred policy: {True if self.pred_policy is not None else False}")
        self.action_space      = spaces.Box(low=np.array([-1 for _ in range(self.env.noutputs)]),
                                            high=np.array([1 for _ in range(self.env.noutputs)]),
                                            dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.env.ninputs)]),
        #                                     high=np.array([1000 for _ in range(self.env.ninputs)]),
        #                                     dtype=np.float64)

        self.observation_space = spaces.Box(low=np.array([0     for _ in range(self.env.ninputs*self.nrobots)]),
                                            high=np.array([1000 for _ in range(self.env.ninputs*self.nrobots)]),
                                            dtype=np.float64)


    def _process_action(self, action):
        if(self.pred_behavior is None and self.pred_policy is None):
            raise ValueError("prey_behavior or prey_policy should be specified")

        action = np.array([[0,0], action]).flatten()
        PredPreyEvorobot._process_action(self, action)

    # if we removed this override and changed the observation space, the agent will know the other's observations
    def _process_observation(self):
        # Knows nothing from the observations from the other agent info
        ob = PredPreyEvorobot._process_observation(self)
        # print(self.env.ninputs)
        # return ob[24:]#self.env.ninputs]
        return ob

    # TODO: think about it more and read about it
    def _process_reward(self, ob, returned_reward, done):
        # Dense reward based on catching without taking into consideration the distance between them
        prey_reward = 1
        predetor_reward = -1
        # dist = np.linalg.norm(ob[0] - ob[1]) # this was made if the agent returned x and y positions
        # eps = 200
        # print(f"distance: {dist}")
        # if (dist < eps):
        if(self.caught):   # if the predetor caught the prey
            # self.caught = True
            prey_reward = -10
            predetor_reward = 10
        if(self.steps_done):
            prey_reward = 10
            predetor_reward = -10            
        return prey_reward