import os
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey
from stable_baselines3.ppo.ppo import PPO as sb3PPO

import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sorting as utsrt

# These envs are wrappers for the original environments to be able to train one agent while another agent in the environment is following a specific policy

# Explanation for usage of these environments:
# * We use one of these envs 
#       Let's explain assuming that we are using SelfPlayPredEnv
# * This means that we are training the predator agent in the environment
# * And the other agent (The prey) is following a specific policy
# * self.prey_policy = self -> here we make the prey_policy var in the environment equal to the object itself
# * As during the processing the action vector for the pred agent that is input to the environment
# * We need to generate the action for the prey agent using compute_action function that is defined here in the wrapper
# * This is instead of creating another environemnt for the prey (object from another class), here we just point to the same class but we integrate a function
#     that loads the prey agent model and use compute_action function from the object itself that is identified here
# * Why I have called it compute_action() -> Because in ray/rllib it is compute_action, so prey_policy can be an object from rllib model directly without any changes in the code

# * The action to the pred is being computed by the policy that loads outside this class
# * Then the action is generated and passed to step function
# * Then we need to comput the actions for the opponent based on the loaded model in the reset
# * Computing the actions for the opponent is based on passing the observation from this environment to the model
#       This means that the same observation fed to the pred is fed to the prey
#       This means if we need different observations for both agents, this will not work
#           TODO: How can we make it? -> Sol.: Add get_prey_obs(), get_pred_obs() to PredPreyEvorobot with NotImplemented and implement them only pred and prey envs
#           and instead of passing the self.obs in PredPreyEvorobot._process_action "self.ac[2:] = self.prey_policy.compute_action(self.ob)", we will pass get_prey_obs(self.get_prey_obs())

# TODO: Take care that with every reset we load a model, think about it

OS = False#True

# Parent class for all using SB3 functions to predict
#TODO: Opponent_filename should be changed as it is not only for OS
class SelfPlayEnvSB3:
    def __init__(self, algorithm_class, archive):
        self.algorithm_class = algorithm_class  # algorithm class for the opponent
        self.opponent_policy = None             # The policy itself after it is loaded
        self.opponent_policy_filename = None    # Current loaded policy name -> File -> as it was implement first to be stored on disk (But now in cache=archive) 
        self.target_opponent_policy_filename = None
        self._name = None
        self.archive = archive  # opponent archive
        self.OS = OS

        if(archive is None):
            self.OS = True
        self.states = None
        

    def set_target_opponent_policy_filename(self, policy_filename):
        self.target_opponent_policy_filename = policy_filename

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    # Solution: environment of Pred will have a modified step function that will call the parent environment step and then take what observation it is related to  
    # Compute actions for the opponent agent in the environment (Note: that the action for )
    # This is only be called for the opponent agent
    # TODO: This should be renamed -> compute_opponent_action or opponent_compute_policy-> Change them in PredPrey1v1.py
    def compute_action(self, obs): # the policy
        if self.opponent_policy is None:
            return self.action_space.sample() # return a random action
        else:
            action = None
            if(isinstance(self.opponent_policy, sb3PPO)):
                action, self.states = self.opponent_policy.predict(obs, state=self.states) #it is predict because this is PPO from stable-baselines not rllib
            # if(isinstance(self.opponent_policy, rllibPPO)):
                # action, _ = self.opponent_policy.compute_action(obs) #it is predict because this is PPO from stable-baselines not rllib

            return action

    def _load_opponent(self, opponent_filename):
        # print(f"Wants to load {opponent_filename}")
        # Prevent reloading the same policy again or reload empty policy -> empty policy means random policy
        if opponent_filename is not None and opponent_filename != self.opponent_policy_filename:
            # print("loading model: ", opponent_filename)
            self.opponent_policy_filename = opponent_filename
            if self.opponent_policy is not None:
                del self.opponent_policy
            # if(isinstance(self.algorithm_class, sb3PPO) or isinstance(super(self.algorithm_class), sb3PPO)):
            if(not self.OS):
                self.opponent_policy = self.archive.load(name=opponent_filename, env=self, algorithm_class=self.algorithm_class) # here we load the opponent policy
            if(self.OS):
                self.opponent_policy = self.algorithm_class.load(opponent_filename, env=self) # here we load the opponent policy


    def reset(self):
        self.states = None
        if(self.OS):
            print(f"Reset, env name: {self._name}, OS, target_policy: {self.target_opponent_policy_filename}")
        if(not self.OS):
            print(f"Reset, env name: {self._name}, archive_id: {self.archive.random_id}, target_policy: {self.target_opponent_policy_filename}")
        self._load_opponent(self.target_opponent_policy_filename)

class SelfPlayPredEnv(SelfPlayEnvSB3, PredPrey1v1Pred):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs['seed_val']
        del kwargs['seed_val']
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="prey"
        PredPrey1v1Pred.__init__(self, seed_val=seed_val)
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Pred.reset(self)


class SelfPlayPreyEnv(SelfPlayEnvSB3, PredPrey1v1Prey):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs['seed_val']
        del kwargs['seed_val']
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="pred"
        PredPrey1v1Prey.__init__(self, seed_val=seed_val)
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Prey.reset(self)