import os
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey
from stable_baselines3.ppo.ppo import PPO as sb3PPO

from bach_utils.os import *

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

# Parent class for all using SB3 functions to predict
class SelfPlayEnvSB3:
    def __init__(self, log_dir, algorithm_class, env_opponent_name, opponent_selection="latest", startswith_keyword = "history"):
        self.log_dir = log_dir
        self.algorithm_class = algorithm_class
        self.opponent_policy = None
        self.opponent_policy_filename = None
        self.env_opponent_name = env_opponent_name
        self.target_opponent_policy_filename = None
        self.opponent_selection = opponent_selection
        self.startswith_keyword = startswith_keyword
        

    def set_target_opponent_policy_filename(self, policy_filename):
        self.target_opponent_policy_filename = policy_filename

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    # Compute actions for the opponent agent in the environment (Note: that the action for )
    # This is only be called for the opponent agent
    def compute_action(self, obs): # the policy
        if self.opponent_policy is None:
            return self.action_space.sample() # return a random action
        else:
            action = None
            if(isinstance(self.opponent_policy, sb3PPO)):
                action, _ = self.opponent_policy.predict(obs) #it is predict because this is PPO from stable-baselines not rllib
            # if(isinstance(self.opponent_policy, rllibPPO)):
                # action, _ = self.opponent_policy.compute_action(obs) #it is predict because this is PPO from stable-baselines not rllib

            return action

    def _sample_opponent(self):        
        loading_path = os.path.join(self.log_dir, self.env_opponent_name)
        files_list = get_startswith(loading_path, self.startswith_keyword)
        # print(f"Found {len(files_list)} files in {loading_path}")
        opponent_filename = None
        if(len(files_list) > 0):
            if(self.opponent_selection == "random"):
                opponent_filename = get_random_from(files_list)[0]
            
            elif(self.opponent_selection == "latest"):
                sort_steps(files_list) # TODO: Take care about this computation bottelneck here O(NlogN), it will be a headach for large archives
                latest = files_list[-1]
                opponent_filename = latest

            elif(self.opponent_selection == "highest"):
                sort_metric(files_list)
                target = files_list[-1]
                opponent_filename = target

            elif(self.opponent_selection == "lowest"):
                sort_metric(files_list)
                target = files_list[0]
                opponent_filename = target

            opponent_filename = os.path.join(self.log_dir, self.env_opponent_name, opponent_filename)
            return opponent_filename

    def _load_opponent(self, opponent_filename):
        # print(f"Wants to load {opponent_filename}")
        if opponent_filename is not None and opponent_filename != self.opponent_policy_filename:
            # print("loading model: ", opponent_filename)
            self.opponent_policy_filename = opponent_filename
            if self.opponent_policy is not None:
                del self.opponent_policy
            # if(isinstance(self.algorithm_class, sb3PPO) or isinstance(super(self.algorithm_class), sb3PPO)):
            self.opponent_policy = self.algorithm_class.load(opponent_filename, env=self) # here we load the opponent policy


    def reset(self):
        if(self.target_opponent_policy_filename is not None):
            self._load_opponent(self.target_opponent_policy_filename)
            self.target_opponent_policy_filename = None # as if we want to have a specific opponent policy, we need to set it before each reset
        else:
            opponent_filename = self._sample_opponent()
            self._load_opponent(opponent_filename)

class SelfPlayPredEnv(SelfPlayEnvSB3, PredPrey1v1Pred):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        SelfPlayEnvSB3.__init__(self, *args, **kwargs, env_opponent_name="prey")
        PredPrey1v1Pred.__init__(self)
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Pred.reset(self)


class SelfPlayPreyEnv(SelfPlayEnvSB3, PredPrey1v1Prey):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        SelfPlayEnvSB3.__init__(self, *args, **kwargs, env_opponent_name="pred")
        PredPrey1v1Prey.__init__(self)
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Prey.reset(self)



# import os
# from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
# from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey

# class SelfPlayEnvSB3:
#     def __init__(self, **kwargs):
#         self.log_dir = kwargs["log_dir"]
#         self.algorithm_class = kwargs["algorithm_class"]
#         self.best_model = None
#         self.best_model_filename = None
#         self.env_opponent_name = kwargs["env_opponent_name"]

#     # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
#     def compute_action(self, obs): # the policy
#         if self.best_model is None:
#             return self.action_space.sample() # return a random action
#         else:
#             action, _ = self.best_model.predict(obs) # it is predict because this is PPO from stable-baselines not rllib
#             return action

#     def reset(self):
#         # load model if it's there
#         modellist = [f for f in os.listdir(os.path.join(self.log_dir, self.env_opponent_name)) if f.startswith("history")]
#         modellist.sort()
#         if len(modellist) > 0:
#             filename = os.path.join(self.log_dir, self.env_opponent_name, modellist[-1]) # the latest model
#             if filename != self.best_model_filename:
#                 print("loading model: ", filename)
#                 self.best_model_filename = filename
#                 if self.best_model is not None:
#                     del self.best_model
#                 self.best_model = self.algorithm_class.load(filename, env=self)

# class SelfPlayPredEnv(SelfPlayEnvSB3, PredPrey1v1Pred):
#     # wrapper over the normal single player env, but loads the best self play model
#     def __init__(self, **kwargs):
#         kwargs["env_opponent_name"] = "prey"
#         SelfPlayEnvSB3.__init__(self, **kwargs)
#         PredPrey1v1Pred.__init__(self)
#         self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

#     # Change to search only for the prey
#     def reset(self):
#         SelfPlayEnvSB3.reset(self)
#         return PredPrey1v1Pred.reset(self)


# class SelfPlayPreyEnv(SelfPlayEnvSB3, PredPrey1v1Prey):
#     # wrapper over the normal single player env, but loads the best self play model
#     def __init__(self, **kwargs):
#         kwargs["env_opponent_name"] = "pred"
#         SelfPlayEnvSB3.__init__(self, **kwargs)
#         PredPrey1v1Prey.__init__(self)
#         self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

#     # Change to search only for the prey
#     def reset(self):
#         SelfPlayEnvSB3.reset(self)
#         return PredPrey1v1Prey.reset(self)