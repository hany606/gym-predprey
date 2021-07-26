import os
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey

class SelfPlayEnvSB3:
    def __init__(self, log_dir, algorithm_class, env_opponent_name):
        self.log_dir = log_dir
        self.algorithm_class = algorithm_class
        self.best_model = None
        self.best_model_filename = None
        self.env_opponent_name = env_opponent_name

    # TODO: This works fine for identical agents but different agents will not work as they won't have the same action spaec
    def compute_action(self, obs): # the policy
        if self.best_model is None:
            return self.action_space.sample() # return a random action
        else:
            action, _ = self.best_model.predict(obs) # it is predict because this is PPO from stable-baselines not rllib
            return action

    def reset(self):
        # load model if it's there
        modellist = [f for f in os.listdir(os.path.join(self.log_dir, self.env_opponent_name)) if f.startswith("history")]
        modellist.sort()
        if len(modellist) > 0:
            filename = os.path.join(self.log_dir, self.env_opponent_name, modellist[-1]) # the latest model
            if filename != self.best_model_filename:
                print("loading model: ", filename)
                self.best_model_filename = filename
                if self.best_model is not None:
                    del self.best_model
                self.best_model = self.algorithm_class.load(filename, env=self)

class SelfPlayPredEnv(SelfPlayEnvSB3, PredPrey1v1Pred):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, log_dir, algorithm_class):
        SelfPlayEnvSB3.__init__(self, log_dir, algorithm_class, env_opponent_name="prey")
        PredPrey1v1Pred.__init__(self)
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Pred.reset(self)


class SelfPlayPreyEnv(SelfPlayEnvSB3, PredPrey1v1Prey):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, log_dir, algorithm_class):
        SelfPlayEnvSB3.__init__(self, log_dir, algorithm_class, env_opponent_name="pred")
        PredPrey1v1Prey.__init__(self)
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Prey.reset(self)



#         import os
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