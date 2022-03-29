from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey

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
from bach_utils.SelfPlay import SelfPlayEnvSB3

class SelfPlayPredEnv(SelfPlayEnvSB3, PredPrey1v1Pred):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui')
        reward_type = kwargs.pop('reward_type', None)
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="prey"
        PredPrey1v1Pred.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.prey_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Pred.reset(self)


class SelfPlayPreyEnv(SelfPlayEnvSB3, PredPrey1v1Prey):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, *args, **kwargs):
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui')
        reward_type = kwargs.pop('reward_type', None)
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)  # env_opponent_name="pred"
        PredPrey1v1Prey.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.pred_policy = self # It replaces the policy for the other agent with the best policy that found during reset (This class have it)

    # Change to search only for the prey
    def reset(self):
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1Prey.reset(self)