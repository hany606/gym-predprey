import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PredPrey-Evorobot-1v1-v0',
    entry_point='gym_predprey.envs:PredPreyEvorobot',
)

# register(
#     id='PredPrey-1v1-v0',
#     entry_point='gym_predprey.envs:PredPrey1v1',
# )

# register(
#     id='PredPrey-Superior-1v1-v0',
#     entry_point='gym_predprey.envs:PredPrey1v1Super',
# )

register(
    id='PredPrey-Pred-v0',
    entry_point='gym_predprey.envs:PredPrey1v1Pred',
)

register(
    id='PredPrey-Prey-v0',
    entry_point='gym_predprey.envs:PredPrey1v1Prey',
)


register(
    id='SelfPlay1v1-Pred-v0',
    entry_point='gym_predprey.envs:SelfPlayPredEnv',
    kwargs={"log_dir":None, "algorithm_class":None}
)

register(
    id='SelfPlay1v1-Prey-v0',
    entry_point='gym_predprey.envs:SelfPlayPreyEnv',
    kwargs={"log_dir":None, "algorithm_class":None}

)