import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PredPrey-1v1-v0',
    entry_point='gym_predprey.envs:PredPrey1v1',
)

register(
    id='PredPrey-Superior-1v1-v0',
    entry_point='longdpole.envs:PredPrey1v1Super',
)

register(
    id='PredPrey-pred-v0',
    entry_point='longdpole.envs:PredPrey1v1Pred',
)

register(
    id='PredPrey-prey-v0',
    entry_point='longdpole.envs:PredPrey1v1Prey',
)