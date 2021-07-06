import gym
import gym_predprey
from time import sleep

env = gym.make('PredPrey-1v1-v0')
env.reset()
for _ in range (1000):
    action = env.action_space.sample()
    # action[0] = 0
    # action[1] = 1
    # action[2] = 1
    # action[3] = 1
    print(action)
    observation, reward, done, info = env.step(action)
    # print(observation)
    # print(reward)
    env.render()
    sleep(0.01)
    if (done):
        break
env.close()