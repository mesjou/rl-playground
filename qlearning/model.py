""""
import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(100):
    env.render()
    a = env.step(env.action_space.sample()) # take a random action
    print(a)
env.close()
"""""

#import distutils.spawn
import gym
import numpy as np


env = gym.make('MountainCar-v0')


if __name__ == "__main__" :
    print(env.action_space)
    print(env.observation_space)


#num_states = (env.observation_space.high - env.observation_space.low)*\np.array([10, 100])
#print('num',num_states)