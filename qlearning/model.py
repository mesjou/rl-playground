import gym

env = gym.make('MountainCar-v0')


if __name__ == "__main__" :
    print(env.action_space)
    print(env.observation_space)
