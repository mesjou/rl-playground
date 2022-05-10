import agent
import model
#import qtablememory
import gym
import numpy as np
from wrapper import ObservationDiscretizer



def qlearning(episodes = 5000, minep = 4000, alpha = 0.3 , gamma = 0.9, epsilon = 0):
    env = gym.make('MountainCar-v0')
    ag = agent.QLearnAgent()
    ag.qtable[ag.num_states[0] - 1] = np.array(
        [[0 for i in range(model.env.action_space.n)] for j in range(ag.num_states[1])])
    env = ObservationDiscretizer(env, ag.num_states)
    all_reward, episode_length,l, total_episode_length, finished_episodes = 0, 0, 0, 0,0
    for i in range(episodes):# Number of Episodes
        state = env.reset()
        done = False

        """"
                #decrease epsilon
                if i < minep:
                    epsilon2 = round((-epsilon)*(1/minep)* i + epsilon,2)
                else:
                    epsilon2 = 0
                """
        epsilon2 = 0

        # decrease learning rate alpha
        if i <= minep:
            alpha2 = alpha
            # alpha2 = (alpha-1)*(1/minep)* i + 1
        #elif minep < i < 2 * minep:
        #    alpha2 = (alpha - 1) * (1 / (2 * minep)) * i + 1
        else:
            alpha2 = 0.01

        while not done:  # one episode
            episode_length +=1
            action = ag.act(state,epsilon2)
            obs, reward, done, info = env.step(action)  # step, action decided by act()
            ag.learn(action, reward, state, obs, done, alpha2, gamma)#action, reward, state, obs, done, alpha, gamma
            state = obs
            if i > episodes - 5:
                env.render()

        total_episode_length += episode_length
        if episode_length < 200:
            finished_episodes +=1
        if (i%1000 == 0 and i>0) or i+1 == episodes:
            print('In', (i - 1000, i) ,finished_episodes,'Episodes are finished and Average Episode Length ', total_episode_length/1000)
            total_episode_length, finished_episodes= 0 ,0
        episode_length = 0
    model.env.close()
    return ag.qtable


#Q = qlearning(20000,1000)
