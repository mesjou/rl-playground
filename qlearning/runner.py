import agent
import gym
import numpy as np
from wrapper import ObservationDiscretizer

n_states = (50,20)
env = gym.make('MountainCar-v0')
env = ObservationDiscretizer(env,n_states)




def qlearning(episodes = 12000, minep = 1000, alpha = 0.3, n_states = None , gamma = 0.9, epsilon = 0):
    if n_states is None:
        n_states = (50, 20)
    ag = agent.QLearnAgent(env.action_space.n, n_states)
    ag.qtable[- 1] = np.array([[0 for i in range(ag.n_actions)] for j in range(ag.n_states[1])])
    all_reward, episode_length,l, total_episode_length, finished_episodes = 0, 0, 0, 0,0
    for i in range(episodes):# Number of Episodes
        state = env.reset()
        done = False

        # decrease learning rate alpha
        if i <= minep:
            alpha2 = alpha
        elif minep < i < 6 * minep:
            alpha2 = 0.01
        else:
            alpha2 = 0.001

        while not done:  # one episode
            episode_length +=1
            action = ag.act(state,epsilon)
            next_state, reward, done, info = env.step(action)
            ag.learn(action, reward, state, next_state, done, alpha2, gamma)
            state = next_state
            if i > episodes - 5:
                env.render()

        total_episode_length += episode_length
        if episode_length < 200:
            finished_episodes +=1
        if (i%1000 == 0 and i>0) or i+1 == episodes:
            print('In the Episodes', i - 999,'to', i ,'a total of', finished_episodes,
                  'Episodes were finished successfully and Average Episode Length is',
                  total_episode_length/1000)
            total_episode_length, finished_episodes= 0 ,0
        episode_length = 0
    env.close()
    return ag.qtable