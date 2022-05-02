import agent
import model
#import qtablememory
import gym
import numpy as np



test = agent.QLearnAgent()

def qlearning(episodes = 5000, ag = test, alpha = 0.3 , gamma = 0.9, epsilon = 0.1):
    #ag= agent.QLearnAgent()
    rewardlist = []
    all_reward, episode_length,l, total_episode_length, finished_episodes = 0, 0, 0, 0,0
    for i in range(episodes):# Number of Episodes
        observation = model.env.reset()
        ag.done = False
        ag.discretize_state(observation)
        if i < episodes * 1/8:
            epsilon2 = 0.7
            #epsilon2 = (epsilon-1)*(4/episodes)* i + 1
        else:
            epsilon2 = epsilon
        while not ag.done:  # one episode
            episode_length +=1
            ag.act(epsilon2)
            ag.learn(alpha, gamma,l)
            if i > episodes - 10:
                l=1
                model.env.render()
            if ag.reward != None:
                all_reward += ag.reward
                if ag.reward not in rewardlist:
                    rewardlist.append(ag.reward)
        total_episode_length += episode_length
        if episode_length < 200:
            finished_episodes +=1
        if i%1000 == 0 and i>0:
            print('In', (i - 1000, i) ,finished_episodes,'Episodes are finished and Average Episode Length ', episode_length/1000)
            #print('Average Reward in', (i-100,i), 'is', all_reward/100)
            all_reward, finished_episodes= 0 ,0
        episode_length = 0
    #print(ag.qtable)
    #print(rewardlist)
    #return ag.qtable



Q = qlearning(10000)

"""
        observation = model.env.reset()
        ag.discretize_state(observation)
        while not ag.done:#one episode
            action = ag.act()
            oldq = ag.qtable[ag.obs[0]][ag.obs[1]][action]
            observation, ag.reward, ag.done, info = model.env.step(action)
            model.env.render()
            ag.discretize_state(observation)
            newq =  round(oldq + alpha * (ag.reward + gamma * np.max(ag.qtable[ag.obs[0]][ag.obs[1]])-oldq), 4)
            print('newq', newq)
            print('pos',ag.obs[0])
            print('velo',ag.obs[1] )
            ag.qtable[ag.obs[0]][ag.obs[1]][action] = newq
"""



