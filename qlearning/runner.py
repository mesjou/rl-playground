import agent
import model
import qtablememory
import gym
import numpy as np

def qlearning(episodes = 5000, Qtable0 = None, alpha = 0.3 , gamma = 0.9, epsilon = 0.1):
    ag = agent.QLearnAgent()
    print('START OF QLEARN')
    rewardlist = []
    all_reward = 0
    for i in range(episodes):# Number of Episodes
        observation = model.env.reset()
        ag.done = False
        ag.discretize_state(observation)
        av_reward = 0
        if i < episodes * 1/4:
            epsilon2 = 0.7
        else:
            epsilon2 = epsilon
        while not ag.done:  # one episode
            ag.act(epsilon2)
            ag.learn(alpha, gamma, epsilon2)
            if i > episodes - 10:
                model.env.render()
            if ag.reward != None:
                all_reward += ag.reward
                if ag.reward not in rewardlist:
                    rewardlist.append(ag.reward)
        if i%100 == 0 and i>0:
            print('Average Reward in', (i-100,i), 'is', all_reward/100)
            all_reward = 0
    print('Average Reward in', (i - 100, i), 'is', all_reward / 100)
    print(ag.qtable)
    print(rewardlist)
    return ag.qtable



Q = qlearning(2000)

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



