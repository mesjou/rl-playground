import time
from typing import Tuple

import gym

from qlearning.wrapper import ObservationDiscretizer


class GymRunner:
    def __init__(self, env_id: str, seed: int, n_states: Tuple = (50, 20)):
        self.run_name = f"{env_id}__{seed}__{int(time.time())}"

        # env storage
        self.env = gym.make(env_id)
        self.env = ObservationDiscretizer(self.env, n_states)
        self.env.seed(seed)

        # hyperparameters
        self.n_states = n_states
        self.episodes = 12000  # the number of total epochs for learning
        self.start_steps = 1000  # after how many random steps should alpha decrease
        self.alpha_schedule = [0.3, 0.01, 0.001]
        self.epsilon = 0.0
        self.gamma = 0.9

    def obs_shape(self) -> Tuple:
        return self.n_states

    def n_actions(self) -> int:
        return self.env.action_space.n

    def run(self, agent):

        # metric storage
        all_reward, episode_length, l, total_episode_length, finished_episodes = 0, 0, 0, 0, 0

        for i in range(self.episodes):  # Number of Episodes
            state = self.env.reset()
            done = False

            # decrease learning rate alpha
            if i <= self.start_steps:
                alpha = self.alpha_schedule[0]
            elif self.start_steps < i < 6 * self.start_steps:
                alpha = self.alpha_schedule[1]
            else:
                alpha = self.alpha_schedule[2]

            while not done:  # one episode
                episode_length += 1

                action = agent.act(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)
                agent.learn(action, reward, state, next_state, done, alpha, self.gamma)
                state = next_state

                # show learned policy at the end
                if i > self.episodes - 5:
                    self.env.render()

            total_episode_length += episode_length

            if episode_length < 200:
                finished_episodes += 1

            if (i % 1000 == 0 and i > 0) or i + 1 == self.episodes:
                print('In the Episodes', i - 999, 'to', i, 'a total of', finished_episodes,
                      'Episodes were finished successfully and Average Episode Length is',
                      total_episode_length / 1000)
                total_episode_length, finished_episodes = 0, 0
            episode_length = 0
        self.env.close()
        return agent.qtable
