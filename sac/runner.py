import time

import gym
import numpy as np
import tensorflow as tf
from gym import wrappers


class GymRunner:
    def __init__(self, env_id, seed):
        self.run_name = f"{env_id}__{seed}__{int(time.time())}"

        # storage
        self.env = gym.make(env_id)
        self.env = wrappers.RecordVideo(self.env, "./videos/" + str(time.time()) + "/")
        self.env.seed(seed)

        # hyperparameters
        self.epochs = 100  # the number of total epochs for learning
        self.start_steps = 10000  # after how many random steps should training start
        self.n_timesteps = self.env.env._max_episode_steps
        self.prob_act = 0.5  # for random exploration
        self.cons_acts = 4  # for exploration

    def obs_shape(self) -> int:
        obs_shape = self.env.observation_space.shape
        return int(obs_shape[0])

    def action_shape(self) -> int:
        action_shape = self.env.action_space.shape
        return int(action_shape[0])

    def run(self, agent):
        writer = tf.summary.create_file_writer(f"runs/{self.run_name}")
        global_step = 0
        check = 1

        with writer.as_default():
            for epoch in range(1, self.epochs + 1):
                epoch_reward = 0
                epoch_lenght = 0

                obs = self.env.reset()
                for t in range(self.n_timesteps):

                    action = agent.act(obs)

                    # Using the consecutive steps technique
                    if check == 1 and np.random.uniform() < self.prob_act:
                        for i in range(self.cons_acts):
                            self.env.step(action)

                    # todo should done be modified if it is not an real end?
                    next_obs, rew, done, info = self.env.step(action)

                    # train after initialization phase
                    if global_step > self.start_steps:
                        # todo should we learn more than one step?
                        log = agent.learn(obs, action, rew, next_obs, done)

                        # logging
                        tf.summary.scalar("values/Q1", log.q1, global_step)
                        tf.summary.scalar("values/Q2", log.q2, global_step)
                        tf.summary.scalar("values/Q1_var", log.q1_var, global_step)
                        tf.summary.scalar("values/Q2_var", log.q2_var, global_step)
                        tf.summary.scalar("values/LogP", log.logp, global_step)
                        tf.summary.scalar("losses/LossPi", log.loss_actor, global_step)
                        tf.summary.scalar("losses/LossQ1", log.loss_q1, global_step)
                        tf.summary.scalar("losses/LossQ2", log.loss_q2, global_step)

                        if check == 1:
                            print("The buffer is ready, training is starting after {} steps".format(global_step))
                            check = 0
                    else:
                        agent.replay_buffer.store(obs, action, rew, next_obs, done)

                    obs = next_obs

                    # logging
                    epoch_reward += rew
                    epoch_lenght += 1
                    global_step += 1

                    if done:
                        print(f"epoch={epoch}, episodic_return={epoch_reward}, epoch lenghts={epoch_lenght}")
                        tf.summary.scalar("charts/episodic_return", epoch_reward, global_step)
                        tf.summary.scalar("charts/episodic_length", epoch_lenght, global_step)
                        break

        writer.flush()
