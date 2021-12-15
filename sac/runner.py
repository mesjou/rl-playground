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

        # hyperparameters
        self.epochs = 100  # the number of total epochs for learning
        self.start_steps = 10000
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
                        agent.learn(obs, action, rew, next_obs, done)
                        if check == 1:
                            print("The buffer is ready, training is starting after {} steps".format(global_step))
                            check = 0
                    else:
                        agent.replay_buffer.store(obs, action, rew, next_obs, done)

                    obs = next_obs

                    epoch_reward += rew
                    epoch_lenght += 1
                    global_step += 1

                    if done:
                        print(epoch, " Epsiode reward: ", np.sum(epoch_reward), ", time steps: ", epoch_lenght)
                        break

        writer.flush()

        # print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
        # tf.summary.scalar("charts/episodic_return", item["episode"]["r"], global_step)
        # tf.summary.scalar("charts/episodic_length", item["episode"]["l"], global_step)
        # tf.summary.scalar("charts/learning_rate", agent.optimizer._decayed_lr(np.float32), global_step)
        # tf.summary.scalar("losses/value_loss", value_loss, global_step)
        # tf.summary.scalar("losses/policy_loss", loss_info.policy_loss, global_step)
        # tf.summary.scalar("losses/entropy", loss_info.entropy_loss, global_step)
        # tf.summary.scalar("losses/clipfrac", np.mean(loss_info.clip_fracs), self.global_step)
        # tf.summary.scalar("losses/explained_variance", explained_var, self.global_step)
