import time

import gym
import numpy as np
import tensorflow as tf
from ppo.utils import make_env


class GymRunner:
    def __init__(self, env_id, seed, num_envs=4):
        self.run_name = f"{env_id}__{seed}__{int(time.time())}"

        # hyperparameters
        self.num_envs = 4  # the number of parallel game environments
        self.gamma = 0.99  # the discount factor gamma
        self.gae_lambda = 0.95  # the lambda for the general advantage estimation
        self.max_grad_norm = 0.5  # the maximum norm for the gradient clipping
        self.target_kl = 0.01  # the target kullback leibler divergence threshold
        self.total_timesteps = 200000  # total timesteps of the experiments
        self.rollout_length = 128  # the number of steps to run in each environment per policy rollout
        self.batch_size = int(num_envs * self.rollout_length)  # number of experiences for learning
        self.num_minibatches = 4  # the number of mini-batches per learning step
        self.minibatch_size = int(self.batch_size // self.num_minibatches)  # the size of one minibatch
        self.epochs = int(self.total_timesteps // self.batch_size)  # the number of total epochs for learning
        self.train_iters = 4  # the number of iters to update the policy

        # env state
        self.envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i) for i in range(num_envs)])
        self.global_step = 0
        self.current_obs = self.envs.reset()
        self.current_dones = np.zeros(self.num_envs)

    def obs_shape(self):
        obs_shape = self.envs.single_observation_space.shape
        return obs_shape

    def action_shape(self):
        action_shape = self.envs.single_action_space.shape
        return action_shape

    def action_size(self):
        action_size = int(self.envs.single_action_space.n)
        return action_size

    def rollout(self, agent):

        # set up storage
        obs_batch = np.zeros((self.rollout_length, self.num_envs) + self.obs_shape(), dtype=np.float32)
        actions_batch = np.zeros((self.rollout_length, self.num_envs) + self.action_shape(), dtype=np.int32)
        log_probs_batch = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        rewards_batch = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)
        dones_batch = np.zeros((self.rollout_length, self.num_envs), dtype=np.bool)
        values_batch = np.zeros((self.rollout_length, self.num_envs), dtype=np.float32)

        for step in range(self.rollout_length):
            self.global_step += 1 * self.num_envs

            actions, values, log_probs, entropy = agent.action_and_value(self.current_obs)

            # store state and agent decisions
            obs_batch[step] = self.current_obs
            actions_batch[step] = actions
            log_probs_batch[step] = log_probs
            dones_batch[step] = self.current_dones
            values_batch[step] = tf.reshape(values, [-1])

            # play games one step
            self.current_obs, rewards, self.current_dones, info = self.envs.step(actions)
            rewards_batch[step] = rewards

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                    tf.summary.scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                    tf.summary.scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
                    break

        # advantages
        actions, values, log_probs, entropy = agent.action_and_value(self.current_obs)
        advantages_batch = self.advantages(dones_batch, values_batch, rewards_batch, self.current_dones, values)
        returns_batch = advantages_batch + values_batch

        return self.flatten_batch(
            obs_batch, actions_batch, log_probs_batch, returns_batch, advantages_batch, values_batch
        )

    def advantages(self, dones_batch, values_batch, rewards_batch, dones, values):
        """bootstrap values if not done and calculate advantages."""
        advantages_batch = np.zeros_like(rewards_batch)
        lastgaelam = 0

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_non_terminal = 1.0 - dones
                next_values = values
            else:
                next_non_terminal = 1.0 - dones_batch[t + 1]
                next_values = values_batch[t + 1]
            delta = rewards_batch[t] + self.gamma * next_values * next_non_terminal - values_batch[t]
            advantages_batch[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam

        return advantages_batch

    def flatten_batch(self, obs_batch, actions_batch, log_probs_batch, returns_batch, advantages_batch, values_batch):
        obs_batch = obs_batch.reshape((-1,) + self.obs_shape())
        actions_batch = actions_batch.reshape((-1,) + self.action_shape())
        log_probs_batch = log_probs_batch.reshape(-1)
        returns_batch = returns_batch.reshape(-1)
        advantages_batch = advantages_batch.reshape(-1)
        values_batch = values_batch.reshape(-1)
        return obs_batch, actions_batch, log_probs_batch, returns_batch, advantages_batch, values_batch

    def run(self, agent):
        writer = tf.summary.create_file_writer(f"runs/{self.run_name}")
        with writer.as_default():
            for epoch in range(1, self.epochs + 1):
                obs, actions, log_probs, returns, advantages, values = self.rollout(agent)
                loss_info = self.train(agent, obs, actions, log_probs, returns, advantages, values)

                y_pred, y_true = values, returns
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                tf.summary.scalar("charts/learning_rate", agent.optimizer._decayed_lr(np.float32), self.global_step)
                tf.summary.scalar("losses/value_loss", loss_info.value_loss, self.global_step)
                tf.summary.scalar("losses/policy_loss", loss_info.policy_loss, self.global_step)
                tf.summary.scalar("losses/entropy", loss_info.entropy_loss, self.global_step)
                tf.summary.scalar("losses/approx_kl", loss_info.approx_kl, self.global_step)
                tf.summary.scalar("losses/clipfrac", np.mean(loss_info.clip_fracs), self.global_step)
                tf.summary.scalar("losses/explained_variance", explained_var, self.global_step)

        writer.flush()
        self.envs.close()

    def train(self, agent, obs, actions, log_probs, returns, advantages, values):
        inds = np.arange(self.batch_size)

        for _ in range(self.train_iters):
            np.random.shuffle(inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                i = inds[start:end]

                # take gradient and backpropagate
                with tf.GradientTape() as tape:
                    loss_info = agent.loss(obs[i], actions[i], returns[i], advantages[i], values[i], log_probs[i])
                trainable_variables = agent.model.trainable_variables
                grads = tape.gradient(loss_info.total_loss, trainable_variables)

                # clip gradients for slight updates
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                agent.optimizer.apply_gradients(zip(grads, trainable_variables))

            if loss_info.approx_kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {_} due to reaching max kl.")
                break

        return loss_info
