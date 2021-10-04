import argparse
import os
import random
import time
from distutils.util import strtobool
import collections
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


PPOLossInfo = collections.namedtuple('PPOLossInfo', (
    'total_loss',
    'v_loss',
    'pg_loss',
    'entropy_loss',
    'approx_kl',
    'clipfracs',
))


ActionValues = collections.namedtuple('ActionValues', (
    'actions',
    'values',
    'logits',
    'entropy',
))


class MyLRSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate, num_updates):
        super(LearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.num_updates = num_updates

    def __call__(self, step):
        frac = 1.0 - (step - 1.0) / self.num_updates
        return frac * self.initial_learning_rate


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=200000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, # todo cahnge default to true
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class PPOAgent(tf.keras.Model):
    def __init__(self, envs):
        super().__init__()

        self.act_size = envs.single_action_space.n

        # critic_network
        inputs = tf.keras.layers.Input(shape=(int(np.product(envs.single_observation_space.shape)),), name="observations")
        first_layer = tf.keras.layers.Dense(64, name="1st_critic_layer", activation='tanh')(inputs)
        second_layer = tf.keras.layers.Dense(64, name="2nd_critic_layer", activation='tanh')(first_layer)
        value_out = tf.keras.layers.Dense(1, name="critic_value")(second_layer)

        # actor network
        first_layer = tf.keras.layers.Dense(64, name="1st_actor_layer", activation='tanh')(inputs)
        second_layer = tf.keras.layers.Dense(64, name="2nd_actor_layer", activation='tanh')(first_layer)
        logits_out = tf.keras.layers.Dense(self.act_size, activation=None, name="action_logits")(second_layer)
        # todo add normalization to output layers

        self.base_model = tf.keras.Model(inputs, [logits_out, value_out])

    def call(self, inputs):
        logits, values = self.base_model(inputs)
        return logits, values

    def logp(self, logits, action):
        """Get the log-probability based on the action drawn from prob-distribution"""
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(action, depth=self.act_size)
        logp = tf.reduce_sum(one_hot * logp_all, axis=-1)
        return logp

    def entropy(self, logits=None):
        """Entropy term for more exploration based on OpenAI Baseline openai/baselines/common/distributions.py"""
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis=-1, keepdims=True)
        p0 = exp_a0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def get_action_and_value(self, obs, actions=None):
        logits, values = self.base_model(obs)
        if actions is None:
            actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        return ActionValues(
            actions=np.squeeze(actions),
            values=tf.squeeze(values),
            logits=tf.squeeze(self.logp(logits, actions)),
            entropy=self.entropy(logits),
        )

    def get_loss(self, obs, actions, returns, advantages, values, old_log_probs, clip_coef, norm_adv, ent_coef, vf_coef):
        # get new prediciton of logits, entropy and value
        new_action_values = self.get_action_and_value(obs, actions)

        logratio = new_action_values.logits - old_log_probs
        ratio = tf.exp(logratio)

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        approx_kl = tf.reduce_mean((ratio - 1) - logratio)
        clipfracs = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_coef), tf.int32))
        # get advantages
        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        min_adv = tf.where(advantages > 0, (1 + clip_coef) * advantages,
                           (1 - clip_coef) * advantages)
        pg_loss = -tf.reduce_mean(tf.math.minimum(ratio * advantages, min_adv))

        # Value loss
        if args.clip_vloss:
            v_loss_unclipped = tf.square(new_action_values.values - returns)
            v_clipped = values + tf.clip_by_value(
                new_action_values.values - values,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss_max = tf.math.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean(v_loss_max)
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(new_action_values.values - returns))

        entropy_loss = -tf.reduce_mean(new_action_values.entropy)

        total_loss = pg_loss + ent_coef * entropy_loss + v_loss * vf_coef

        return PPOLossInfo(
            total_loss=total_loss,
            v_loss=v_loss,
            pg_loss=pg_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clipfracs=clipfracs,
        )


if __name__ == "__main__":

    args = parse_args()

    # tensorboard
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = tf.summary.create_file_writer(f"runs/{run_name}")

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = PPOAgent(envs)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    done = np.zeros(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size

    # anneal learning rate if instructed to do so
    if args.anneal_lr:
        learning_rate = MyLRSchedule(args.learning_rate, num_updates)
    else:
        learning_rate = args.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

    with writer.as_default():
        for update in range(1, num_updates + 1):

            # ALGO Logic: Storage setup
            obs = []
            actions = []
            logprobs = []
            rewards = []
            dones = []
            values = []

            # todo init as numpy arrays then we do not need to change data types, or directly in tensorflow?

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs.append(next_obs)
                dones.append(done)

                # ALGO LOGIC: action logic
                action_value = agent.get_action_and_value(next_obs)
                values.append(tf.reshape(action_value.values, [-1]))
                actions.append(action_value.actions)
                logprobs.append(action_value.logits)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action_value.actions)
                rewards.append(reward)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        tf.summary.scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        tf.summary.scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
                        # todo does it end the episode for all three environments? this needs improvement

            # convert data to numpy array
            obs = np.array(obs, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            logprobs = np.array(logprobs, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.bool)
            values = np.array(values, dtype=np.float32)

            # bootstrap value if not done
            next_action_values = agent.get_action_and_value(next_obs)
            advantages = np.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_action_values.values
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizaing the policy and value network
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # take gradient and backpropagate
                    with tf.GradientTape() as tape:
                        loss_info = agent.get_loss(
                            b_obs[mb_inds],
                            b_actions[mb_inds],
                            b_returns[mb_inds],
                            b_advantages[mb_inds],
                            b_values[mb_inds],
                            b_logprobs[mb_inds],
                            args.clip_coef,
                            args.norm_adv,
                            args.ent_coef,
                            args.vf_coef,
                        )

                    trainable_variables = agent.base_model.trainable_variables  # take all trainable variables into account
                    grads = tape.gradient(loss_info.total_loss, trainable_variables)
                    grads, grad_norm = tf.clip_by_global_norm(grads, args.max_grad_norm)  # clip gradients for slight updates

                    optimizer.apply_gradients(zip(grads, trainable_variables))

                if args.target_kl is not None:
                    if loss_info.approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values, b_returns
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            tf.summary.scalar("charts/learning_rate", optimizer._decayed_lr(np.float32), global_step)
            tf.summary.scalar("losses/value_loss", loss_info.v_loss, global_step)
            tf.summary.scalar("losses/policy_loss", loss_info.pg_loss, global_step)
            tf.summary.scalar("losses/entropy", loss_info.entropy_loss, global_step)
            tf.summary.scalar("losses/approx_kl", loss_info.approx_kl, global_step)
            tf.summary.scalar("losses/clipfrac", np.mean(loss_info.clipfracs), global_step)
            tf.summary.scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            tf.summary.scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.flush()
        envs.close()
