# RL playground

Implementation of:
- PPO with TensorFlow 2.0 to solve OpenAI's CartPole-v1
- SAC with TensorFlow 2.0 to solve OpenAI's MountainCarContinuous-v0

The simplicity makes understanding of PPO straightforward.
All steps are represented in two files:
`agent.py` implements the actor and critic networks.
In Addition, it implements the forward pass (values/action) and the loss fuction.
`runner.py` implements the rollout of the multiple parallel environments.
In addition, it implements the advantage calculation and training of the agent on mini-batches.

In the future, I plan to extend the repo with other RL algos (e.g. A3C or continuous PPO).

## Getting Started

Prerequisites (Example with pyenv python version handling for MacOS):

* macOS
* pyenv installed

Clone the repo and install requirements with the correct pypthon version
```bash
git clone https://github.com/mesjou/rl-playground.git && cd rl-playground

pyenv install 3.8.10
pyenv virtualenv 3.8.10 rl-project
pyenv local rl-project

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run Game

Run the game and visualize in TensorBoard
```bash
python3 cartpole_ppo.py
tensorboard --logdir=runs/
```

## Authors

- [mesjou](https://github.com/mesjou)

## References

I have been heavily relying on the `cleanrl` repo:
* https://github.com/vwxyzjn/cleanrl

Additional resources for PPO:
* https://github.com/tensorflow/agents
* https://github.com/jw1401/PPO-Tensorflow-2.0
* https://github.com/lilianweng/deep-reinforcement-learning-gym

Additional resources for SAC:
* https://github.com/RickyMexx/SAC-tf2
* https://github.com/openai/spinningup
