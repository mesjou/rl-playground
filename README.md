# rl-playground

Simple single file implementation of PPO with TensorFlow 2.0

I created the repo to understand PPO and make all of the basic functionality available in
one single file to quickly grasp the concepts.
I plan to extend the algorithms in hte future with additional other RL algos.

## Get started

Prerequisites (Example for MacOS only):

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


## References

I have been heavily relying on the `cleanrl` repo:
* https://github.com/vwxyzjn/cleanrl

Additional resources:
* https://github.com/tensorflow/agents
* https://github.com/jw1401/PPO-Tensorflow-2.0
