### Proximal Policy Optimization(PPO)

Impletementation of the PPO method proposed by OpenAI.

PPO is an trust-region policy optimization method. It used a penalty instead of a constraint in the TRPO objective.

Related papers:
- [PPO - J Schulman et al.](https://arxiv.org/abs/1707.06347)
- [TRPO - J Schulman et](https://arxiv.org/abs/1502.05477)

### Requirements
- Python 3.x
- Tensorflow 1.3.0
- gym 0.9.4

### Run
      python3 train_PPO.py  # -h to show avaliavle arguments

### Results
<img src="https://github.com/borgwang/reinforce_py/raw/master/images/ppo_score.png" width = "600" height = "440" alt="ppo-score" align=center />

Smoothed rewards in 1M steps

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/ppo_losses.png" width = "600" height = "440" alt="ppo-lossed" align=center />

Policy loss and value function loss during training.
