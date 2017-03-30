# Reinforcement learning in Python  

Implementations of popular RL algorithms using Python.  
* [Tabular TD-Learning](https://github.com/borgwang/reinforce_py/tree/master/Temporal-Difference)  
* [DQN](https://github.com/borgwang/reinforce_py/tree/master/DQN)  
* [REINFORCE](https://github.com/borgwang/reinforce_py/tree/master/REINFORCE)  
* [Actor-Critic](https://github.com/borgwang/reinforce_py/tree/master/Actor-Critic)  
* [DDPG](https://github.com/borgwang/reinforce_py/tree/master/DDPG)  


## Requirement  
* Python 2.7  
* [numpy](http://www.numpy.org/)   
* [tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  

## Environment setting  
1. GridWolrd  
A **GridWolrd** is a typical model for tabular reinforcement learning methods. It has a 10x10 state space and an action space of {up, down, left right} to move the agent around. There is a target point with +1 reward in the environment and some bomb point with -1 rewards. Also we set a bit cost(-0.01) when the agent take one step so that it tends to find out the optimal path faster.   
Note that GridWolrd is a definitized model (Taking an action A in state S, you can get the definitized distribution of the next state S'). More environments(including high-dimensional one) will be release later.  
A typical GridWolrd may look like this.   
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/gridworld.png)  

2. Pong  
The game of Pong is an Atari game which user control one of the paddle (the other one is control by a decent AI) and you have to bounce the ball past the other side. In reinforcement learning setting, the state is raw pixels and the action is moving the paddle UP or DOWN.  
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/pong.png)  
We will solve the game of Pong using REINFORCE and Actor-Critic Algorithms.  

3. CartPole  
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/cartpole.png)  
In CartPole, a pole is attached by an un-actuated joint to a cart, which move along a track. The agent contrl the cart by moving left or right in order to ballance the pole.  
We will solve CartPole using DQN Algorithm.  

4. Walker2d  
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/walker2d.png)  
Walker2d is a continuous control task base on Mujoco. The goal is to make a two-dimensional bipedal robot walk forward as fast as possible.
We solve the Walker2d using DDPG Algorithm.

## Algorithms  
1. Temporal Difference  
Temporal Difference (TD) learning is a prediction-based reinforcement learning alogorithm. It is a combination of Monte Carlo (MC) ideas and Dynamic Programming (DP) ideas.   
For more details about TD algorithm please refer to Chap 6 of [Reinforcement Learning: An Introduction 2nd Edition](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)  
In GridWolrd environment, the TD agent try to figure out the optimal (shortest) path to the target.   

2. DQN  
DQN is a deep reinforcement learning architecture proposed by DeepMind in 2013. They used this architecture to play Atari games and obtained a hunman-level performance.  
Here we implement a simple version of DQN to solve game of CartPole.  
Related paper:
    * [Mnih et al., 2013](https://arxiv.org/pdf/1312.5602.pdf)   
    * [Mnih et al., 2015](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)

3. REINFORCE & Actor-Critic  
Both of these alogorithm belong to Policy Gradient methods, which directly parameterize the policy rather than a state value function.  
For more details about policy gradient algorithms, see Chap 13 of  [Reinforcement Learning: An Introduction 2nd Edition](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)  
We use REINFORCE and Actor-Critic Methods to solve game of Pong.  

4. DDPG
Deep Deterministic Policy Gradient is base on Deterministic Policy Gradient methods proposed by Sliver et al., 2014. DDPG is a policy-based RL Algorithm which can solve high-dimensional (even continuous) action spaces tasks.  
We use DDPG to solve a continuous control task Walker2d.   

## LICENCE  
MIT
