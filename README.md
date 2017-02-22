# Reinforcement learning in Python  

Implementations of some RL algorithms in [Reinforcement Learning: An Introduction](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html) using Python.  


## Requirement  
* Python 2.7 or Python 3.3+  
* [numpy](http://www.numpy.org/)   

## Environment setting  
1. GridWolrd  
A **GridWolrd** is a typical model for tabular reinforcement learning methods. It has a 10x10 state space and an action space of {up, down, left right} to move the agent around. There is a target point with +1 reward in the environment and some bomb point with -1 rewards. Also we set a bit cost(-0.01) when the agent take one step so that it tends to find out the optimal path faster. Some walls were set to increase the difficulty.   
Note that GridWolrd is a definitized model (Taking an action A in state S, you can get the definitized distribution of the next state S'). More environments(including high-dimensional one) will be release later.  

A typical GridWolrd may look like this.   
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/grid.png)  
(DIY your own GridWorld by modify the code in **Envs.py**)  

2. Pong  
The game of Pong is an Atari game which user control one of the paddle (the other one is control by a decent AI) and you have to bounce the ball past the other side. In reinforcement learning setting, the state is raw pixels and the action is moving the paddle UP or DOWN.  
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/pong.png)  
We will solve the game of Pong using REINFORCE, Actor-Critic and DQN respectively.  

## Usage  
1. Value-based Algorithms (DP, MC, TD)  
![image](https://github.com/borgwang/reinforce_py/raw/master/imgs/usage.png)   
In GridWolrd environment, the agent try to figure out the optimal (shortest) path to the target.   
Running **main.py** will generate an TD(Q-learning) agent by default.  

2. REINFORCE  
Run **./REINFORCE/train_REINFORCE.py** to train an agent of playing Pong.  
Run **./REINFORCE/evaluation.py** to test a trained agent.  

3. Actor-Critic  
Run **./Actor-Critic/train_actor-critic.py** to train an agent of playing Pong.  
Run **./Actor-Critic/evaluation.py** to evaluate a trained model.  

For more details about policy gradient algorithms, see Chap 13 of  [Reinforcement Learning: An Introduction 2nd Edition](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)  

## Algorithms  
1. Dynamic Programming (GridWorld)  
2. Monte-Carlo (GridWorld)  
3. Temporal Difference (GridWorld)
4. REINFORCE (Pong)
5. Actor-Critic (Pong)  


## LICENCE  
MIT.
