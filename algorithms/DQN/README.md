## DQN  
DQN is a deep reinforcement learning architecture proposed by DeepMind in 2013. They used this architecture to play Atari games and obtained a hunman-level performance.  
Here we implement a simple version of DQN to solve game of CartPole.   

Related papers:  
* [Mnih et al., 2013](https://arxiv.org/pdf/1312.5602.pdf)   
* [Mnih et al., 2015](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)  

Double DQN is also implemented. Pass a --double=True argument to enable Double DQN. Since the CartPole task is easy to solve, Double DQN actually produce very little effect.  

## CartPole  
In CartPole, a pole is attached by an un-actuated joint to a cart, which move along a track. The agent contrl the cart by moving left or right in order to ballance the pole. More about CartPole see [OpenAI wiki](https://github.com/openai/gym/wiki/CartPole-v0)  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/cartpole.png" width = "256" height = "200" alt="CartPole" align=center />   

We will solve CartPole using DQN Algorithm.   

## Requirements  
* [Numpy](http://www.numpy.org/)   
* [Tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  


## Run  
    python train_DQN.py
    python train_DQN.py -h   # training options and hyper parameters settings


## Training plot  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/dqn.png" width = "512" height = "400" alt="DQN" align=center />   
