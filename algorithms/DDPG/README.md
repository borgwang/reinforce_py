## Deep Deterministic Policy Gradients  
Deep Deterministic Policy Gradient is base on [Deterministic Policy Gradient](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf) methods proposed by Sliver et al., 2014. DDPG is a policy-based RL Algorithm which can solve high-dimensional (even continuous) action spaces tasks.  
Related papers:  
* [Sliver te al., 2014](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf)  
* [Lillicrap, Timothy P., et al., 2015](https://arxiv.org/pdf/1509.02971.pdf)  

We use DDPG to solve a continuous control task Walker2d.   


## Walker2d
Walker2d is a continuous control task base on [Mujoco](http://www.mujoco.org/). The goal is to make a two-dimensional bipedal robot walk forward as fast as possible.  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/walker2d.png" width = "256" height = "256" alt="walker2d" align=center />   


## Requirements  
* [Numpy](http://www.numpy.org/)   
* [Tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  
* [Mujoco](https://www.roboti.us/index.html)

## Results  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/walker2d.gif" width = "300" height = "300" alt="walker2d" align=center />   
Results after training for 1.5M steps

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/ddpg.png" width = "600" height = "400" alt="ddpg" align=center />
Training rewards (smoothed)
