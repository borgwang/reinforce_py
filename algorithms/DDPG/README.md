### Deep Deterministic Policy Gradients (DDPG)

Deep Deterministic Policy Gradient is a model-free off-policy actor-critic algorithm which combines DPG ([Deterministic Policy Gradient](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf)) and DQN.

Related papers:  
- [Lillicrap, Timothy P., et al., 2015](https://arxiv.org/pdf/1509.02971.pdf)  

#### Walker2d

Here we use DDPG to solve a continuous control task Walker2d. Walker2d is a continuous control task based on [Mujoco](http://www.mujoco.org/) engine. The goal is to make a two-dimensional bipedal robot walk forward as fast as possible.  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/walker2d.png" width = "256" height = "256" alt="walker2d" align=center />   

#### Requirements  

* [Numpy](http://www.numpy.org/)   
* [Tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  
* [Mujoco](https://www.roboti.us/index.html)

#### Run  

```bash
python train_ddpg.py     
python train_ddpg.py -h   # training options and hyper parameters settings
```

#### Results  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/walker2d.gif" width = "300" height = "300" alt="walker2d" align=center />   
Results after training for 1.5M steps

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/ddpg.png" width = "600" height = "400" alt="ddpg" align=center />
Training rewards (smoothed)
