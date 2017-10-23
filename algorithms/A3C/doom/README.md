## Asynchronous Advanced Actor-Critic (A3C)   
Implementation of the A3C method proposed by Google DeepMind.  

Related papers:  
* [Asynchronous Methods for Deep Reinforcement Learning](http://diyhpl.us/~bryan/papers2/ai/machine-learning/Asynchronous%20methods%20for%20deep%20reinforcement%20learning%20-%202016.pdf)  

## ViZDoom  
[ViZDoom](http://vizdoom.cs.put.edu.pl/) is a Doom-based AI research platform for reinforcement learning from raw visual information. The agent recieve raw visual information and make actions(moving, picking up items and attacking monsters) to maximize scores.  

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/doom.png" width = "256" height = "200" alt="doom" align=center />  

In this repository, we implement A3C to slove basic ViZDoom task.   

## Requirements  
* [Numpy](http://www.numpy.org/)   
* [Tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  
* [scipy](https://www.scipy.org/)  
* [ViZDoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)  

## Run  
      python train_A3C.py
      python train_A3C.py -h   # show all optimal arguments

## Components  
 `train_A3C.py` create a master(global) network and multiple workers(local) network.    
 `worker.py` is the worker class implementation.    
 `net.py` construct Actor-Critic network.    
 `env_doom` is a warper of ViZDoom environment.  
