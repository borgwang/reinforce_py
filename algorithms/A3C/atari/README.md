## Asynchronous Advanced Actor-Critic (A3C)   
Implementation of the A3C method proposed by Google DeepMind.  

Related papers:  
* [Asynchronous Methods for Deep Reinforcement Learning](http://diyhpl.us/~bryan/papers2/ai/machine-learning/Asynchronous%20methods%20for%20deep%20reinforcement%20learning%20-%202016.pdf)  


## Requirements  
* [Numpy](http://www.numpy.org/)   
* [Tensorflow](http://www.tensorflow.org)  
* [gym](https://gym.openai.com)  

## Run  
      python train_A3C.py
      python train_A3C.py -h   # show all optimal arguments

## Components  
 `train_A3C.py` create a master(global) network and multiple workers(local) network.    
 `worker.py` is the worker class implementation.    
 `net.py` construct Actor-Critic network.    
 `stari_env` is a warper of gym environment.  

## Note  
Still buggy currently. WIP.
