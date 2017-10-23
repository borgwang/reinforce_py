## Cross-Entropy Method

Cross-entropy method is a derivative-free policy optimize approach. It simply sample some policies, pick some good ones(elite policies) and move current policy towards these elite policies, ignoring all other information other than **rewards** collected during episode.
CEM works quiet well in tasks with simply policies(small parameter space).

## Requirements
* [Numpy](http://www.numpy.org/)   
* [gym](https://gym.openai.com)  

## Test environments  
* Discrete: CartPole-v0,  Acrobot-v1, MountainCar-v0  
* Continuous:  Pendulum-v0, BipedalWalker-v2  


## Reference  
[John Schulman MLSS 2016](http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html)  
