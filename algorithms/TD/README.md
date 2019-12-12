### Temporal Difference  

Temporal Difference (TD) learning is a prediction-based reinforcement learning algorithm. It is the combination of Monte Carlo (MC) and Dynamic Programming (DP). 

For more details about TD algorithm please refer to Chap 6 of [Reinforcement Learning: An Introduction 2nd Edition](http://incompleteideas.net/sutton/book/the-book-2nd.html)  

### GridWolrd  

A GridWolrd is a typical environment for tabular reinforcement learning. It has a 10x10 state space and an action space of {up, down, left right} to move the agent around. There is a target point with +1 reward in the environment and some bomb point with -1 rewards. Also we set a bit cost(-0.01) when the agent take one step so that it tends to find out the optimal path faster. 

A typical GridWolrd may look like this. 

<img src="https://github.com/borgwang/reinforce_py/raw/master/images/gridworld.png" width = "256" height = "200" alt="grid" align=center />  

In GridWolrd environment, the TD agent try to figure out the optimal (i.e. the shortest) path to the target.   


### Run  

```
cd reinforce_py/algorithms/TD
python train_TD.py --algorithm=qlearn/sarsa    # Q-learning or SARSA
``` 
