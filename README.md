# Quantum-Reinforcement-Learning

This repo is based on the book "Deep Reinforcement Learning Hands-On" (Third Edition) by Maxim Lapan. The goal is to test quantum agents for the various popular Reinforcement Learning algorithm explained (really well) in the book. 

## Reinforcement Learning Algorithms 

###  Cross-Entropy Method

The cross-entropy method is a sort of _hello world_ of Reinforcement Learning. It is a model-free, policy based and on-policy RL method. 

* **model-free** - The agent just takes some observations at given time steps and does computation on it to decide the action rather than trying to predict next observations/rewards. 

* **policy-based** - The method tries to approximate the policy for the agent. This policy is generally a probability distribution over the agents. 

* **on-policy** - The method does not learn from any available historical data. Instead it learns from the fresh data being generated by the environment from the policy that it is trying to approximate.

#### Algorithm 

At core, the method relies on training on _good_ episodes and throwing away the bad ones. This notion of _good_ and _bad_ episode is generally based on the reward a particular episode managed to achieve as compared to the acceptable reward value for the environment in consideration. 

1. Play _N_ episodes in the enviroment. (This value sort of corresponds to the batch size as per the ML conventions).

2. Calculate the total reward for each episode and decided a reward boundary. A general choice is 50%ile, 70%ile or 75%ile of all rewards.

3. Discard all the episodes with reward less than the reward boundary. 

4. Train on the remaining _good_ episodes. 

5. Repeat until convergence is achieved. 

#### Results 

As per the book, we solve the `CartPole-v1` and `FrozenLake-v1` environments for both classical and quantum agents. 
