import torch
import torch.nn as nn 
from torch.optim import Adam
import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List

from .utils import Episode, EpisodeStep

class CrossEntropy(nn.Module):

    def __init__(self, env:gym.Env,  batch_size:int = 16):
        super().__init__()

        
        assert env.observation_space.shape is not None, "Observation space cannot be empty!"
        self.observation_size = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.env = env 
        self.agent = None
        self.objective = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        print(f'Observation space size: {self.observation_size}')
        print(f'Action space size: {self.n_actions}')

    def yield_batch(self):
        batch = []
        episode_reward = 0.0
        episode_steps = []
        obs, _ = self.env.reset()
        sm = nn.Softmax(dim=1) # using softmax separately and not in the agent structure for efficient computation
        while True:
            obs_v = torch.tensor(obs, dtype=torch.float32)
            act_probs_v = sm(self.agent(obs_v.unsqueeze(0)))
            act_probs = act_probs_v.data.numpy()[0]
            action = np.random.choice(len(act_probs), p=act_probs)
            next_obs, reward, is_done, is_trunc, _ = self.env.step(action)
            episode_reward += float(reward)
            step = EpisodeStep(observation=obs, action=action)
            episode_steps.append(step)
            if is_done or is_trunc:
                e = Episode(reward=episode_reward, steps=episode_steps)
                batch.append(e)
                episode_reward = 0.0
                episode_steps = []
                next_obs, _ = self.env.reset()
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            obs = next_obs


    def filter_batch(self, batch:List[Episode], reward_boundary:float):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = float(np.percentile(rewards,reward_boundary))
        reward_mean = float(np.mean(rewards))

        train_obs: List[np.ndarray] = []
        train_act: List[int] = []
        for episode in batch:
            if episode.reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, episode.steps))
            train_act.extend(map(lambda step: step.action, episode.steps))

        train_obs_v = torch.FloatTensor(np.vstack(train_obs))
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean

    def train(self, agent:Callable, optimizer:Callable, reward_boundary:float = 0.50):
        self.agent = agent
        self.optimizer = Adam(params=self.agent.parameters(), lr=0.01) if optimizer is None else optimizer
        mean_rewards = []
        for iter_no, batch in enumerate(self.yield_batch()):
            obs_v, acts_v, reward_b, reward_m = self.filter_batch(batch, reward_boundary)
            self.optimizer.zero_grad()
            action_scores_v = self.agent(obs_v)
            loss_v = self.objective(action_scores_v, acts_v)
            loss_v.backward()
            self.optimizer.step()
            mean_rewards.append(reward_m)
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            if iter_no > 50: # for cartpole
                return mean_rewards

            

    def train_classical_and_quantum(self,classical_agent,quantum_agent,reward_boundary:float = 0.50,plot:bool=False, optimizer:Callable|None = None ):
        classical_mean_rewards = self.train(agent=classical_agent,optimizer=optimizer,reward_boundary=reward_boundary)
        quantum_mean_rewards = self.train(agent=quantum_agent,optimizer=optimizer,reward_boundary=reward_boundary)
        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(classical_mean_rewards,label='Classical Rewards',alpha=0.5)
            ax.plot(quantum_mean_rewards,label='Quantum Rewards',alpha=0.5)
            ax.legend()
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Mean Reward')
            ax.set_title('Classical Agent vs Quantum Agnent')
            fig.savefig('classical_vs_quantum_agent.png')
                

