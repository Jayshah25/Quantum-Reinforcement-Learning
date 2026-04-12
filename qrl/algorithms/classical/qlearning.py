'''
Implementation of Q-Learning for discrete MDPs as QLearning class using PyTorch.

Q-learning is model-free: it learns Q(s,a) directly from environment interaction
without building an explicit transition/reward model.

Bellman update (off-policy TD(0)):

    Q[s,a] <- Q[s,a] + α · (r + γ · max_a' Q[s',a'] - Q[s,a])

Author: Jay Shah (@Jayshah25)

Contact: jay.shah@qrlqai.com

License: Apache-2.0
'''

from typing import Optional, Tuple, Union

import gymnasium as gym
import torch


class QLearning:
    """
    Q-Learning for tabular, model-free RL over discrete MDPs.

    Maintains a state-action value function Q(s,a) updated via off-policy
    TD(0) after each environment step:

        Q[s,a] <- Q[s,a] + α · (r + γ · max_a' Q[s',a'] - Q[s,a])

    Unlike ValueIteration / QValueIteration, no transition model is built.
    Q is updated incrementally from single (s, a, r, s') transitions.

    Parameters
    ----------
    env : gym.Env
        Gymnasium or qrl-qai environment with discrete observation and action spaces.
    gamma : float
        Discount factor in [0, 1).
    alpha : float
        Learning rate (step size) in (0, 1].
    num_test_episodes : int
        Number of episodes used for evaluation in training loops.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, optional
        Defaults to float32.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.9,
        alpha: float = 0.2,
        num_test_episodes: int = 20,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.num_test_episodes = num_test_episodes
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)

        # Q[s, a], shape (n_states, n_actions), initialised to zero
        self._Q = torch.zeros(
            (self.n_states, self.n_actions), dtype=self.dtype, device=self.device
        )

        obs, *_ = self.env.reset()
        self._state: int = int(obs)


    def sample_env(self) -> Tuple[int, int, float, int]:
        """
        Take one random step in the environment.

        Returns
        -------
        Tuple[int, int, float, int]
            (state, action, reward, next_state)
        """
        action = int(self.env.action_space.sample())
        old_state = self._state
        new_state, reward, is_done, is_trunc, *_ = self.env.step(action)
        new_state = int(new_state)
        if is_done or is_trunc:
            obs, *_ = self.env.reset()
            self._state = int(obs)
        else:
            self._state = new_state
        return old_state, action, float(reward), new_state

    def value_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """
        Apply one Q-learning (off-policy TD(0)) update.

            Q[s,a] <- Q[s,a] + α · (r + γ · max_a' Q[s',a'] - Q[s,a])

        Parameters
        ----------
        state : int
            Current state index.
        action : int
            Action taken.
        reward : float
            Observed reward.
        next_state : int
            Resulting next-state index.
        """
        best_next = self._Q[next_state].max()          # max_a' Q[s', a']
        td_target = reward + self.gamma * best_next
        td_error = td_target - self._Q[state, action]
        self._Q[state, action] += self.alpha * td_error


    def select_action(self, state: Union[int, torch.Tensor]) -> int:
        """
        Greedy action: argmax_a Q(s, a).

        Parameters
        ----------
        state : int or 0-d Tensor
            Current state index.

        Returns
        -------
        int
            Greedy action.
        """
        if isinstance(state, torch.Tensor):
            state = int(state.item())
        return int(self._Q[state].argmax().item())

    def play_episode(self, env: gym.Env) -> float:
        """
        Run one full greedy episode (no learning, no exploration).

        Parameters
        ----------
        env : gym.Env
            A separate environment instance so self.env is not disturbed.

        Returns
        -------
        float
            Total undiscounted reward accumulated over the episode.
        """
        total_reward = 0.0
        obs, *_ = env.reset()
        state = int(obs)
        while True:
            action = self.select_action(state)
            try:
                new_state, reward, is_done, is_trunc, *_ = step_result
            except ValueError:
                # Fallback: classic gym API (obs, reward, done, info)
                new_state, reward, is_done, *_ = step_result
                is_trunc = False

            # new_state, reward, is_done, is_trunc, *_ = env.step(action)
            total_reward += float(reward)
            if is_done or is_trunc:
                break
            state = int(new_state)
        return total_reward

    @property
    def Q(self) -> torch.Tensor:
        """Current state-action value function, shape (n_states, n_actions)."""
        return self._Q

    @property
    def V(self) -> torch.Tensor:
        """
        State-value function derived from Q: V(s) = max_a Q(s,a).
        Shape (n_states,). Recomputed on access.
        """
        return self._Q.max(dim=1).values

    def get_policy(self) -> torch.Tensor:
        """
        Greedy policy: pi[s] = argmax_a Q(s,a).

        Returns
        -------
        torch.Tensor
            Long tensor of shape (n_states,).
        """
        return self._Q.argmax(dim=1)
