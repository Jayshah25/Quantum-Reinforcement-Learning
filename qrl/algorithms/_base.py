"""Shared model-based agent base for tabular RL (discrete MDPs)."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch


class BaseIteration:
    """
    Shared base class for tabular model-based RL agents (Value Iteration, QValueIteration).

    Maintains empirical estimates of the transition probability P(s'|s,a) and
    mean reward R(s,a,s') from environment interaction. Subclasses implement
    the specific Bellman update and action-selection strategy.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium or qrl-qai environment with discrete observation and action spaces.
    gamma : float
        Discount factor in [0, 1).
    num_test_episodes : int
        Number of episodes used for evaluation (informational; used by training loops).
    device : torch.device, optional
        Compute device. Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, optional
        Floating-point dtype for all tensors. Defaults to float32.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.9,
        num_test_episodes: int = 20,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.env = env
        _, *_ = self.env.reset() #reset the environment and get the initial state
        self.gamma = gamma
        self.num_test_episodes = num_test_episodes
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)

        # Visitation counts C[s, a, s']
        self._counts = torch.zeros(
            (self.n_states, self.n_actions, self.n_states),
            dtype=self.dtype,
            device=self.device,
        )
        # Accumulated rewards and visit counts for mean reward R[s, a, s']
        self._reward_sum = torch.zeros(
            (self.n_states, self.n_actions, self.n_states),
            dtype=self.dtype,
            device=self.device,
        )
        self._reward_count = torch.zeros(
            (self.n_states, self.n_actions, self.n_states),
            dtype=self.dtype,
            device=self.device,
        )

        self._state: int = 0


    def _get_P(self) -> torch.Tensor:
        """
        Empirical transition probability P[s, a, s'], shape (n_s, n_a, n_s).
        Unseen (s, a) pairs are given a uniform distribution over s'.
        """
        row_sum = self._counts.sum(dim=2, keepdim=True).clamp(min=1.0)
        return self._counts / row_sum

    def _get_R(self) -> torch.Tensor:
        """
        Mean reward tensor R[s, a, s'], shape (n_s, n_a, n_s).
        Zero for unseen transitions.
        """
        count = self._reward_count.clamp(min=1.0)
        return self._reward_sum / count

    def _record_transition(
        self, state: int, action: int, new_state: int, reward: float
    ) -> None:
        """Update empirical model with a single observed transition."""
        self._counts[state, action, new_state] += 1.0
        self._reward_sum[state, action, new_state] += reward
        self._reward_count[state, action, new_state] += 1.0

    def play_n_random_steps(self, n: int) -> None:
        """
        Collect n random environment steps to seed the transition/reward model.
        Should be called before the first planning update.
        """
        for _ in range(n):
            action = int(self.env.action_space.sample())
            step_result = self.env.step(action)
            try:
                new_state, reward, is_done, is_trunc, *_ = step_result
            except ValueError:
                # Fallback: classic gym API (obs, reward, done, info)
                new_state, reward, is_done, *_ = step_result
                is_trunc = False
            new_state = int(new_state)
            self._record_transition(self._state, action, new_state, float(reward))
            if is_done or is_trunc:
                obs, *_ = self.env.reset()
                self._state = int(obs)
            else:
                self._state = new_state

    def play_episode(self, env: gym.Env) -> float:
        """
        Run one full episode with the current policy, updating the model on-the-fly.

        Parameters
        ----------
        env : gym.Env
            A separate environment instance to avoid interfering with self.env.

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
            new_state, reward, is_done, is_trunc, *_ = env.step(action)
            new_state = int(new_state)
            self._record_transition(state, action, new_state, float(reward))
            total_reward += float(reward)
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward


    def select_action(self, state: int) -> int:  # pragma: no cover
        raise NotImplementedError