'''
Implementation of Q-Value Iteration for discrete MDPs as QValueIteration class using PyTorch

Author: Jay Shah (@Jayshah25)

Contact: jay.shah@qrlqai.com

License: Apache-2.0
'''

from typing import Optional, Union

import gymnasium as gym
import torch

from .._base import BaseIteration


class QValueIteration(BaseIteration):
    """
    Q-Value Iteration for tabular, model-based RL over discrete MDPs.

    Maintains a state-action value function Q(s,a) and applies the Bellman
    optimality operator until convergence:

        Q[s,a] <- Σ_s'  P(s'|s,a) · (R(s,a,s') + γ · max_a' Q(s',a'))

    Action selection reads directly from Q with no recomputation.
    V(s) = max_a Q(s,a) is available as a derived property.

    Compared to ValueIteration:
    - Q(s,a) is stored persistently rather than computed transiently.
    - Action selection is O(n_actions) per state rather than O(n_actions × n_states).
    - Q(s,a) is the natural precursor to Q-learning and function approximation
      (e.g. DQN, quantum RL agents), making this the more forward-compatible choice.

    Parameters
    ----------
    env : gym.Env
        Gymnasium or qrl-qai environment with discrete observation and action spaces.
    gamma : float
        Discount factor in [0, 1).
    num_test_episodes : int
        Informational; used by external training loops for evaluation.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, optional
        Defaults to float32.

    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.9,
        num_test_episodes: int = 20,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super().__init__(env, gamma, num_test_episodes, device, dtype)

        # Primary stored quantity: Q[s, a], shape (n_states, n_actions)
        self._Q = torch.zeros(
            (self.n_states, self.n_actions), dtype=self.dtype, device=self.device
        )

    # ── planning ──────────────────────────────────────────────────────────────

    def qvalue_iteration(
        self,
        max_iters: Optional[int] = None,
        tol: float = 1e-6,
    ) -> int:
        """
        Run Q-Value Iteration to convergence (or max_iters).

        Parameters
        ----------
        max_iters : int, optional
            Hard cap on Bellman updates. Runs until |Q_new - Q|_inf < tol if None.
        tol : float
            Convergence threshold on the sup-norm of the Q-value change.

        Returns
        -------
        int
            Number of iterations performed.
        """
        P = self._get_P()   # (n_s, n_a, n_s)
        R = self._get_R()   # (n_s, n_a, n_s)
        Q = self._Q

        for i in range(max_iters or int(1e9)):
            # V(s') = max_a' Q(s', a') — implicit value, not stored separately
            V_from_Q = Q.max(dim=1).values                           # (n_s,)

            # Q[s,a] = Σ_s' P[s,a,s'] · (R[s,a,s'] + γ · V(s'))
            V_expand = V_from_Q.unsqueeze(0).unsqueeze(0)            # (1, 1, n_s)
            Q_new = (P * (R + self.gamma * V_expand)).sum(dim=2)     # (n_s, n_a)

            diff = (Q_new - Q).abs().max().item()
            Q = Q_new

            if diff < tol:
                self._Q = Q
                return i + 1

        self._Q = Q
        return max_iters

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: Union[int, torch.Tensor]) -> int:
        """
        Greedy action: argmax_a Q(s, a). Direct table lookup — no recomputation.

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

    # ── inspection ────────────────────────────────────────────────────────────

    @property
    def Q(self) -> torch.Tensor:
        """Current state-action value function, shape (n_states, n_actions)."""
        return self._Q

    @property
    def V(self) -> torch.Tensor:
        """
        State-value function derived from Q: V(s) = max_a Q(s,a).
        Shape (n_states,). Not stored — recomputed on access.
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