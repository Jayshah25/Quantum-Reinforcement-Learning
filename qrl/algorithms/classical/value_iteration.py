"""Value Iteration for discrete MDPs."""

'''
Implementation of Value Iteration for discrete MDPs as ValueIteration class using PyTorch

Author: Jay Shah (@Jayshah25)

Contact: jay.shah@qrlqai.com

License: Apache-2.0
'''

from typing import Optional, Union

import gymnasium as gym
import torch

from .._base import BaseIteration


class ValueIteration(BaseIteration):
    """
    Value Iteration for tabular, model-based RL over discrete MDPs.

    Maintains a state-value function V(s) and applies the Bellman optimality
    operator until convergence:

        V[s] <- max_a  Σ_s'  P(s'|s,a) · (R(s,a,s') + γ · V(s'))

    Action selection is greedy with respect to V via a one-step lookahead.
    Q(s,a) is never stored — it is computed transiently during planning and
    action selection.

    Parameters
    ----------
    env : gym.Env
        Gymnasium or qrl-qaienvironment with discrete observation and action spaces.
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

        # Primary stored quantity: V[s], shape (n_states,)
        self._V = torch.zeros(self.n_states, dtype=self.dtype, device=self.device)


    def value_iteration(
        self,
        max_iters: Optional[int] = None,
        tol: float = 1e-6,
    ) -> int:
        """
        Run Value Iteration to convergence (or max_iters).

        Parameters
        ----------
        max_iters : int, optional
            Hard cap on Bellman updates. Runs until |V_new - V|_inf < tol if None.
        tol : float
            Convergence threshold on the sup-norm of the value change.

        Returns
        -------
        int
            Number of iterations performed.
        """
        P = self._get_P()   # (n_s, n_a, n_s)
        R = self._get_R()   # (n_s, n_a, n_s)
        V = self._V

        for i in range(max_iters or int(1e9)):
            # Q[s,a] = Σ_s' P[s,a,s'] · (R[s,a,s'] + γ · V[s'])
            V_expand = V.unsqueeze(0).unsqueeze(0)               # (1, 1, n_s)
            Q = (P * (R + self.gamma * V_expand)).sum(dim=2)     # (n_s, n_a)

            V_new = Q.max(dim=1).values                          # (n_s,)
            diff = (V_new - V).abs().max().item()
            V = V_new

            if diff < tol:
                self._V = V
                return i + 1

        self._V = V
        return max_iters

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: Union[int, torch.Tensor]) -> int:
        """
        Greedy action w.r.t. V via one-step lookahead.

            a* = argmax_a  Σ_s' P(s'|s,a) · (R(s,a,s') + γ · V(s'))

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
        P = self._get_P()
        R = self._get_R()
        q = (P[state] * (R[state] + self.gamma * self._V)).sum(dim=1)  # (n_a,)
        return int(q.argmax().item())

    # ── inspection ────────────────────────────────────────────────────────────

    @property
    def V(self) -> torch.Tensor:
        """Current state-value function, shape (n_states,)."""
        return self._V

    def get_policy(self) -> torch.Tensor:
        """
        Greedy policy derived from V.

        Returns
        -------
        torch.Tensor
            Long tensor of shape (n_states,) where entry s is argmax_a Q(s,a).
        """
        P = self._get_P()
        R = self._get_R()
        V_expand = self._V.unsqueeze(0).unsqueeze(0)             # (1, 1, n_s)
        Q = (P * (R + self.gamma * V_expand)).sum(dim=2)         # (n_s, n_a)
        return Q.argmax(dim=1)