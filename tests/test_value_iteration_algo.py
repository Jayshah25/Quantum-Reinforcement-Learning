"""
Tests for BaseIteration, ValueIteration, and QValueIteration.

Uses FrozenLake-v1 (is_slippery=False) for deterministic, reproducible transitions.

Run with:
    pytest test_algorithms.py -v
"""

import pytest
import torch
import gymnasium as gym
from qrl.algorithms.classical import ValueIteration, QValueIteration
from qrl.algorithms._base import BaseIteration


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return gym.make("FrozenLake-v1", is_slippery=False)   # 16 states, 4 actions

@pytest.fixture
def test_env():
    return gym.make("FrozenLake-v1", is_slippery=False)

@pytest.fixture
def vi(env):
    return ValueIteration(env=env, gamma=0.9)

@pytest.fixture
def qvi(env):
    return QValueIteration(env=env, gamma=0.9)


# ─── BaseIteration ────────────────────────────────────────────────────────────

def test_base_initialization(vi):
    """n_states, n_actions and model tensors must have correct shapes at init."""
    assert vi.n_states  == 16
    assert vi.n_actions == 4
    assert vi._counts.shape       == (16, 4, 16), "_counts shape must be (n_s, n_a, n_s)"
    assert vi._reward_sum.shape   == (16, 4, 16), "_reward_sum shape must be (n_s, n_a, n_s)"
    assert vi._reward_count.shape == (16, 4, 16), "_reward_count shape must be (n_s, n_a, n_s)"
    assert vi._counts.sum()       == 0,           "_counts must be all-zero before any steps"


def test_base_get_P_zeros_for_unseen_state_action_pairs(vi):
    """
    _get_P() rows must be all-zero for unseen (s,a) pairs.

    Implementation: counts / clamp(counts.sum(), min=1).
    For unseen pairs counts=0, so P=0/1=0 — rows sum to 0, not 1.
    After recording a transition the affected row must sum to 1.
    """
    P_before = vi._get_P()
    assert P_before.shape == (16, 4, 16), "_get_P() must return shape (n_s, n_a, n_s)"
    # No data yet — every row should be zero
    assert P_before.sum().item() == 0.0, \
        "All rows must be zero before any transitions are recorded"

    # Record one transition and verify that row now sums to 1
    vi._record_transition(state=0, action=1, new_state=4, reward=0.0)
    P_after = vi._get_P()
    assert abs(P_after[0, 1, :].sum().item() - 1.0) < 1e-6, \
        "Row for a seen (s,a) pair must sum to 1 after recording a transition"


def test_base_record_transition_updates_counts_and_rewards(vi):
    """_record_transition() must correctly update counts and reward accumulators."""
    vi._record_transition(state=0, action=1, new_state=4, reward=0.5)
    assert vi._counts[0, 1, 4]       == 1.0, "_counts should be incremented by 1"
    assert vi._reward_sum[0, 1, 4]   == 0.5, "_reward_sum should accumulate the reward"
    assert vi._reward_count[0, 1, 4] == 1.0, "_reward_count should be incremented by 1"
    assert abs(vi._get_R()[0, 1, 4].item() - 0.5) < 1e-6, \
        "_get_R() must return the mean reward for the recorded transition"


def test_base_play_n_random_steps_populates_model(vi):
    """Total transition count after play_n_random_steps(n) must equal n."""
    n = 50
    vi.play_n_random_steps(n)
    assert vi._counts.sum().item() == n, \
        f"Expected {n} total recorded transitions, got {vi._counts.sum().item()}"


def test_base_select_action_raises_not_implemented(env):
    """Calling select_action() on BaseIteration directly must raise NotImplementedError."""
    base = BaseIteration(env=env, gamma=0.9)
    with pytest.raises(NotImplementedError):
        base.select_action(0)


def test_base_play_episode_returns_float(vi, test_env):
    """play_episode() must return a float total reward."""
    vi.play_n_random_steps(100)
    vi.value_iteration()
    result = vi.play_episode(test_env)
    assert isinstance(result, float), \
        f"play_episode() must return float, got {type(result)}"


# ─── ValueIteration ──────────────────────────────────────────────────────────

def test_vi_initialization(vi):
    """_V must be a zero tensor of shape (n_states,) on construction."""
    assert vi._V.shape == (16,), "_V must have shape (n_states,)"
    assert vi._V.sum() == 0.0,   "_V must be all-zero before planning"


def test_vi_value_iteration_returns_positive_int(vi):
    """value_iteration() must return a positive int (number of Bellman sweeps performed)."""
    vi.play_n_random_steps(100)
    n_iters = vi.value_iteration()
    assert isinstance(n_iters, int) and n_iters >= 1, \
        "value_iteration() must return a positive int (number of Bellman sweeps)"


def test_vi_value_iteration_updates_V_when_reward_seen(vi):
    """_V must become non-zero after value_iteration() once a reward has been recorded."""
    # Manually inject a known reward so the test is not dependent on random exploration
    vi._record_transition(state=14, action=2, new_state=15, reward=1.0)
    vi.value_iteration()
    assert vi._V.sum().item() != 0.0, \
        "_V must be non-zero after value_iteration() when at least one reward is recorded"


def test_vi_select_action_returns_valid_action(vi):
    """select_action() must return an int in [0, n_actions) for every state."""
    vi.play_n_random_steps(100)
    vi.value_iteration()
    for state in range(vi.n_states):
        action = vi.select_action(state)
        assert isinstance(action, int), "select_action() must return int"
        assert 0 <= action < vi.n_actions, \
            f"Action {action} out of range [0, {vi.n_actions}) for state {state}"


def test_vi_get_policy_shape_dtype_and_range(vi):
    """get_policy() must return a long tensor of shape (n_states,) with valid actions."""
    vi.play_n_random_steps(100)
    vi.value_iteration()
    policy = vi.get_policy()
    assert policy.shape == (16,),          "Policy must have shape (n_states,)"
    assert policy.dtype == torch.long,     "Policy must be a long (int64) tensor"
    assert int(policy.min()) >= 0,         "All actions must be >= 0"
    assert int(policy.max()) < vi.n_actions, "All actions must be < n_actions"


@pytest.mark.parametrize("target_reward", [0.8])
def test_vi_solves_frozen_lake(target_reward, test_env):
    """ValueIteration must achieve > 80% success on deterministic FrozenLake."""
    env   = gym.make("FrozenLake-v1", is_slippery=False)
    agent = ValueIteration(env=env, gamma=0.9)
    best  = 0.0
    for _ in range(30):
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = sum(agent.play_episode(test_env) for _ in range(20)) / 20
        best   = max(best, reward)
    assert best >= target_reward, \
        f"ValueIteration should reach {target_reward:.0%} on deterministic FrozenLake, got {best:.2f}"


# ─── QValueIteration ─────────────────────────────────────────────────────────

def test_qvi_initialization(qvi):
    """_Q must be a zero tensor of shape (n_states, n_actions) on construction."""
    assert qvi._Q.shape == (16, 4), "_Q must have shape (n_states, n_actions)"
    assert qvi._Q.sum() == 0.0,     "_Q must be all-zero before planning"


def test_qvi_qvalue_iteration_returns_positive_int(qvi):
    """qvalue_iteration() must return a positive int (number of Bellman sweeps performed)."""
    qvi.play_n_random_steps(100)
    n_iters = qvi.qvalue_iteration()
    assert isinstance(n_iters, int) and n_iters >= 1, \
        "qvalue_iteration() must return a positive int (number of Bellman sweeps)"


def test_qvi_qvalue_iteration_updates_Q_when_reward_seen(qvi):
    """_Q must become non-zero after qvalue_iteration() once a reward has been recorded."""
    # Manually inject a known reward so the test is not dependent on random exploration
    qvi._record_transition(state=14, action=2, new_state=15, reward=1.0)
    qvi.qvalue_iteration()
    assert qvi._Q.sum().item() != 0.0, \
        "_Q must be non-zero after qvalue_iteration() when at least one reward is recorded"


def test_qvi_V_property_equals_max_Q(qvi):
    """V property must equal max_a Q(s,a) for every state."""
    qvi.play_n_random_steps(100)
    qvi.qvalue_iteration()
    assert torch.allclose(qvi.V, qvi._Q.max(dim=1).values), \
        "V(s) must equal max_a Q(s,a) for all states"


def test_qvi_select_action_returns_valid_action(qvi):
    """select_action() must return an int in [0, n_actions) for every state."""
    qvi.play_n_random_steps(100)
    qvi.qvalue_iteration()
    for state in range(qvi.n_states):
        action = qvi.select_action(state)
        assert isinstance(action, int), "select_action() must return int"
        assert 0 <= action < qvi.n_actions, \
            f"Action {action} out of range [0, {qvi.n_actions}) for state {state}"


def test_qvi_get_policy_shape_dtype_and_range(qvi):
    """get_policy() must return a long tensor of shape (n_states,) with valid actions."""
    qvi.play_n_random_steps(100)
    qvi.qvalue_iteration()
    policy = qvi.get_policy()
    assert policy.shape == (16,),           "Policy must have shape (n_states,)"
    assert policy.dtype == torch.long,      "Policy must be a long (int64) tensor"
    assert int(policy.min()) >= 0,          "All actions must be >= 0"
    assert int(policy.max()) < qvi.n_actions, "All actions must be < n_actions"


@pytest.mark.parametrize("target_reward", [0.8])
def test_qvi_solves_frozen_lake(target_reward, test_env):
    """QValueIteration must achieve > 80% success on deterministic FrozenLake."""
    env   = gym.make("FrozenLake-v1", is_slippery=False)
    agent = QValueIteration(env=env, gamma=0.9)
    best  = 0.0
    for _ in range(30):
        agent.play_n_random_steps(100)
        agent.qvalue_iteration()
        reward = sum(agent.play_episode(test_env) for _ in range(20)) / 20
        best   = max(best, reward)
    assert best >= target_reward, \
        f"QValueIteration should reach {target_reward:.0%} on deterministic FrozenLake, got {best:.2f}"