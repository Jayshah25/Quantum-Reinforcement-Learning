import numpy as np
import os
import pennylane as qml
import pytest
from qrl.env.core.probability import ProbabilityV0


@pytest.fixture
def simple_target():
    return np.array([0.5, 0.5])


@pytest.fixture
def env(simple_target):
    return ProbabilityV0(n_qubits=1, target_distribution=simple_target, max_steps=5)


def test_env_initialization(env, simple_target):
    assert env.n_qubits == 1
    assert np.allclose(env.target_distribution, simple_target)
    assert env.action_space.shape == (env.n_params,)
    assert env.observation_space.shape == (2**env.n_qubits,)


def test_target_distribution_sum():
    with pytest.raises(AssertionError):
        ProbabilityV0(n_qubits=1, target_distribution=np.array([0.3, 0.3]))


def test_reset_returns_params(env):
    params, info = env.reset()
    assert isinstance(params, np.ndarray)
    assert isinstance(info, dict)
    assert len(params) == env.n_params
    assert env.current_step == 0
    assert env.history == []
    assert env.rewards == []


def test_step_changes_state(env):
    env.reset()
    action = np.zeros(env.n_params)
    obs, reward, done, info = env.step(action)
    assert obs.shape == (2**env.n_qubits,)
    assert isinstance(reward, float) or np.isscalar(reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert len(env.history) == 1
    assert len(env.rewards) == 1


def test_step_increments_step(env):
    env.reset()
    env.step(np.zeros(env.n_params))
    assert env.current_step == 1


def test_done_on_max_steps(env):
    env.reset()
    done = False
    for _ in range(env.max_steps):
        _, _, done, _ = env.step(np.zeros(env.n_params))
    assert done is True


def test_cost_fn_is_nonnegative(env):
    params = np.random.uniform(0, 2*np.pi, size=env.n_params)
    cost = env.cost_fn(params)
    assert isinstance(cost, float)
    # Since cost_fn returns -reward, it should be >= 0 in most cases
    assert cost >= 0


def test_history_and_rewards_grow(env):
    env.reset()
    for i in range(3):
        env.step(np.zeros(env.n_params))
    assert len(env.history) == 3
    assert len(env.rewards) == 3


def test_sample_run_and_render():
    file_path = r"results/tests/probability.mp4"
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path)   
 
    n_qubits = 2
    target_distribution = np.array([0.25, 0.25, 0.25, 0.25])  # Example target distribution

    # initialize environment
    env = ProbabilityV0(
        n_qubits=n_qubits,
        target_distribution=target_distribution,
        alpha=0.7,   # KL vs L2 weight
        beta=0.01,   # step penalty
        max_steps=100
    )

    # Reset environment
    params, _ = env.reset()


    # Use PennyLane's optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    # params = env.params.copy()
    for step in range(env.max_steps):
        params, cost_val = opt.step_and_cost(env.cost_fn, params)
        probs = env.circuit(params)

        # Save history for rendering
        env.history.append(probs)
        env.params = params  # update env params
        reward = -cost_val
        env.rewards.append(-cost_val)

        if reward > -1e-2:  # close to perfect
            break

    # Animate the full evolution
    env.render(save_path=file_path)
    assert os.path.exists(file_path), "Render did not create the expected file"


def test_invalid_ansatz_raises(simple_target):
    class BadAnsatz:
        pass
    with pytest.raises(ValueError):
        ProbabilityV0(n_qubits=1, target_distribution=simple_target, ansatz=BadAnsatz())
