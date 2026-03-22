import os
import pytest
import numpy as np
from qrl.env.core.bloch_sphere import BlochSphereV0, BlochSphereV1


def test_initialization():
    target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    env = BlochSphereV0(target_state=target_state, max_steps=10,reward_tolerance=0.99)
    assert len(env.history) == 0, "History should have length 0 after initialization"
    assert np.allclose(env.target_state, target_state), "Target state should match the user input"

@pytest.mark.parametrize("target_state, action, fidelity, message", [
    (np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex), "H", 1.0, "|<+|+>|^2 == 1"),
    (np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex), "H", 0.0, "|<-|+>|^2 == 0"),
])
def test_reward(target_state, action, fidelity, message):
    gate_dict = {"H": 0, "X": 1, "Y": 2, "Z": 3, "S": 4, "SDG": 5, "T": 6, "TDG": 7,
               "RX_pi_2": 8, "RX_pi_4": 9, "RX_-pi_4": 10,
               "RY_pi_2": 11, "RY_pi_4": 12, "RY_-pi_4": 13,
               "RZ_pi_2": 14, "RZ_pi_4": 15, "RZ_-pi_4": 16}
    env = BlochSphereV0(target_state=target_state, max_steps=10,reward_tolerance=0.99)
    _, reward, _, _ = env.step(action=gate_dict[action])
    assert round(reward, 2) == round(fidelity, 2), message

@pytest.mark.parametrize("ffmpeg, save_path, file_name, extension", [
    (True, r"results/tests", "bloch_sphereV0", "mp4"),
    (False, r"results/tests", "bloch_sphereV0", "gif"),
])
def test_sample_run_and_render(ffmpeg, save_path, file_name, extension):
    file_path = save_path + os.sep + file_name + "." + extension
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path)
    if not os.path.exists(save_path):  # check if the directory exists
        os.makedirs(save_path)

    target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # Target vector is |+>

    env = BlochSphereV0(target_state=target_state,max_steps=20,reward_tolerance=0.99,ffmpeg=ffmpeg)

    # Reset environment
    obs, _ = env.reset()

    # Randomly sample actions and execute a sample run
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        if done:
            break

    # Render Bloch sphere
    save_path_without_extension = save_path + os.sep + file_name
    env.render(save_path_without_extension=save_path_without_extension)

    assert os.path.exists(file_path), "Render did not create the expected file"


def test_initialization():
    env = BlochSphereV1(target_state=2, max_steps=10, reward_tolerance=0.99)
    assert env.target_state_index == 2,         "Target state index should match the user input"
    assert env.max_steps           == 10,        "max_steps should match the user input"
    assert env.observation_space.n == 6,         "Observation space should have 6 discrete states"
    assert env.action_space.n      == 4,         "Action space should have 4 discrete actions"
    assert env.history             == [],        "History should be empty before reset"
    assert env.terminated          is None,      "terminated should be None before reset"
    assert env.truncated           is None,      "truncated should be None before reset"


def test_reset():
    env = BlochSphereV1(target_state=3, max_steps=10, reward_tolerance=0.99)
    obs, info = env.reset()
    assert obs              == 0,        "Initial observation should always be state index 0 (|0⟩)"
    assert env.history      == [0],      "History should contain only the initial state after reset"
    assert env.steps        == 0,        "Step counter should be 0 after reset"
    assert env.terminated   is False,    "terminated should be False after reset"
    assert env.truncated    is False,    "truncated should be False after reset"
    assert "fidelity"       in info,     "Info dict should contain fidelity"
    assert "gate"           in info,     "Info dict should contain gate"
    assert info["gate"]     == "reset",  "Gate label in info should be 'reset' after reset"


@pytest.mark.parametrize("target_state, action, expected_next_state, message", [
    (3, 0, 2,  "H|0⟩ → |+⟩  (state index 2)"),
    (3, 1, 1,  "X|0⟩ → |1⟩  (state index 1)"),
    (3, 2, 0,  "Z|0⟩ → |0⟩  (global phase, stays at index 0)"),
    (0, 0, 2,  "H|0⟩ → |+⟩  (state index 2), different target"),
    (0, 3, 0,  "S|0⟩ → |0⟩  (global phase, stays at index 0)"),
])
def test_state_transitions(target_state, action, expected_next_state, message):
    env = BlochSphereV1(target_state=target_state, max_steps=10, reward_tolerance=0.99)
    env.reset()
    obs, _, _, _, info = env.step(action)
    assert obs                  == expected_next_state, message
    assert info["state_index"]  == expected_next_state, message


@pytest.mark.parametrize("target_state, action, expected_reward, message", [
    (2, 0, 1.0, "H|0⟩ → |+⟩: reward should be 1.0 (target reached)"),
    (1, 1, 1.0, "X|0⟩ → |1⟩: reward should be 1.0 (target reached)"),
    (2, 2, 0.0, "Z|0⟩ → |0⟩: reward should be 0.0 (target not reached)"),
    (2, 1, 0.0, "X|0⟩ → |1⟩: reward should be 0.0 (not |+⟩)"),
])
def test_reward(target_state, action, expected_reward, message):
    env = BlochSphereV1(target_state=target_state, max_steps=10, reward_tolerance=0.99)
    env.reset()
    _, reward, _, _, _ = env.step(action)
    assert reward == expected_reward, message


def test_termination_and_truncation():
    # --- terminated=True when target is reached ---
    env = BlochSphereV1(target_state=2, max_steps=10, reward_tolerance=0.99)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)   # H|0⟩ → |+⟩
    assert reward     == 1.0,  "Reward should be 1.0 on success"
    assert terminated is True, "terminated should be True when target is reached"
    assert truncated  is False,"truncated should be False when target is reached"

    # --- truncated=True when max_steps exhausted without success ---
    env = BlochSphereV1(target_state=2, max_steps=3, reward_tolerance=0.99)
    env.reset()
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(2)    # Z: |0⟩ → |0⟩ (no progress)
    assert terminated is False,"terminated should be False on truncation"
    assert truncated  is True, "truncated should be True when max_steps is exhausted"


@pytest.mark.parametrize("ffmpeg, save_path, file_name, extension", [
    (False, r"results/tests", "bloch_sphereV1", "gif"),
])
def test_sample_run_and_render(ffmpeg, save_path, file_name, extension):
    import torch

    file_path = save_path + os.sep + file_name + "." + extension
    if os.path.exists(file_path):
        os.remove(file_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    env = BlochSphereV1(target_state=4, max_steps=10, reward_tolerance=0.99, ffmpeg=ffmpeg)
    env.reset()

    # Run a random episode
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    # Minimal mock agent required by _render_graph
    class MockAgent:
        _V      = torch.zeros(6)
        _counts = torch.zeros(6, 4, 6)
        def get_policy(self): return torch.zeros(6, dtype=torch.long)

    env._render_graph(agent=MockAgent())
    save_path_without_extension = save_path + os.sep + file_name
    env.render(save_path_without_extension=save_path_without_extension,
               interval=200, ffmpeg=ffmpeg)

    assert os.path.exists(file_path), "Render did not create the expected file"