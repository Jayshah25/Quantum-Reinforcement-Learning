import os
import pytest
import numpy as np
from qrl.env.core.bloch_sphere import BlochSphereV0


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


def test_sample_run_and_render():
    file_path = r"results/tests/bloch_sphere.mp4"
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path)   
     
    target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # Target vector is |+>

    env = BlochSphereV0(target_state=target_state,max_steps=20,reward_tolerance=0.99)

    # Reset environment
    obs, _ = env.reset()

    # Randomly sample actions and execute a sample run
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        if done:
            break

    # Render Bloch sphere
    env.render(save_path=file_path)

    assert os.path.exists(file_path), "Render did not create the expected file"