import pytest 
import numpy as np
from qrl.env.core.compiler import CompilerV0
from qrl.env.core.utils import RZ, RY
import os

def test_initialization():
    np.random.seed(42)  # For reproducibility
    theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
    target_unitary = (RZ(phi) @ RY(theta) @ RZ(lam))  # general SU(2)
    env = CompilerV0(target_unitary=target_unitary, max_steps=10, reward_tolerance=0.99)
    assert len(env.history) == 0, "History should have length 0 after initialization"
    assert np.allclose(env.target_unitary, target_unitary), "Target unitary should match the user input"


@pytest.mark.parametrize("target_unitary, action, gate_fidelity, message", [
    (np.eye(2, dtype=complex), "X", 0.0, "Trace(I†X) == 0"),
    ((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex), "H", 1.0, "Trace(H†H) == 1"),
])
def test_reward(target_unitary, action, gate_fidelity, message):
    gate_dict = {"H": 0, "X": 1, "Y": 2, "Z": 3, "S": 4, "SDG": 5, "T": 6, "TDG": 7,
               "RX_pi_2": 8, "RX_pi_4": 9, "RX_-pi_4": 10,
               "RY_pi_2": 11, "RY_pi_4": 12, "RY_-pi_4": 13,
               "RZ_pi_2": 14, "RZ_pi_4": 15, "RZ_-pi_4": 16}
    env = CompilerV0(target_unitary=target_unitary, max_steps=10,reward_tolerance=0.99)
    _, reward, _, _ = env.step(action=gate_dict[action])
    assert round(reward, 2) == round(gate_fidelity, 2), message

@pytest.mark.parametrize("ffmpeg, save_path_without_extension, extension", [
    (True, r"results/tests/compilerV0", "mp4"),
    (False, r"results/tests/compilerV0", "gif"),
])
def test_sample_run_and_render(ffmpeg, save_path_without_extension, extension):
    file_path = save_path_without_extension + "." + extension
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path)
    np.random.seed(42)  # For reproducibility
    theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
    target_unitary = (RZ(phi) @ RY(theta) @ RZ(lam))  # general SU(2)

    # Initialize environment with 1 qubit gate
    env = CompilerV0(target_unitary=target_unitary, max_steps=20, reward_tolerance=0.99,ffmpeg=ffmpeg)

    # Reset
    obs, _ = env.reset()

    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        if done:
            break

    # Render Bloch sphere
    env.render(save_path_without_extension=save_path_without_extension)

    assert os.path.exists(file_path), "Render did not create the expected file"