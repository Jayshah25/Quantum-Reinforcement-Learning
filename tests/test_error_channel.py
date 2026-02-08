import pytest 
import numpy as np
from qrl.env.core.error_channel import ErrorChannelV0
import os

def test_initialization():
    env = ErrorChannelV0(
        n_qubits=3,
        faulty_qubits={0: 1.0, 2: 1.0},
        max_steps=6,
        seed=42,
    )
    assert len(env.corrections) == 0, "Corrections should have length 0 after initialization"


@pytest.mark.parametrize("error_qubit, error_prob, action_qubit, expected_reward, message", [
    (0, 1.0, 0, 0.0, "Bit-flip error corrected"),
])
def test_reward(error_qubit, error_prob, action_qubit, expected_reward, message):
    env = ErrorChannelV0(
        n_qubits=3,
        faulty_qubits={error_qubit: error_prob},
        max_steps=6,
        seed=42,
    )
    _, reward, _, _ = env.step(action_qubit)
    assert np.round(reward, 2) == round(expected_reward, 2), message

@pytest.mark.parametrize("ffmpeg, save_path, file_name, extension", [
    (True, r"results/tests", "error_channelV0", "mp4"),
    (False, r"results/tests", "error_channelV0", "gif"),
])
def test_sample_run_and_render(ffmpeg, save_path, file_name, extension):
    file_path = save_path + os.sep + file_name + "." + extension
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path)

    if not os.path.exists(save_path):  # check if the directory exists
        os.makedirs(save_path)

    np.random.seed(42)  # For reproducibility
    env = ErrorChannelV0(
        n_qubits=3,
        faulty_qubits={0: 1.0, 2: 1.0},
        max_steps=6,
        seed=42,
        ffmpeg=ffmpeg
    )
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
    save_path_without_extension = save_path + os.sep + file_name
    env.render(save_path_without_extension=save_path_without_extension, interval_ms=700)

    assert os.path.exists(file_path), "Render did not create the expected file"