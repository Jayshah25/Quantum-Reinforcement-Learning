import pytest 
import numpy as np
from qrl.env.core.expressibility import ExpressibilityV0
import os

@pytest.fixture
def env_fixture():
    return ExpressibilityV0(n_qubits=2, n_pairs_eval=10, bins=10, seed=42, max_steps=5)

def test_initialization(env_fixture):
    env = env_fixture
    assert len(env.history) == 0, "History should have length 0 after initialization"
    assert env.current_step == 0, "Current step should have length 0 after initialization"
    assert len(env.blocks) == 0, "Blocks should have length 0 after initialization"

def test_reward_penalizes_depth(env_fixture):
    """Check that adding more depth reduces reward due to lambda_depth penalty."""
    env = env_fixture
    obs, _ = env.reset()

    # Step with a RotX (depth = 1)
    obs1, reward1, done1, info1 = env.step(0)

    # Step with another RotX (depth = 2)
    obs2, reward2, done2, info2 = env.step(0)

    # Reward2 should be less than or equal to Reward1 because of extra depth penalty
    assert reward2 <= reward1, f"Expected reward2 <= reward1, got {reward2} > {reward1}"

    # Step with a RotX (depth = 1)
    obs1, reward1, done1, info1 = env.step(0)

    # Step with another RotX (depth = 2)
    obs2, reward2, done2, info2 = env.step(0)

    # Reward2 should be less than or equal to Reward1 because of extra depth penalty
    assert reward2 <= reward1, f"Expected reward2 <= reward1, got {reward2} > {reward1}"

@pytest.mark.parametrize("ffmpeg, save_path, file_name, extension", [
    (True, r"results/tests", "expressibilityV0", "mp4"),
    (False, r"results/tests", "expressibilityV0", "gif"),
])
def test_sample_run_and_render(ffmpeg, save_path, file_name, extension):
    file_path = save_path + os.sep + file_name + "." + extension
    if os.path.exists(file_path):   # check if the file exists
        os.remove(file_path) 
        
    if not os.path.exists(save_path):  # check if the directory exists
        os.makedirs(save_path)
  
    env = ExpressibilityV0(n_qubits=3, n_pairs_eval=60, bins=40, seed=7,ffmpeg=ffmpeg)
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total += reward
    save_path_without_extension = save_path + os.sep + file_name
    env.render(save_path_without_extension=save_path_without_extension)

    assert os.path.exists(file_path), "Render did not create the expected file"