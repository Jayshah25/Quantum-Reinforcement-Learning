from qrl.env import make
import random


def render_sample_run(save_path:str):
    env = make("CleaningRobot-v0")
    env.reset()

    while not env.done:
        env.handle_events()
        action = random.choice(env.action_space())
        obs, reward, done, _ = env.step(action)
        env.render(capture=True)  # set capture=True to store frames

    env.save_video(save_path)
    env.close()
