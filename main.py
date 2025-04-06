from qrl.cross_entropy import CrossEntropy
from qrl.agents import ClassicalNNAgent, RandomQuantumAgent
from qrl.utils import render_sample_run

import gymnasium as gym


def train_classical_and_quantum_():
    env = gym.make("CartPole-v1")

    classical_agent = ClassicalNNAgent(input_size=env.observation_space.shape[0],hidden_size=128,num_hidden_layers=2,output_size=env.action_space.n)
    quantum_agent = RandomQuantumAgent(input_size=env.observation_space.shape[0])

    solver = CrossEntropy(env=env)
    solver.train_classical_and_quantum(classical_agent=classical_agent,quantum_agent=quantum_agent,plot=True)

render_sample_run(save_path=r"videos/cleaning_robot_episode.mp4")