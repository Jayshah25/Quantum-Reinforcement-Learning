from qrl.cross_entropy import CrossEntropy
from qrl.agents import ClassicalNNAgent, RandomQuantumAgent

import gymnasium as gym

env = gym.make("CartPole-v1")

classical_agent = ClassicalNNAgent(input_size=env.observation_space.shape[0],hidden_size=128,num_hidden_layers=2,output_size=env.action_space.n)
quantum_agent = RandomQuantumAgent(input_size=env.observation_space.shape[0])

solver = CrossEntropy(env=env)
solver.train_classical_and_quantum(classical_agent=classical_agent,quantum_agent=quantum_agent,plot=True)