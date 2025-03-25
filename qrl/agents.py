import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

class ClassicalNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        '''
        input_size (int): It represents the observation space size
        hidden_size (int): Number of neruons in the hidden layers
        num_hidden_layers (int): Number of hidden layers
        output_size (int): It represents the action space size (number of actions)
        '''
        super(ClassicalNNAgent, self).__init__()
        layers = []


        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers (all having the same hidden_size)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer (no activation, can add softmax/sigmoid if needed)
        layers.append(nn.Linear(hidden_size, output_size))

        # Wrap layers in nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.network(x)


class RandomQuantumAgent(nn.Module):
    def __init__(self, input_size:int,n_layers:int=2,n_rotations:int=3, seed:int=42):
        '''
        input_size (int): It represents the observation space size and the number of qubits in the quantum circuit
        '''
        super(RandomQuantumAgent, self).__init__()
        dev = qml.device("default.qubit", wires=input_size)

        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(input_size))
            qml.RandomLayers(weights=weights, wires=range(input_size), seed=seed)
            return qml.probs(wires=0)
                
        shape = qml.RandomLayers.shape(n_layers=n_layers, n_rotations=n_rotations)
        weight_shapes = {"weights": shape}
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes) #variational quantum circuit

    def forward(self, x: torch.Tensor):
        return self.vqc(x).reshape(-1,1)