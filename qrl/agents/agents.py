import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class ClassicalNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        """
        Feedforward neural network agent for discrete action prediction.

        ``ClassicalNNAgent`` implements a fully connected multilayer perceptron
        (MLP) that maps observations to action logits or values. It is intended
        for use as a baseline classical agent in reinforcement learning settings.

        The network consists of an input layer, a configurable number of hidden
        layers with ReLU activations, and a linear output layer.

        Parameters
        ----------
        input_size : int
            Dimensionality of the observation space.
        hidden_size : int
            Number of neurons in each hidden layer.
        num_hidden_layers : int
            Number of hidden layers in the network.
        output_size : int
            Dimensionality of the output space (e.g., number of actions).
        """
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
    def __init__(self, input_size:int, output_size:int, n_layers:int=2, n_rotations:int=3, seed:int=42):
        """
        Hybrid quantum-classical agent with a random variational quantum circuit.

        ``RandomQuantumAgent`` implements a PennyLane-based variational quantum
        circuit (VQC) wrapped as a PyTorch module. The circuit uses angle
        embedding for inputs and randomly structured layers to generate quantum
        features, which are measured as Pauli-Z expectation values.

        This agent is primarily intended as a quantum baseline or exploratory
        agent rather than a fully trainable policy network.

        Parameters
        ----------
        input_size : int
            Dimensionality of the observation space and the number of qubits
            in the quantum circuit.
        output_size : int
            Number of measured qubits / outputs produced by the circuit.
        n_layers : int, optional
            Number of random quantum layers. Default is 2.
        n_rotations : int, optional
            Number of rotation gates per random layer. Default is 3.
        seed : int, optional
            Random seed controlling the structure of the random quantum layers.
            Default is 42.
        """
        super(RandomQuantumAgent, self).__init__()
        dev = qml.device("default.qubit", wires=input_size)

        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(input_size))
            qml.RandomLayers(weights=weights, wires=range(input_size), seed=seed)
            return [qml.expval(qml.PauliZ(i)) for i in range(output_size)]
                
        shape = qml.RandomLayers.shape(n_layers=n_layers, n_rotations=n_rotations)
        weight_shapes = {"weights": shape}
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes) #variational quantum circuit

    def forward(self, x: torch.Tensor):
        return self.vqc(x).reshape(-1,1)