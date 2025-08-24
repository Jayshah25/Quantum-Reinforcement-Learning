from abc import ABC, abstractmethod
import gymnasium as gym
import pennylane as qml


class QuantumEnv(gym.Env, ABC):
    """Abstract base class for all QRL quantum environments"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_qubits=1):
        super().__init__()

        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    @abstractmethod
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply an action (quantum gate or sequence) and return the new observation and reward"""
        pass

    @abstractmethod
    def render(self, mode="human"):
        """Visualize current state """
        pass

    def close(self):
        pass