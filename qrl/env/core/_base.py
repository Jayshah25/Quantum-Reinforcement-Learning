from abc import ABC, abstractmethod
import gymnasium as gym
import pennylane as qml


class QuantumEnv(gym.Env, ABC):
    """Abstract base class for all QRL quantum environments"""

    def __init__(self, n_qubits=1):
        super().__init__()

        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    @abstractmethod
    def get_reward(self):
        """Get reward for current state."""
        pass

    @abstractmethod
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply an action (quantum gate or sequence) and return the new observation and reward"""
        pass

    @abstractmethod
    def render(self):
        """Animate the episode"""
        pass