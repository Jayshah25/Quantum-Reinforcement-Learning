
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """Abstract base class for all Quantum Reinforcement Learning environments."""

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns the initial observation."""
        pass

    @abstractmethod
    def step(self, action):
        """Takes an action and returns a tuple (observation, reward, done, info)."""
        pass

    @abstractmethod
    def render(self):
        """Renders the environment (optional for quantum environments)."""
        pass

    @abstractmethod
    def action_space(self):
        """Returns the available actions."""
        pass

    @abstractmethod
    def observation_space(self):
        """Returns the observation space."""
        pass
