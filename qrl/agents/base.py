from abc import ABC, abstractmethod
import gymnasium as gym
import pennylane as qml

class BaseAgent(ABC):
    def select_action(self, observation):
        # 'forward' method for PyTorch based DL agents
        if hasattr(self, 'forward'):
            return self.forward(observation)
        # 'get_action' method for non-DL based agents
        elif hasattr(self, 'get_action'):
            return self.get_action(observation)
        else:
            raise NotImplementedError("Agent must implement either 'get_action' or 'forward' method.")

    @abstractmethod
    def train(self, env):
        # encodes the training logic of the underlying RL technique
        pass

    @abstractmethod
    def get_frames(self, agent_type):
        # get the recorded env frames during training/inference for the agent
        # aget_type can be 'classical' or 'quantum'
        pass
