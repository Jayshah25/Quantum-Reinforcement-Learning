'''
Implementation of ProbabilityV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''
from gymnasium import spaces
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
from .base__ import QuantumEnv

class ProbabilityV0(QuantumEnv):
    """
    Reinforcement learning environment for training parameterized quantum circuits 
    to approximate a target probability distribution over computational basis states.
    It is based on the `QuantumEnv` base class.

    **Purpose**
    ------------
    The goal of `ProbabilityV0` is to optimize variational quantum circuits such that 
    the measured probability distribution of outcomes matches a given target 
    distribution. This is useful in tasks like quantum compilation, quantum generative 
    modeling, and distribution learning.

    **Environment Dynamics**
    ------------------------
    - **State (observation):**
      The current probability distribution over `2**n_qubits` basis states obtained 
      from the quantum circuit.

    - **Action:**
      A continuous vector of parameter updates (`Box(low=-0.1, high=0.1, shape=(n_params,)`), 
      which perturbs the trainable parameters of the ansatz.

    - **Reward:**
      Defined as the *negative* of a weighted cost combining:
        - KL divergence between the target distribution and the circuit's output.
        - L2 distance between the target and circuit distributions.
      The weighting is controlled by `alpha` (for KL vs. L2) and `beta` (step penalty).

    - **Episode Termination:**
      - If the reward is below the specified tolerance.
      - If the number of steps reaches `max_steps`.

    **Key Parameters**
    -----------------
    - `n_qubits (int)`: Number of qubits in the circuit.
    - `target_distribution (np.ndarray)`: Target probability distribution (must sum to 1).
    - `ansatz (callable, optional)`: Custom circuit ansatz. Defaults to a simple 
      layer of RY rotations if not provided.
    - `max_steps (int, default=100)`: Maximum steps per episode.
    - `tolerance (float, default=-1e3)`: Reward threshold for termination.
    - `alpha (float, default=0.5)`: Weight between KL divergence and L2 error.
    - `beta (float, default=0.01)`: Penalty weight for step count.
    - `ffmpeg` (bool, default=False): If `True`, uses FFmpeg for saving animations; otherwise uses Pillow (GIF).

    **Visualization**
    ----------------
    - `render()`: Creates an animation showing the evolution of the learned 
      distribution and corresponding rewards across training.


    **Applications**
    ----------------
    - Distribution learning with quantum circuits.
    - Testing expressivity of variational ansätze.
    - Benchmarking optimization of quantum neural networks.
    
    """
    def __init__(self, 
                 n_qubits: int,
                 target_distribution: np.ndarray,
                 ansatz=None,**kwargs):
        super(ProbabilityV0, self).__init__()

        assert np.isclose(np.sum(target_distribution), 1.0), \
            "Target distribution must sum to 1."
        self.n_qubits = n_qubits
        self.target_distribution = target_distribution
        self.max_steps = kwargs.get("max_steps", 100)
        self.tolerance = kwargs.get("tolerance", -1e3)
        self.alpha = kwargs.get("alpha", 0.5)  # weight for KL vs L2
        self.beta = kwargs.get("beta", 0.01)    # step penalty weight
        ffmpeg = kwargs.get("ffmpeg", False)
        self.render_extension = "mp4" if ffmpeg else "gif"
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        if ffmpeg==True and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found on system. Please install ffmpeg or set ffmpeg=False")


        # Define PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # If no ansatz is provided, define a simple one
        if ansatz is None:
            def default_ansatz(params, wires):
                for i, w in enumerate(wires):
                    qml.RY(params[i], wires=w)
            self.ansatz = default_ansatz
            self.n_params = self.n_qubits
        else:
            self.ansatz = ansatz
            try:
                self.n_params = ansatz.n_params  # If ansatz object has attribute
            except:
                raise ValueError("Please specify ansatz with n_params attribute.")

        # QNode
        @qml.qnode(self.dev)
        def circuit(params):
            self.ansatz(params, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        self.circuit = circuit

        # Spaces
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2**self.n_qubits,), dtype=np.float32)

        # Internal state
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.current_step = 0
        self.history = []
        self.rewards = []

    def cost_fn(self, params):
        probs = self.circuit(params)

        # KL divergence (target || probs)
        kl_div = np.sum(self.target_distribution * np.log((self.target_distribution + 1e-10) / (probs + 1e-10)))

        # L2 error
        l2_error = np.linalg.norm(self.target_distribution - probs, ord=2)

        # Reward
        reward = -(self.alpha * kl_div + (1 - self.alpha) * l2_error)

        return -reward

    def step(self, action):
        self.params = (self.params + action)  # keep params bounded
        self.current_step += 1

        probs = self.circuit(self.params)
        reward = self.cost_fn(self.params)

        done = reward < self.tolerance or self.current_step >= self.max_steps
        self.history.append(probs)
        self.rewards.append(reward)

        return probs, reward, done, {}



    def reset(self):
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.current_step = 0
        self.history = []
        self.rewards = []
        return self.params, {}

    def render(self, save_path_without_extension=None):
        """
        Create an animation showing how the distribution evolves over steps,
        including reward values in the title.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(2**self.n_qubits)
        width = 0.4

        target_bar = ax.bar(x - 0.2, self.target_distribution, width=width, label="Target")
        current_bar = ax.bar(x + 0.2, self.history[0], width=width, label="Prediction")

        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(self.history[0])))
        ax.set_xticklabels([f"|{i}⟩" for i in range(len(self.history[0]))])
        ax.set_xlabel("Basis states")
        ax.set_ylabel("Probability")
        ax.legend()
        def update(frame):
            probs = self.history[frame]
            for bar, new_height in zip(current_bar, probs):
                bar.set_height(new_height)
            ax.set_title(f"Step {frame} | Reward: {np.array(self.rewards[frame].item()):.4f}")
            return current_bar

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), blit=False)

        if save_path_without_extension:
            ani.save(f"{save_path_without_extension}.{self.render_extension}", writer=self.writer, fps=2)
        else:
            plt.show()

