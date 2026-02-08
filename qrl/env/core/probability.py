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
    Probability distribution matching environment for variational quantum circuits.

    ``ProbabilityV0`` is a ``gymnasium.Env``-compatible environment that trains a
    parameterized quantum circuit to approximate a target probability distribution
    over computational basis states. The agent optimizes continuous circuit
    parameters so that the measurement statistics of the circuit match a specified
    target distribution.

    This environment is suitable for distribution learning, quantum generative
    modeling, and variational circuit optimization tasks.

    Key properties
    --------------
    - **Action space**: Continuous parameter updates applied to the circuit ansatz.
    - **Observation space**: Probability distribution over ``2**n_qubits`` basis
    states produced by the current circuit.
    - **Reward**: Negative weighted cost combining KL divergence and L2 distance to
    the target distribution, with an additional step penalty.
    - **Termination**: Success when the reward exceeds the specified tolerance or
    truncation at ``max_steps``.

    Visualization
    -------------
    The ``render()`` method animates the evolution of the learned probability
    distribution relative to the target distribution, along with the reward
    trajectory over training steps.

    Input Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    target_distribution : np.ndarray
        Target probability distribution over computational basis states.
    ansatz : callable or None
        Custom parameterized circuit ansatz. If ``None``, a default RY-based ansatz
        is used.
    max_steps : int
        Maximum number of optimization steps per episode.
    tolerance : float
        Reward threshold for early termination.
    alpha : float
        Weight balancing KL divergence and L2 distance.
    beta : float
        Penalty weight for step count.
    ffmpeg : bool
        Whether to use FFmpeg when saving animations.

    See Also
    --------
    :doc:`tutorials/probability`
        Tutorial on probability distribution learning with variational circuits.

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

    def get_reward(self, params):
        """
        Compute the reward for a given set of circuit parameters.

        The reward is based on a weighted combination of:
        - Kullback–Leibler (KL) divergence between the target distribution and
        the circuit output distribution.
        - L2 distance between the target and circuit distributions.

        Parameters
        ----------
        params : np.ndarray
            Vector of variational circuit parameters.

        Returns
        -------
        float
            Scalar reward value encouraging the circuit output distribution
            to match the target distribution.
        """
        probs = self.circuit(params)

        # KL divergence (target || probs)
        kl_div = np.sum(self.target_distribution * np.log((self.target_distribution + 1e-10) / (probs + 1e-10)))

        # L2 error
        l2_error = np.linalg.norm(self.target_distribution - probs, ord=2)

        # Reward
        reward = -(self.alpha * kl_div + (1 - self.alpha) * l2_error)

        return -reward

    def step(self, action):
        """
        Execute one optimization step.

        Updates the circuit parameters using the provided action, evaluates
        the resulting probability distribution, computes the reward, and
        checks termination conditions.

        Parameters
        ----------
        action : np.ndarray
            Parameter update vector applied additively to the current
            circuit parameters.

        Returns
        -------
        observation : np.ndarray
            Probability distribution over computational basis states produced
            by the circuit after the parameter update.
        reward : float
            Reward value after applying the action.
        done : bool
            True if the episode has terminated due to reaching the reward
            tolerance or the maximum number of steps.
        info : dict
            Empty dictionary provided for compatibility with Gymnasium-style APIs.
        """
        self.params = (self.params + action)  # keep params bounded
        self.current_step += 1

        probs = self.circuit(self.params)
        reward = self.get_reward(self.params)

        done = reward < self.tolerance or self.current_step >= self.max_steps
        self.history.append(probs)
        self.rewards.append(reward)

        return probs, reward, done, {}



    def reset(self):
        """
        Reset the environment to a random initial parameter configuration.

        Initializes the circuit parameters randomly, clears episode history,
        and resets the step counter.

        Returns
        -------
        observation : np.ndarray
            Initial circuit parameter vector.
        info : dict
            Empty dictionary provided for compatibility with Gymnasium-style APIs.
        """
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.current_step = 0
        self.history = []
        self.rewards = []
        return self.params, {}

    def render(self, save_path_without_extension=None):
        """
        Render the evolution of the probability distribution over training steps.

        The animation shows a bar plot comparing the target probability
        distribution with the circuit's predicted distribution at each step.
        Reward values are displayed in the plot title.

        Parameters
        ----------
        save_path_without_extension : str or None, optional
            Path (without file extension) to save the animation.
            If provided, the animation is saved using the configured writer
            (MP4 for FFmpeg or GIF for Pillow). If None, the animation is
            displayed interactively.

        Returns
        -------
        None
            This method produces a visualization but does not return a value.
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

