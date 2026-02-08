'''
Implementation of ErrorChannelV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''

from gymnasium import spaces
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import shutil
from .base__ import QuantumEnv


class ErrorChannelV0(QuantumEnv):
    """
    Multi-qubit error mitigation environment with bit-flip noise.

    ``ErrorChannelV0`` is a ``gymnasium.Env``-compatible environment that models a
    noisy multi-qubit quantum system affected by independent bit-flip error channels.
    Each qubit may experience noise with a different probability, and the agent’s
    task is to apply corrective Pauli-X operations to recover the target computational
    basis state ``|0…0⟩``.

    The environment captures a simplified quantum error mitigation scenario, where
    the agent sequentially selects qubits on which to apply corrections based on
    observed measurement probabilities.

    Key properties
    --------------
    - **Action space**: Discrete choice of qubit index on which to apply an ``X`` gate.
    - **Observation space**: Probability distribution over all computational basis
    states (shape ``(2**n_qubits,)``).
    - **Reward**: Negative mean-squared error between the corrected distribution and
    the ideal ``|0…0⟩`` distribution.
    - **Termination**: Success when perfect correction is achieved or truncation at
    ``max_steps``.

    Rendering
    ---------
    The ``render()`` method visualizes the mitigation process using a side-by-side
    animation that compares ideal, noisy, and corrected probability distributions,
    along with a dynamically updated circuit diagram showing applied corrections.

    Input Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    faulty_qubits : dict[int, float] or None
        Mapping from qubit indices to bit-flip probabilities.
    max_steps : int
        Maximum number of correction steps per episode.
    seed : int or None
        Random seed for reproducibility.
    ffmpeg : bool
        Whether to use FFmpeg to save animations as ,p4 or save it as GIFs with Pillow.

    See Also
    --------
    :doc:`tutorials/error_channel`
        Tutorial on multi-qubit error mitigation with bit-flip noise.

    """

    def __init__(
        self,
        n_qubits: int = 3,
        faulty_qubits: dict = None,   # {qubit_idx: error_type, ...}
        max_steps: int = 10,
        seed: int = None,
        ffmpeg: bool = False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.GATE_ID2NAME = {0: "X", 1: "Y", 2: "Z"}


        # Default: one random faulty qubit with random noise prob
        if faulty_qubits is None:
            qubit = int(self.rng.integers(0, n_qubits))
            noise = self.rng.uniform(0.1, 0.5)  # noise prob between 0.1 and 0.5
            faulty_qubits = dict(zip([qubit], [noise]))
        self.faulty_qubits = faulty_qubits

        # Device
        self.dev = qml.device("default.mixed", wires=n_qubits)

        # Action = (gate, qubit)
        self.action_space = spaces.MultiDiscrete([n_qubits])

        # Observation = probs over 2^n states
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2**n_qubits,), dtype=np.float32
        )

        # Target = |0...0>
        self.target_probs = np.zeros(2**n_qubits)
        self.target_probs[0] = 1.0

        self.history = []
        self.corrections = []
        self.current_step = 0

        self._build_qnodes()
        self.reset()
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension = "mp4" if ffmpeg else "gif"

        if ffmpeg==True and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found on system. Please install ffmpeg or set ffmpeg=False")


        # util vars
        self.corrected = None

    def _apply_noise(self):
        """
        Apply bit-flip noise channels to the faulty qubits.

        For each qubit specified in ``self.faulty_qubits``, a PennyLane
        ``BitFlip`` channel is applied with the corresponding noise probability.

        Returns
        -------
        None
            This method applies noise operations to the quantum circuit but
            does not return a value.
        """
        for qubit, noise in self.faulty_qubits.items():
            qml.BitFlip(noise, wires=qubit)

    def _apply_gate(self, wire):
        """
        Apply a corrective single-qubit Pauli-X gate.

        This operation represents a bit-flip correction applied to a specific
        qubit wire.

        Parameters
        ----------
        wire : int
            Index of the qubit on which the Pauli-X correction is applied.

        Returns
        -------
        None
            This method applies a quantum gate but does not return a value.
        """
        qml.PauliX(wires=wire)

    def _build_qnodes(self):
        """
        Build PennyLane QNodes for noisy and corrected circuits.

        This method constructs and assigns three QNodes:

        1. ``qnode_noisy``:
        Circuit with noise applied and no corrections.
        2. ``qnode_corrected``:
        Circuit with noise followed by the currently selected correction
        operations.
        3. ``qnode_draw``:
        A separate copy of the corrected circuit used exclusively for
        rendering and visualization.

        Returns
        -------
        None
            This method initializes internal QNode attributes but does not
            return a value.
        """
        @qml.qnode(self.dev)
        def qnode_noisy():
            self._apply_noise()
            return qml.probs(wires=range(self.n_qubits))
        self.qnode_noisy = qnode_noisy

        @qml.qnode(self.dev)
        def qnode_corrected(k: int):
            self._apply_noise()
            for i in range(k):
                self._apply_gate(self.corrections[i])
            return qml.probs(wires=range(self.n_qubits))
        self.qnode_corrected = qnode_corrected

        # separate copy of corrected circuit for rendering
        @qml.qnode(self.dev)
        def qnode_draw(k: int):
            self._apply_noise()
            for i in range(k):
                self._apply_gate(self.corrections[i])
            return qml.probs(wires=range(self.n_qubits))
        self.qnode_draw = qnode_draw


    def reset(self, *, seed=None):
        """
        Reset the environment to the initial noisy state.

        Clears the correction history, resets the step counter, and evaluates
        the noisy circuit without any corrective operations.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility. If provided, reinitializes the
            internal random number generator.

        Returns
        -------
        observation : np.ndarray
            Initial corrected probability distribution over computational
            basis states (identical to the noisy distribution at reset),
            with dtype ``float32``.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.corrections.clear()
        self.history.clear()

        probs_noisy = self.qnode_noisy()
        probs_corrected = np.array(probs_noisy, copy=True)
        reward = 0.0

        self.history.append(dict(
            step=0, 
            # gate=None, 
            qubit=None,
            probs_noisy=probs_noisy,
            probs_corrected=probs_corrected,
            reward=reward,
        ))

        return probs_corrected.astype(np.float32)
    
    def get_reward(self, action):
        """
        Apply a correction action and compute the reward.

        The selected qubit index is appended to the correction list, the noisy
        and corrected circuits are evaluated, and the reward is computed as the
        negative mean-squared error between the corrected and target probability
        distributions.

        Parameters
        ----------
        action : int
            Index of the qubit on which a Pauli-X correction is applied.

        Returns
        -------
        float
            Reward value defined as the negative mean-squared error between
            the corrected probability distribution and the target distribution.
        """
        qubit_idx = int(action)
        self.corrections.append(qubit_idx)
        self.current_step += 1
        k = len(self.corrections)
        noisy = self.qnode_noisy() # noisy circuit

        self.corrected = self.qnode_corrected(k) # noisy + corrections circuit

        reward = -np.mean((self.target_probs - self.corrected)**2) # Minimum can be 0
        self.history.append(dict(
        step=self.current_step,
        # gate=self.GATE_ID2NAME[gate_id],
        qubit=qubit_idx,
        probs_noisy=noisy,
        probs_corrected=self.corrected,
        reward=reward,
        ))
        
        return reward

    def step(self, action):
        """
        Execute one environment step.

        Applies a correction action, updates internal state and history,
        computes the reward, and checks termination conditions.

        Parameters
        ----------
        action : int
            Index of the qubit on which a Pauli-X correction is applied.

        Returns
        -------
        observation : np.ndarray
            Corrected probability distribution over computational basis states,
            with dtype ``float32``.
        reward : float
            Negative mean-squared error between the corrected and target
            distributions.
        done : bool
            True if the episode has terminated due to reaching the maximum
            number of steps or achieving perfect correction.
        info : dict
            Dictionary containing metadata about the environment, including
            the mapping of faulty qubits.
        """
        
        reward = self.get_reward(action)
        done = self.current_step >= self.max_steps or np.round(reward, 3) == 0.0
        obs = self.corrected.astype(np.float32)
        info = {"faulty_qubits": self.faulty_qubits}
        return obs, reward, done, info

    def _basis_labels(self):
        """
        Generate labels for computational basis states.

        Returns
        -------
        list of str
            List of strings representing computational basis states in Dirac
            notation (e.g., ``"|000⟩"``, ``"|001⟩"``).
        """
        return [f"|{i:0{self.n_qubits}b}⟩" for i in range(2**self.n_qubits)]

    def render(self, save_path_without_extension=None, interval_ms=600):
        """
        Render the error-mitigation process as an animated visualization.

        The animation consists of:
        - A bar chart comparing ideal, noisy, and corrected probability
        distributions for each computational basis state.
        - A dynamically updated ASCII-style circuit diagram showing the
        applied correction operations.

        Parameters
        ----------
        save_path_without_extension : str or None, optional
            Path (without file extension) to save the animation.
            If provided, the animation is saved using the configured writer
            (MP4 for FFmpeg or GIF for Pillow). If None, the animation is
            displayed interactively.
        interval_ms : int, optional
            Time between animation frames in milliseconds. Default is 600.

        Returns
        -------
        None
            This method produces a visualization but does not return a value.
        """
        if not self.history:
            print("Nothing to animate.")
            return

        n_states = 2**self.n_qubits
        x = np.arange(n_states)
        labels = self._basis_labels()

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
        plt.tight_layout()

        width = 0.25
        bars_ideal = axL.bar(x - width, self.target_probs, width, alpha=0.5, label="Ideal")
        bars_noisy = axL.bar(x, np.zeros(n_states), width, alpha=0.8, label="Noisy")
        bars_corr  = axL.bar(x + width, np.zeros(n_states), width, alpha=0.8, label="Corrected")

        axL.set_ylim(0, 1)
        axL.set_xticks(x)
        axL.set_xticklabels(labels)
        axL.set_ylabel("Probability")
        axL.legend(loc="upper right")

        axR.axis("off")
        axR.set_title("Circuit")
        text_box = axR.text(0.5, 0.5, "", ha="center", va="center", fontsize=12, family="monospace")


        def draw_circuit_on_axis(k):      
            text = qml.draw(self.qnode_draw)(k)
            text_box.set_text(text)


        def update(frame):
            rec = self.history[frame]
            noisy, corr = rec["probs_noisy"], rec["probs_corrected"]

            for b, h in zip(bars_noisy, noisy): b.set_height(h)
            for b, h in zip(bars_corr, corr): b.set_height(h)

            title = f"Step {rec['step']}"
            # if rec["gate"] is not None:
            title += f" | qubit [{rec['qubit']}]"
            title += f" | reward={rec['reward']:.3f}"
            axL.set_title(title)

            draw_circuit_on_axis(min(rec["step"], len(self.corrections)))
            return (*bars_noisy, *bars_corr)

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=interval_ms, blit=False)

        if save_path_without_extension:
            fps = max(1, int(1000/interval_ms))
            ani.save(f"{save_path_without_extension}.{self.render_extension}", writer=self.writer, fps=fps)
        else:
            plt.show()

