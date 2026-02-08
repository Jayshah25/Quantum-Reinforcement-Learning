'''
Implementation of ExpressibilityV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''

from typing import List, Tuple, Optional, Dict
import pennylane as qml
from pennylane import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import shutil
from .base__ import QuantumEnv


class ExpressibilityV0(QuantumEnv):
    """
    Parameterized circuit expressibility optimization environment.

    ``ExpressibilityV0`` is a ``gymnasium.Env``-compatible environment that models
    the construction of parameterized quantum circuits with high expressibility.
    In the context of variational quantum algorithms, expressibility measures how
    well an ansatz can explore the Hilbert space of quantum states relative to the
    Haar-random distribution.

    The agent incrementally builds a circuit by adding or removing predefined
    rotation and entangling blocks, or by explicitly terminating construction.
    Rewards encourage circuits whose fidelity distribution closely matches the
    Haar distribution, while penalizing excessive circuit depth and two-qubit gate
    usage.

    Key properties
    --------------
    - **Action space**: Discrete set of architectural edits (add/remove blocks or
    terminate construction).
    - **Observation space**: Vector of circuit statistics summarizing depth,
    parameter count, entanglement, and recent expressibility estimates
    (shape ``(7,)``).
    - **Reward**: Negative KL divergence to the Haar distribution with regularization
    penalties for depth and two-qubit gates.
    - **Termination**: Explicit termination by the agent or truncation at
    ``max_steps``.

    Rendering
    ---------
    The ``render()`` method visualizes expressibility optimization via a two-panel
    animation showing the circuit’s fidelity distribution compared to the
    Haar-random distribution alongside a block-level diagram of the evolving
    circuit architecture.

    Input Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    max_blocks : int
        Maximum number of blocks allowed in the circuit.
    max_steps : int
        Maximum number of construction steps per episode.
    n_pairs_eval : int
        Number of random state pairs used to estimate expressibility.
    bins : int
        Number of histogram bins for fidelity distributions.
    lambda_depth : float
        Penalty weight for circuit depth.
    lambda_2q : float
        Penalty weight for two-qubit gate usage.
    terminate_bonus : float
        Bonus reward for explicit termination.
    device_name : str
        PennyLane device backend used for simulation.
    seed : int or None
        Random seed for reproducibility.
    allow_all_to_all : bool
        Whether to allow all-to-all entangling blocks.
    ffmpeg : bool
        Whether to use FFmpeg when saving animations.

    See Also
    --------
    :doc:`tutorials/expressibility`
        Tutorial on optimizing ansatz expressibility with block-based circuits.

    """

    def __init__(
        self,
        n_qubits: int = 4,
        max_blocks: int = 12,
        max_steps: int = 20,
        n_pairs_eval: int = 120,
        bins: int = 50,
        lambda_depth: float = 0.002,
        lambda_2q: float = 0.002,
        terminate_bonus: float = 0.1,
        device_name: str = "default.qubit",
        seed: Optional[int] = None,
        allow_all_to_all: bool = False,
        ffmpeg: bool = False,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.max_blocks = max_blocks
        self.max_steps = max_steps
        self.n_pairs_eval = n_pairs_eval
        self.bins = bins
        self.lambda_depth = lambda_depth
        self.lambda_2q = lambda_2q
        self.terminate_bonus = terminate_bonus
        self.allow_all_to_all = allow_all_to_all

        self.D = 2 ** n_qubits
        self._rng = np.random.default_rng(seed)

        self.blocks: List[str] = []
        self.device = qml.device(device_name, wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.device) #self._circuit is method defined below

        self.ACTION_NAMES = [
                        "RotX",
                        "RotY",
                        "RotZ",
                        "RotXYZ",
                        "EntRingCNOT",
                        "EntLadderCZ",
                        "RemoveLast",
                        "Terminate",
                        ]

        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))

        high = np.array([
            10_000, # depth
            max_blocks, # number of blocks
            10_000, # number of two-qubit gates
            10_000, # number of rotational trainable parameters
            1_000, # entanglement density
            10.0, # last value of expressibility 
            max_steps # steps left
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)

        self.current_step = 0
        self.last_express = None
        self.last_reward = 0.0
        self.info_last_hist = None
        self.history = []
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension = "mp4" if ffmpeg else "gif"
        if ffmpeg==True and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found on system. Please install ffmpeg or set ffmpeg=False")
        
        # utils
        self.done = None
        self.terminated = None
        self.obs = None
        self.info = None

    def get_reward(self, action):
        """

        The selected action modifies the circuit architecture by adding,
        removing, or terminating block construction. Expressibility is
        evaluated after the update, and a reward is computed based on the
        circuit's deviation from the Haar distribution and architectural
        penalties.

        Parameters
        ----------
        action : int
            Index of the selected architectural action.

        Returns
        -------
        reward : float
            Reward value combining expressibility and architectural penalties.
        """

        assert self.action_space.contains(action)
        self.done = False
        self.terminated = False

        # available actions
        if action == 0 and len(self.blocks) < self.max_blocks:
            self.blocks.append("RotX")
        elif action == 1 and len(self.blocks) < self.max_blocks:
            self.blocks.append("RotY")
        elif action == 2 and len(self.blocks) < self.max_blocks:
            self.blocks.append("RotZ")
        elif action == 3 and len(self.blocks) < self.max_blocks:
            self.blocks.append("RotXYZ")
        elif action == 4 and len(self.blocks) < self.max_blocks:
            self.blocks.append("EntRingCNOT")
        elif action == 5 and len(self.blocks) < self.max_blocks:
            self.blocks.append("EntLadderCZ")
        elif action == 6 and self.blocks:
            self.blocks.pop()
        elif action == 7:
            self.done = True
            self.terminated = True

        # update current step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # get expressibility for the current circuit
        express, kl, hist_c, hist_haar = self._expressibility()
        self.last_express = express
        self.info_last_hist = {"P_C": hist_c, "P_Haar": hist_haar}

        depth, n_blocks, n_twoq, n_params, ent_density = self._arch_stats()

        reward = -kl - self.lambda_depth * depth - self.lambda_2q * n_twoq
        if self.terminated:
            reward += self.terminate_bonus

        self.last_reward = reward

        # check if the action is supported
        self.obs = self._make_obs()
        self.info = {
            "kl": float(kl),
            "expressibility": float(express),
            "depth": int(depth),
            "n_twoq": int(n_twoq),
            "params": int(n_params),
            "blocks": list(self.blocks),
            "terminated": self.terminated
        }

        # store info for visualization in render()
        self.history.append({
            "P_C": hist_c,
            "P_Haar": hist_haar,
            "blocks": self.blocks.copy(),
            "reward": self.last_reward,
            "express": self.last_express,
            })

        return reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment to an empty circuit.

        Clears the current circuit architecture, resets internal counters, and
        initializes the observation vector corresponding to an empty ansatz.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility. If provided, reinitializes the
            internal random number generator.
        options : dict or None, optional
            Additional reset options (currently unused, included for
            Gymnasium compatibility).

        Returns
        -------
        observation : np.ndarray
            Initial observation vector describing an empty circuit, shape ``(7,)``.
        info : dict
            Empty dictionary provided for Gymnasium API compatibility.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.blocks = []
        self.current_step = 0
        self.last_express = None
        self.last_reward = 0.0
        self.info_last_hist = None
        obs = self._make_obs()
        self.history = []
        return obs, {}

    def step(self, action: int):
        """
        Execute one architecture-modification step by calling the get_reward method.

        Parameters
        ----------
        action : int
            Index of the selected architectural action.

        Returns
        -------
        observation : np.ndarray
            Updated observation vector summarizing circuit statistics,
            shape ``(7,)``.
        reward : float
            Reward value combining expressibility and architectural penalties.
        done : bool
            True if the episode ended due to termination by agent or truncation, 
            False otherwise.
        info : dict
            Diagnostic information including expressibility, KL divergence,
            depth, parameter count, current block sequence, and 
            terminated (true if agent explicitly terminated, false if episode ended due to max steps).
        """

        reward = self.get_reward(action)
        return self.obs, reward, self.done, self.info


    def render(self, save_path_without_extension=None, interval=800):
        """
        Render the expressibility optimization process as an animation.

        The animation shows:
        1. A histogram of circuit fidelity distribution compared to the
        Haar-random distribution.
        2. A block-diagram visualization of the evolving circuit architecture.

        Parameters
        ----------
        save_path_without_extension : str or None, optional
            Path (without file extension) to save the animation.
            If provided, the animation is saved using the configured writer
            (MP4 for FFmpeg or GIF for Pillow). If None, the animation is
            displayed interactively.
        interval : int, optional
            Delay between animation frames in milliseconds. Default is 800.

        Returns
        -------
        None
            This method produces a visualization but does not return a value.
        """

        if not hasattr(self, "history") or len(self.history) == 0:
            print("No history available for animation.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Init function
        def init():
            ax1.clear()
            ax2.clear()

        # Update function for each frame
        def update(frame):
            ax1.clear()
            ax2.clear()

            info = self.history[frame]

            # Fidelity distribution vs Haar
            hist_c = info["P_C"]
            hist_haar = info["P_Haar"]
            bins = len(hist_c)
            xs = np.linspace(0, 1, bins)

            ax1.bar(xs, hist_c, width=1.0/bins, alpha=0.6, label="Circuit Fidelity Dist.")
            ax1.plot(xs, hist_haar, "r-", lw=2, label="Haar Distribution")
            ax1.set_title(f"Expressibility (step {frame})")
            ax1.set_xlabel("Fidelity")
            ax1.set_ylabel("Density")
            ax1.legend()

            # Block diagram
            blocks = info["blocks"]
            ax2.set_xlim(0, max(1, len(blocks)))
            ax2.set_ylim(0, 1)
            ax2.axis("off")
            for i, b in enumerate(blocks):
                rect = Rectangle((i, 0.4), 0.9, 0.2, facecolor="skyblue", edgecolor="k")
                ax2.add_patch(rect)
                ax2.text(i+0.45, 0.5, b, ha="center", va="center", fontsize=8)
            ax2.set_title("Circuit Architecture Blocks")

            reward = info["reward"]
            express = info["express"]
            plt.suptitle(f"Reward={reward:.3f}, Expressibility={express:.3f}")

        anim = FuncAnimation(fig, update, frames=len(self.history),
                            init_func=init, interval=interval, repeat=False)

        if save_path_without_extension:
            anim.save(f"{save_path_without_extension}.{self.render_extension}", writer=self.writer)
        else:
            plt.show()

    def _make_obs(self):
        """
        Construct the current observation vector.

        The observation encodes summary statistics of the circuit architecture
        and training progress.

        Returns
        -------
        np.ndarray
            Observation vector of shape ``(7,)`` containing:
            ``[depth, n_blocks, n_twoq, n_params, ent_density, last_express, steps_left]``.
        """
        depth, n_blocks, n_twoq, n_params, ent_density = self._arch_stats()
        last_ex = self.last_express if self.last_express is not None else 0.0
        steps_left = max(self.max_steps - self.current_step, 0)
        vec = np.array([
            depth, n_blocks, n_twoq, n_params, int(1e3 * ent_density), last_ex, steps_left
        ], dtype=np.float32)
        return vec

    def _arch_stats(self) -> Tuple[int, int, int, int, float]:
        """
        Compute architecture statistics for the current circuit.

        Returns
        -------
        depth : int
            Total circuit depth.
        n_blocks : int
            Number of blocks in the circuit.
        n_twoq : int
            Total number of two-qubit gates.
        n_params : int
            Number of trainable rotation parameters.
        ent_density : float
            Normalized entangling density of the circuit.
        """
        depth = 0
        n_twoq = 0 # number of two-qubit gates
        n_params = 0
        for b in self.blocks:
            if b == "RotX":
                depth += 1
                n_params += 1 * self.n_qubits
            elif b == "RotY":
                depth += 1
                n_params += 1 * self.n_qubits
            elif b == "RotZ":
                depth += 1
                n_params += 1 * self.n_qubits
            elif b == "RotXYZ":
                depth += 1
                n_params += 3 * self.n_qubits
            elif b == "EntRingCNOT":
                depth += 1
                n_twoq += self.n_qubits
            elif b == "EntLadderCZ":
                depth += 1
                n_twoq += self.n_qubits - 1
            elif b == "EntAllToAllISWAP":
                depth += 1
                n_twoq += self.n_qubits * (self.n_qubits - 1) // 2
        n_blocks = len(self.blocks)

        # maximum number of possible two-qubit connections
        max_edges = self.n_qubits * (self.n_qubits - 1) / 2

        # entangling density
        ent_density = (n_twoq / max(1, n_blocks)) / max(1.0, max_edges)
        return depth, n_blocks, n_twoq, n_params, float(ent_density)

    def _circuit(self, thetas=None):
        """
        Construct and execute the quantum circuit for given parameters.

        The circuit structure is determined by the current block sequence.
        If parameters are not provided, they are sampled randomly.

        Parameters
        ----------
        thetas : np.ndarray or None, optional
            Vector of rotation parameters. If None, parameters are sampled
            from a standard normal distribution.

        Returns
        -------
        np.ndarray
            Statevector of the constructed quantum circuit.
        """
        if thetas is None:
            thetas = self._rng.standard_normal(self._count_rot_params())
        idx = 0
        for b in self.blocks:
            if b == "RotX":
                for w in range(self.n_qubits):
                    qml.RX(thetas[idx], wires=w); idx += 1
            elif b=="RotY":
                for w in range(self.n_qubits):
                    qml.RY(thetas[idx], wires=w); idx += 1
            elif b == "RotZ":
                for w in range(self.n_qubits):
                    qml.RZ(thetas[idx], wires=w); idx += 1
            elif b == "EntRingCNOT":
                for w in range(self.n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
            elif b == "EntLadderCZ":
                for w in range(self.n_qubits - 1):
                    qml.CZ(wires=[w, w + 1])
            elif b == "EntAllToAllISWAP":
                if not self.allow_all_to_all:
                    continue
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.ISWAP(wires=[i, j])
        return qml.state()

    def _count_rot_params(self) -> int:
        """
        Count the number of trainable rotation parameters in the circuit.

        Returns
        -------
        int
            Total number of variational parameters required by the current
            block sequence.
        """
        count = 0
        for b in self.blocks:
            if b=="RotX":
                count += self.n_qubits
            elif b=="RotY":
                count += self.n_qubits
            elif b=="RotZ":
                count += self.n_qubits
            elif b=="RotXYZ":
                count += (3 * self.n_qubits)
        return count

    def _expressibility(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Estimate the expressibility of the current circuit architecture.

        Expressibility is evaluated by comparing the fidelity distribution
        of randomly sampled circuit states against the Haar-random
        distribution.

        Returns
        -------
        expressibility : float
            Expressibility score defined as the negative KL divergence.
        kl : float
            Kullback–Leibler divergence between circuit and Haar distributions.
        hist_c : np.ndarray
            Histogram of circuit fidelity distribution.
        p_haar : np.ndarray
            Haar-random fidelity distribution.
        """
        n_pairs = max(2, self.n_pairs_eval)
        fidelities = np.empty(n_pairs, dtype=np.float64)
        n_params = self._count_rot_params()

        for k in range(n_pairs):

            # generate two random param vectors theta1, theta2
            theta1 = self._rng.uniform(0, 2*np.pi, n_params) if n_params else None
            theta2 = self._rng.uniform(0, 2*np.pi, n_params) if n_params else None
            
            # get output states
            psi = self.qnode(theta1)
            phi = self.qnode(theta2)

            # calculate fidelity of the output states
            fid = np.abs(np.vdot(psi, phi)) ** 2
            fidelities[k] = fid.real

        # generate historgram of all the circuit fidelities calculated
        hist_c, edges = np.histogram(fidelities, bins=self.bins, range=(0.0, 1.0), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dx = edges[1] - edges[0]

        # calculate Haar random distribution of fidelities
        # This is the ideal distribution for a maximally expressive ansatz.
        D = self.D # D = 2**num_qubits
        p_haar = (D - 1) * np.power(1.0 - centers, D - 2)
        p_haar = p_haar / (p_haar.sum() * dx + 1e-12) # normalize

        # calculate KL divergence
        # for kl==0, ansatz distribution perfectly matches Haar distribution
        # for kl==1, ansatz distribution is way of the Haar distribution (This is super bad)
        eps = 1e-12
        kl = float(np.sum(hist_c * np.log((hist_c + eps) / (p_haar + eps))) * dx)
        
        # We generally express expressibility as higher the better
        # for express==0, ansatz distribution perfectly matches Haar distribution
        # for express==-1, ansatz distribution is way off the Haar distribution (This is super bad)
        express = -kl
        return express, kl, hist_c, p_haar

    def action_meanings(self):
        """
        Return a mapping from action indices to action names.

        Returns
        -------
        dict
            Dictionary mapping integer action indices to human-readable
            architectural action names.
        """
        return {i: n for i, n in enumerate(self.ACTION_NAMES)}


