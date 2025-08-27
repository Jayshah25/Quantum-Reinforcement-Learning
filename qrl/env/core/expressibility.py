'''
Implementation of ExpressibilityV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''

from pennylane import numpy as np
from typing import List, Tuple, Optional, Dict
import pennylane as qml

from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from base__ import QuantumEnv


class ExpressibilityV0(QuantumEnv):
    """
    ## Description

    The **ExpressibilityV0** environment simulates the task of building **parameterized quantum circuits**
    that achieve **high expressibility**. It is based on the `QuantumEnv` base class. In variational quantum algorithms (VQAs), 
    the expressibility of an ansatz measures how well the circuit can explore the Hilbert space of possible quantum states.
    This environment challenges an agent to construct expressive circuits while balancing circuit depth
    and gate costs.

    At each step, the agent adds or removes blocks (rotation or entangling layers) or decides to
    terminate construction. The reward is based on how closely the circuit's fidelity distribution
    matches the Haar-random distribution (an idealized benchmark of maximum expressibility), penalized
    by circuit depth and two-qubit gate counts.

    The environment includes a rendering mode that animates:
    1. The evolving **fidelity distribution vs Haar distribution**, and  
    2. The **block diagram** of the constructed circuit architecture.  

    ---

    ## Action Space

    The action space is a **Discrete(8)** space, where each action corresponds to a modification of
    the circuit architecture:

    | ID | Action          | Description                                     |
    |----|-----------------|-------------------------------------------------|
    | 0  | `RotX`          | Add single-qubit RX rotations on all qubits     |
    | 1  | `RotY`          | Add single-qubit RY rotations on all qubits     |
    | 2  | `RotZ`          | Add single-qubit RZ rotations on all qubits     |
    | 3  | `RotXYZ`        | Add RX, RY, RZ rotations on all qubits          |
    | 4  | `EntRingCNOT`   | Add a ring of CNOT entanglers                   |
    | 5  | `EntLadderCZ`   | Add ladder-style CZ entanglers                  |
    | 6  | `RemoveLast`    | Remove the most recently added block            |
    | 7  | `Terminate`     | Stop circuit construction and end the episode   |

    ---

    ## Observation Space

    The observation is a **7-dimensional vector** summarizing circuit statistics:
    [depth, n_blocks, n_twoq, n_params, ent_density, last_express, steps_left]

    Where:
    - **`depth`**: Total circuit depth  
    - **`n_blocks`**: Number of blocks in the circuit  
    - **`n_twoq`**: Number of two-qubit gates  
    - **`n_params`**: Number of trainable rotation parameters  
    - **`ent_density`**: Entangling density relative to possible qubit connections  
    - **`last_express`**: Last computed expressibility score  
    - **`steps_left`**: Remaining steps before max_steps is reached  

    ---

    ## Rewards

    The reward encourages high expressibility while penalizing excessive depth and two-qubit usage:

    \[
    reward = - KL(P_C \,\|\, P_{Haar}) \;-\; \lambda_{depth}\cdot depth \;-\; \lambda_{2q}\cdot n_{twoq}
    \]

    Where:
    - **KL**: Kullback-Leibler divergence between circuit fidelity distribution `P_C`
    and Haar-random distribution `P_Haar`.  
    - **λ_depth, λ_2q**: Regularization weights for penalizing large depth and two-qubit gates.  
    - **Terminate bonus**: A small positive reward (`terminate_bonus`) is added if the agent terminates explicitly.  

    Interpretation:
    - High reward = Circuit closely mimics Haar distribution, shallow, and efficient.  
    - Low reward = Circuit deviates significantly from Haar distribution or becomes overly complex.  

    ---

    ## Starting State

    At the beginning of each episode:
    - The circuit is empty (`blocks = []`).
    - The observation corresponds to a circuit with zero depth, parameters, and entangling density.
    - The first reward is undefined until the agent applies at least one block.

    ---

    ## Episode End

    The episode ends under either condition:
    1. **Termination**: Agent selects `Terminate` action (ID=7).  
    2. **Truncation**: Maximum number of steps (`max_steps`, default=20) is reached.  

    ---

    ## Rendering

    The rendering shows a **two-panel animation**:

    1. **Left panel**:  
    - Histogram of circuit fidelity distribution vs. Haar-random distribution.  
    - The closer the two curves overlap, the more expressive the circuit is.  

    2. **Right panel**:  
    - Block diagram of the constructed circuit architecture, showing the sequence of blocks.  

    The figure title includes **reward and expressibility score** for the current step.  
    The animation can be displayed interactively or saved as an MP4 file.

    ---

    ## Arguments

    - **`n_qubits`** (`int`, default=4): Number of qubits.  
    - **`max_blocks`** (`int`, default=12): Maximum number of blocks allowed in the circuit.  
    - **`max_steps`** (`int`, default=20): Maximum steps per episode.  
    - **`n_pairs_eval`** (`int`, default=120): Number of random state pairs used to evaluate expressibility.  
    - **`bins`** (`int`, default=50): Number of histogram bins for fidelity distribution.  
    - **`lambda_depth`** (`float`, default=0.002): Penalty weight for depth.  
    - **`lambda_2q`** (`float`, default=0.002): Penalty weight for two-qubit gates.  
    - **`terminate_bonus`** (`float`, default=0.1): Bonus reward for explicit termination.  
    - **`device_name`** (`str`, default="default.qubit"): PennyLane device backend.  
    - **`seed`** (`int`, optional): Random seed for reproducibility.  
    - **`allow_all_to_all`** (`bool`, default=False): Allow inclusion of all-to-all entangling ISWAP blocks.  

    Example:

    ```python
    >>> from qrl.env import ExpressibilityV0

    >>> env = ExpressibilityV0(n_qubits=3, n_pairs_eval=60, bins=40, seed=7)
    >>> obs, _ = env.reset()
    >>> obs.shape
    (7,)
    """

    metadata = {"render_modes": ["human"], "name": "ExpressibilityV0"}



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
        self.qnode = qml.QNode(self._circuit, self.device)

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
            10_000, max_blocks, 10_000, 10_000, 1_000, 10.0, max_steps
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)

        self.current_step = 0
        self.last_express = None
        self.last_reward = 0.0
        self.info_last_hist = None
        self.history = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
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

        # check if the action is supported
        assert self.action_space.contains(action)
        done = False
        terminated = False

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
            done = True
            terminated = True

        # update current step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # get expressibility for the current circuit
        express, kl, hist_c, hist_haar = self._expressibility()
        self.last_express = express
        self.info_last_hist = {"P_C": hist_c, "P_Haar": hist_haar}

        depth, n_blocks, n_twoq, n_params, ent_density = self._arch_stats()

        reward = -kl - self.lambda_depth * depth - self.lambda_2q * n_twoq
        if terminated:
            reward += self.terminate_bonus

        self.last_reward = reward

        obs = self._make_obs()
        info = {
            "kl": float(kl),
            "expressibility": float(express),
            "depth": int(depth),
            "n_twoq": int(n_twoq),
            "params": int(n_params),
            "blocks": list(self.blocks),
        }

        # store info for visualization in render()
        self.history.append({
            "P_C": hist_c,
            "P_Haar": hist_haar,
            "blocks": self.blocks.copy(),
            "reward": self.last_reward,
            "express": self.last_express,
            })

        return obs, reward, done, False, info


    def render(self, save_path=None, interval=800):
        """
        Animation of Expressibility:
        1. Fidelity histogram vs Haar distribution per step.
        2. Block diagram evolution of circuit architecture.
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

        if save_path:
            anim.save(save_path, writer="ffmpeg")
        else:
            plt.show()

    def _make_obs(self):
        '''
        Returns the current observation vector [depth, n_blocks, n_twoq, n_params, ent_density, last_ex, steps_left], where
        depth is the circuit depth,
        n_blocks is the number of blocks in the circuit,
        n_twoq is the number of two-qubit gates,
        n_params is the number of rotational parameters,
        ent_density is the entangling density,
        last_ex is the last expressibility value,
        steps_left is the number of steps left with respect to the maximum number of steps.
        '''
        depth, n_blocks, n_twoq, n_params, ent_density = self._arch_stats()
        last_ex = self.last_express if self.last_express is not None else 0.0
        steps_left = max(self.max_steps - self.current_step, 0)
        vec = np.array([
            depth, n_blocks, n_twoq, n_params, int(1e3 * ent_density), last_ex, steps_left
        ], dtype=np.float32)
        return vec

    def _arch_stats(self) -> Tuple[int, int, int, int, float]:
        '''
        Compute architecture statistics for the current circuit.
        Returns depth, n_blocks (number of blocks in the circuit), 
        n_twoq (number of two qubit gates), 
        n_params (number of rotational parameters), ent_density (entangling density)
        '''
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
        '''represents the quantum circuit with the given parameters theta'''
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
        '''Count the number of rotational/trainable/variational parameters in the circuit.'''
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
        return {i: n for i, n in enumerate(self.ACTION_NAMES)}


