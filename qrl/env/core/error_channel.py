'''
Implementation of ErrorChannelV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''

from gymnasium import spaces
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from base__ import QuantumEnv


class ErrorChannelV0(QuantumEnv):
    '''
    ## Description

    The **ErrorChannelV0** environment simulates a **multi-qubit noisy quantum system** where different qubits
    may be subject to different error channels. It is based on the `QuantumEnv` base class. The agent's task is to identify and apply corrective
    single-qubit gates to mitigate the effects of these noise processes and restore the system state
    to the computational basis state **|0...0⟩**.

    This environment models the challenge of **quantum error mitigation/correction** in the presence of:
    - **Bit-flip errors** (X errors),
    - **Phase-flip errors** (Z errors),
    - **Amplitude damping** (energy relaxation).

    The agent interacts by applying Pauli gates (`X`, `Y`, or `Z`) to specific qubits. Rewards are based
    on the probability that the corrected quantum state has returned to the target basis state.

    The environment includes a rendering mode that provides a **bar plot animation** comparing the noisy,
    corrected, and ideal probability distributions, along with the evolving circuit diagram.

    ---

    ## Action Space

    The action space is a **MultiDiscrete( [3, n_qubits] )** space, meaning the agent chooses:

    1. A **gate**: {0: `X`, 1: `Y`, 2: `Z`}  
    2. A **target qubit**: index in `[0, n_qubits-1]`

    Example (for 3 qubits):  
    - `(0, 2)` → Apply `X` gate on qubit 2  
    - `(2, 0)` → Apply `Z` gate on qubit 0  

    ---

    ## Observation Space

    The observation is a probability distribution over all computational basis states
    of the `n_qubits` system:

    \[
    \text{obs} \in [0, 1]^{2^{n\_qubits}}
    \]

    with the constraint:

    \[
    \sum_i \text{obs}[i] = 1
    \]

    For example, with 3 qubits, the observation is a length-8 vector:

    | Index | Basis State | Probability |
    |-------|-------------|-------------|
    | 0     | `|000⟩`     | [0, 1]      |
    | 1     | `|001⟩`     | [0, 1]      |
    | ...   | ...         | ...         |
    | 7     | `|111⟩`     | [0, 1]      |

    ---

    ## Rewards

    The reward is the probability that the corrected quantum state has returned to the target state:

    \[
    reward = P(|0...0⟩)
    \]

    - Maximum reward = 1.0 (perfect correction).  
    - Minimum reward = 0.0 (completely corrupted).  

    The reward typically increases if the agent successfully counteracts the error channels.

    ---

    ## Starting State

    At the start of each episode:
    - A set of **faulty qubits** and their error types is defined.  
    (If not specified, a random qubit and error channel are chosen.)
    - Noise channels are applied to the initial state **|0...0⟩**.
    - The agent begins with no corrections applied.

    The first observation is the noisy probability distribution before any corrections.

    ---

    ## Episode End

    The episode ends if one of the following occurs:

    1. **Termination**:  
    The system reaches the maximum number of steps (`max_steps`, default=10).
    2. **Truncation**:  
    No special truncation — agent plays until max steps are reached.

    ---

    ## Rendering

    The rendering shows a **side-by-side animation** with:

    1. **Left panel**: Bar chart of probabilities for each basis state:  
    - Gray = Ideal noiseless target (|0...0⟩)  
    - Orange = Noisy distribution  
    - Blue = Corrected distribution  

    The chart title updates with **step number, chosen correction, and reward**.

    2. **Right panel**: A dynamically drawn ASCII-style **quantum circuit** reflecting
    the applied corrections.

    The animation can be displayed interactively or saved as an MP4 file.

    ---

    ## Arguments

    - **`n_qubits`** (`int`, default=3): Number of qubits in the system.  
    - **`faulty_errors`** (`dict`, optional): Mapping `{qubit_index: error_type}`, where `error_type` ∈ {`bitflip`, `phaseflip`, `amp_damp`}.  
    - **`noise_prob`** (`float`, default=0.15): Probability strength of the error channels.  
    - **`max_steps`** (`int`, default=10): Maximum number of agent actions per episode.  
    - **`seed`** (`int`, optional): Random seed for reproducibility.  

    Example:

    ```python
    >>> from qrl.env import ErrorChannelV0

    >>> env = ErrorChannelV0(
    ...     n_qubits=3,
    ...     faulty_errors={0: "bitflip", 2: "amp_damp"},
    ...     noise_prob=0.2,
    ...     max_steps=6,
    ...     seed=42,
    ... )

    >>> obs = env.reset()
    >>> obs.shape
    (8,)
    '''

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_qubits: int = 3,
        faulty_errors: dict = None,   # {qubit_idx: error_type, ...}
        noise_prob: float = 0.15,
        max_steps: int = 10,
        seed: int = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_prob = float(noise_prob)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.GATE_ID2NAME = {0: "X", 1: "Y", 2: "Z"}


        # Default: one random faulty qubit with random error type
        if faulty_errors is None:
            q = int(self.rng.integers(0, n_qubits))
            et = self.rng.choice(["bitflip", "phaseflip", "amp_damp"])
            faulty_errors = dict(zip([q], [et]))
        self.faulty_errors = faulty_errors

        # Device
        self.dev = qml.device("default.mixed", wires=n_qubits)

        # Action = (gate, qubit)
        self.action_space = spaces.MultiDiscrete([3, n_qubits])

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

    def _apply_noise(self):
        for q, etype in self.faulty_errors.items():
            if etype == "bitflip":
                qml.BitFlip(self.noise_prob, wires=q)
            elif etype == "phaseflip":
                qml.PhaseFlip(self.noise_prob, wires=q)
            elif etype == "amp_damp":
                qml.AmplitudeDamping(self.noise_prob, wires=q)

    def _apply_gate(self, gate_id, wire):
        if gate_id == 0:
            qml.PauliX(wires=wire)
        elif gate_id == 1:
            qml.PauliY(wires=wire)
        elif gate_id == 2:
            qml.PauliZ(wires=wire)

    def _build_qnodes(self):
        @qml.qnode(self.dev)
        def qnode_noisy():
            self._apply_noise()
            return qml.probs(wires=range(self.n_qubits))
        self.qnode_noisy = qnode_noisy

        @qml.qnode(self.dev)
        def qnode_corrected(k: int):
            self._apply_noise()
            for i in range(k):
                gid, w = self.corrections[i]
                self._apply_gate(gid, w)
            return qml.probs(wires=range(self.n_qubits))
        self.qnode_corrected = qnode_corrected

        # separate copy of corrected circuit for rendering
        @qml.qnode(self.dev)
        def qnode_draw(k: int):
            self._apply_noise()
            for i in range(k):
                gid, w = self.corrections[i]
                self._apply_gate(gid, w)
            return qml.probs(wires=range(self.n_qubits))  
        self.qnode_draw = qnode_draw


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.corrections.clear()
        self.history.clear()

        probs_noisy = self.qnode_noisy()
        probs_corrected = np.array(probs_noisy, copy=True)
        reward = float(probs_corrected[0])

        self.history.append(dict(
            step=0, gate=None, qubit=None,
            probs_noisy=probs_noisy,
            probs_corrected=probs_corrected,
            reward=reward,
        ))

        return probs_corrected.astype(np.float32)

    def step(self, action):
        gate_id, qubit_idx = int(action[0]), int(action[1])
        self.corrections.append((gate_id, qubit_idx))
        self.current_step += 1
        k = len(self.corrections)

        noisy = self.qnode_noisy()
        corrected = self.qnode_corrected(k)

        reward = float(corrected[0]) # Max can be 1, Min can be 0
        done = self.current_step >= self.max_steps

        self.history.append(dict(
            step=self.current_step,
            gate=self.GATE_ID2NAME[gate_id],
            qubit=qubit_idx,
            probs_noisy=noisy,
            probs_corrected=corrected,
            reward=reward,
        ))

        obs = corrected.astype(np.float32)
        info = {"faulty_errors": self.faulty_errors}
        return obs, reward, done, info

    def _basis_labels(self):
        return [f"|{i:0{self.n_qubits}b}⟩" for i in range(2**self.n_qubits)]

    def animate(self, save_path=None, interval_ms=600):
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
            if rec["gate"] is not None:
                title += f" | {rec['gate']} on q[{rec['qubit']}]"
            title += f" | reward={rec['reward']:.3f}"
            axL.set_title(title)

            draw_circuit_on_axis(min(rec["step"], len(self.corrections)))
            return (*bars_noisy, *bars_corr)

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=interval_ms, blit=False)

        if save_path:
            fps = max(1, int(1000/interval_ms))
            ani.save(save_path, writer=FFMpegWriter(fps=fps))
            plt.close(fig)
        else:
            plt.show()

    def render(self, save_path=None, interval=600):
        return self.animate(save_path=save_path, interval_ms=interval)