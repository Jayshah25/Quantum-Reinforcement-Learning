'''
Implementation of CompilerV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''


from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from .utils import GATES, RX, RY, RZ 
from .base__ import QuantumEnv

class CompilerV0(QuantumEnv):
    """
    ## Description

    The **CompilerV0** environment is designed to simulate the task of **quantum gate compilation** for a single-qubit system.
    It is based on the `QuantumEnv` base class. The agent's goal is to sequentially apply quantum gates to approximate a randomly chosen target **unitary operation**
    from the special unitary group SU(2). This mimics a quantum compilation problem where one attempts to
    rewrite a quantum operation in terms of a limited gate set.

    At each step, the agent applies one of several predefined single-qubit gates, evolving the current circuit unitary.
    The agent receives a reward proportional to the **fidelity** between the evolved unitary and the target unitary,
    and the episode terminates when the agent either reaches a sufficiently high fidelity or exhausts the maximum step limit.

    The environment includes a rendering mode that visualizes the **difference matrix** between the target and
    the current unitary as a heatmap evolving over time.

    ---

    ## Action Space

    The action space is **discrete**, where each action corresponds to applying a quantum gate
    from a fixed set of single-qubit operations:

    | Num | Action    | Description                        |
    |-----|-----------|------------------------------------|
    | 0   | `H`       | Hadamard gate                      |
    | 1   | `X`       | Pauli-X gate                       |
    | 2   | `Y`       | Pauli-Y gate                       |
    | 3   | `Z`       | Pauli-Z gate                       |
    | 4   | `S`       | Phase gate                         |
    | 5   | `SDG`     | Conjugate transpose of Phase gate  |
    | 6   | `T`       | π/8 gate                           |
    | 7   | `TDG`     | Conjugate transpose of π/8 gate    |
    | 8   | `RX_pi_2` | X-axis rotation by π/2             |
    | 9   | `RX_pi_4` | X-axis rotation by π/4             |
    | 10  | `RY_pi_2` | Y-axis rotation by π/2             |
    | 11  | `RY_pi_4` | Y-axis rotation by π/4             |
    | 12  | `RZ_pi_2` | Z-axis rotation by π/2             |
    | 13  | `RZ_pi_4` | Z-axis rotation by π/4             |

    ---

    ## Observation Space

    The observation is a flattened representation of the current unitary matrix, expressed in terms
    of its **real and imaginary parts**. This results in an 8-dimensional vector:

    | Num | Observation Component | Range   |
    |-----|------------------------|---------|
    | 0-3 | Real part of unitary   | [-1, 1] |
    | 4-7 | Imag part of unitary   | [-1, 1] |

    This encodes the full \(2 \times 2\) complex unitary matrix.

    ---

    ## Rewards

    The reward is based on the **average gate fidelity** between the target unitary  U_{target}
    and the current unitary U. Specifically:

    \[
    reward = \frac{1}{2} \left| \mathrm{Tr}(U_{target}^\dagger U) \right|
    \]

    - A higher reward indicates closer alignment with the target unitary.
    - The episode terminates early if the reward exceeds `reward_tolerance` (default: 0.98).

    ---

    ## Starting State

    At the start of each episode:
    - The circuit unitary is initialized as the **identity matrix** \( I \).
    - The target unitary is specified by the user at initialization.  
    (By default, this can be drawn from a random **U3(θ, φ, λ)** decomposition in SU(2).)

    The initial observation corresponds to the identity matrix.

    ---

    ## Episode End

    The episode ends if one of the following occurs:

    1. **Termination**:  
    The fidelity between the current and target unitary exceeds the reward tolerance (`reward > 0.98` by default).
    2. **Truncation**:  
    The number of steps exceeds the maximum episode length (`max_steps`, default: 30).

    ---

    ## Rendering

    The environment supports visualization of the compilation process:

    - A heatmap is drawn showing the **magnitude of the difference matrix**:
    \[
    |U_{target} - U|
    \]
    at each step.
    - The heatmap updates dynamically, and the plot title displays the **step number, last applied gate, and reward**.

    The animation can be saved as an MP4 file or displayed interactively.

    ---

    ## Arguments

    - **`target`** (`np.ndarray`): The target \(2 \times 2\) unitary matrix to compile towards.  
    - **`max_steps`** (`int`, default=30): Maximum number of steps per episode.  
    - **`reward_tolerance`** (`float`, default=0.98): Fidelity threshold for early termination.  

    Example:

    ```python
    >>> import numpy as np
    >>> from utils import RY, RZ
    >>> from qrl.env import CompilerV0

    >>> theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
    >>> target = (RZ(phi) @ RY(theta) @ RZ(lam))  # Random SU(2) unitary
    >>> env = CompilerV0(target=target)

    >>> obs, _ = env.reset()
    >>> obs.shape
    (8,)
    """
    def __init__(self, target, max_steps=30, reward_tolerance=0.98):
        super().__init__()
        self.max_steps = max_steps
        self.target = target  # target is a 2x2 unitary matrix
        
        # Observation: real+imag flattened 2x2 unitary = 8 floats
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        self.actions = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
                        "RX_pi_2", "RX_pi_4", "RY_pi_2", "RY_pi_4", "RZ_pi_2", "RZ_pi_4"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.history = []
        self.reward_tolerance = reward_tolerance

    def _unitary_to_obs(self, U):
        return np.concatenate([U.real.flatten(), U.imag.flatten()]).astype(np.float32)

    def reset(self):
        self.steps = 0
        self.U = np.eye(2, dtype=complex)
        
        # Random target unitary: sample U3(θ, φ, λ)
        # theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
        # self.target = (RZ(phi) @ RY(theta) @ RZ(lam))  # general SU(2)
        self.history = [(self.U, 'None', 'None')]
        return self._unitary_to_obs(self.U), {}

    def step(self, action):
        gate = self.actions[action]
        if gate in GATES:
            U_gate = GATES[gate]
        elif "RX" in gate:
            U_gate = RX(eval(gate.split("_")[1].replace("pi", "np.pi")))
        elif "RY" in gate:
            U_gate = RY(eval(gate.split("_")[1].replace("pi", "np.pi")))
        elif "RZ" in gate:
            U_gate = RZ(eval(gate.split("_")[1].replace("pi", "np.pi")))
        
        # Apply gate
        self.U = U_gate @ self.U
        
        # Fidelity: average gate fidelity for 1-qubit
        reward = 0.5 * np.abs(np.trace(np.conj(self.target.T) @ self.U))
        self.steps += 1
        self.history.append((self.U, gate, round(reward, 3)))
        done = reward > self.reward_tolerance or self.steps >= self.max_steps

        return self._unitary_to_obs(self.U), reward, done, {}

    def render(self, save_path=None, interval=800):
        """
        Render the episode as an animation of the difference matrix.
        Only shows |target - current| evolving across steps.
        """

        fig, ax = plt.subplots(figsize=(5, 5))

        # Initial difference
        diff = np.abs(self.target - self.history[0][0])
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Target - Prediction|")

        def update(step):
            # Compute difference matrix
            diff = np.abs(self.target - self.history[step][0])
            im.set_array(diff)

            # Update title with fidelity
            ax.set_title(f"Step {step} | Action: {self.history[step][1]} | Reward={self.history[step][2]}")
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), interval=interval, blit=False, repeat=False
        )

        if save_path:
            ani.save(save_path, writer="ffmpeg")
        else:
            plt.show()
