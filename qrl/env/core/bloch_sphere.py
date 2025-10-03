'''
Implementation of BlochSphereV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''


from gymnasium import spaces
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .base__ import QuantumEnv
from .utils import GATES, RX, RY, RZ


class BlochSphereV0(QuantumEnv):
    """
    ## Description

    `BlochSphereV0` is a single-qubit quantum environment intended for teaching, visualization, and 
    simple RL-style control experiments. It is based on the `QuantumEnv` base class. The environment represents the qubit state on the Bloch sphere and 
    provides a discrete set of quantum gates (Clifford + common rotations) as actions. 
    
    The agent's goal is to steer the qubit from the fixed initial state `|0\rangle` to a specified target 
    state (by default `|+\rangle`) within a minimum number of steps.

    As seen in the **Render** (see `results\core`): a semi-transparent Bloch sphere with labeled poles and 
    equatorial points. The green arrow shows the target Bloch vector while the red arrow shows the evolving 
    state (prediction) across steps. The sphere shows solid X, Y, Z axes and circular outlines for the XY, 
    XZ, and YZ planes for visual reference.

    ## Action Space

    The action space is discrete. 
    
    Each action applies a unitary gate to the current single-qubit state (left-multiplication by its 2x2 
    unitary matrix). Actions are deterministic and map directly to a fixed gate or rotation.

    | Num | Action     | Description              |
    | --- | ---------- | ------------------------ |
    | 0   | `H`        | Hadamard gate            |
    | 1   | `X`        | Pauli-X (NOT)            |
    | 2   | `Y`        | Pauli-Y                  |
    | 3   | `Z`        | Pauli-Z                  |
    | 4   | `S`        | Phase gate (S)           |
    | 5   | `SDG`      | S† (S dagger)            |
    | 6   | `T`        | T gate                   |
    | 7   | `TDG`      | T† (T dagger)            |
    | 8   | `RX_pi_2`  | Rotation about X by +π/2 |
    | 9   | `RX_pi_4`  | Rotation about X by +π/4 |
    | 10  | `RX_-pi_4` | Rotation about X by -π/4 |
    | 11  | `RY_pi_2`  | Rotation about Y by +π/2 |
    | 12  | `RY_pi_4`  | Rotation about Y by +π/4 |
    | 13  | `RY_-pi_4` | Rotation about Y by -π/4 |
    | 14  | `RZ_pi_2`  | Rotation about Z by +π/2 |
    | 15  | `RZ_pi_4`  | Rotation about Z by +π/4 |
    | 16  | `RZ_-pi_4` | Rotation about Z by -π/4 |

    The `action_space` is a `gymnasium.spaces.Discrete(len(actions))` where the integer index selects the action above.

    ## Observation Space

    The observation is a 3-dimensional `ndarray` corresponding to the Bloch vector `(x, y, z)` of the current pure qubit state. 
    Shape: `(3,)`, dtype `float32`, with each component bounded in `[-1, 1]`.

    | Num | Observation component | Min | Max | Meaning                                 |
    | --- | --------------------- | --: | --: | --------------------------------------- |
    | 0   | `x`                   |  -1 |   1 | 2 Re(ρ₀₁) — X component of Bloch vector |
    | 1   | `y`                   |  -1 |   1 | 2 Im(ρ₁₀) — Y component of Bloch vector |
    | 2   | `z`                   |  -1 |   1 | ρ₀₀ - ρ₁₁ — Z component of Bloch vector |

    Notes:

    * Internally the environment stores the statevector `|ψ⟩` (complex two-component vector) and converts to the Bloch vector for observations using the density matrix `ρ = |ψ⟩⟨ψ|`.

    ## Rewards

    Reward is defined as the quantum state fidelity (squared overlap) between the current state and the target state:

    ```
    reward = |⟨target | state⟩|^2
    ```

    This produces a continuous reward in `[0, 1]`. 
    
    Typical usage in episodic RL: treat higher reward as progress; the environment flags `done` when reward 
    exceeds `reward_tolerance` (default `0.99`). The environment does not subtract step penalties by default,
    but users can wrap or modify rewards to encourage shorter trajectories.

    ## Starting State

    On `reset()` the environment:

    * sets `steps = 0`;
    * initializes the qubit to `|0⟩` (statevector `[1, 0]`);
    * sets `history` to contain the initial Bloch vector and metadata `('None','None')`.

    By default the target state is `|+⟩ = (|0⟩ + |1⟩)/√2` (unless a different `target_state` is passed to the constructor). The code example in this class shows the `target_state` argument available in `__init__` so users can choose a different target on env construction.

    ## Episode End

    An episode terminates (i.e., `done=True`) when either:

    1. **Success / Termination**: The fidelity `|⟨target|state⟩|^2` is strictly greater than `reward_tolerance` (default `0.99`).
    2. **Truncation**: The number of steps reaches `max_steps` (default `20`).

    The `step()` method returns `(observation, reward, done, info)` matching `gymnasium.Env` conventions. `info` is currently empty but `history` is stored on the environment instance for inspection and rendering.

    ## Render

    `render(save_path=None, interval=800)` produces a Matplotlib 3D animation visualizing the Bloch sphere, target vector (green), and the recorded state trajectory (red vector updated per frame). 
    If `save_path` is provided, the animation is saved (FFmpeg writer is used); otherwise the animation is shown with `plt.show()`.

    Rendering details:

    * Sphere is drawn once as a translucent mesh.
    * X/Y/Z axes are drawn as solid lines and equatorial/circular outlines for XY/XZ/YZ planes are shown.
    * Labels placed for canonical states: `|0⟩`, `|1⟩`, `|+⟩`, `|−⟩`, `|+i⟩`, `|-i⟩`.
    * The animation title displays `Step`, `Reward` (rounded), and `Gate` applied at that step.

    ## Arguments (Constructor & Reset Options)

    Constructor signature (as implemented):

    ```python
    BlochSphereV0(target_state, max_steps=20, reward_tolerance=0.99,ffmpeg=False)
    ```

    * `target_state`: complex 2-vector specifying the target pure state. If `None`, defaults to `|+⟩`.
    * `max_steps`: maximum number of actions per episode (truncation threshold).
    * `reward_tolerance`: fidelity threshold above which the episode is marked successful.
    * `ffmpeg`: if `True`, uses FFmpeg for saving animations; otherwise uses Pillow (GIF).
    
    `reset()` currently ignores an `options` dict for seeding or custom initialization; this can be extended to allow randomized initial states or alternative targets.

    ### Example

    ```python
    import numpy as np
    from qrl.env import BlochSphereV0

    # target |+> state
    target = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    env = BlochSphereV0(target_state=target, max_steps=20, reward_tolerance=0.99)
    obs, _ = env.reset()

    for _ in range(env.max_steps):
        a = env.action_space.sample()
        obs, reward, done, _ = env.step(a)
        if done:
            break

    env.render()  # or env.render(save_path='results/bloch.mp4')
    ```

    ## Implementation Notes & Extensions

    * The environment expects unitary matrices to be available via a `GATES` dict for named Clifford gates and helper functions `RX(theta)`, `RY(theta)`, `RZ(theta)` for parameterized rotations. Those helpers should return 2×2 NumPy/Pennylane-arrays that multiply the statevector.
    * Currently the state is pure and represented as a statevector. To support mixed states or noise channels, one could change internal storage to density matrices and adapt `_state_to_bloch` accordingly.
    * Reward shaping: to encourage shorter trajectories, add a step penalty (e.g., `-0.01`) or give a sparse success reward on reaching the target.
    * Observation augmentation: include the current step number, recent gate applied, or the fidelity as extra observation channels if training agents that benefit from that information.

    ## Version History

    * **v0**: Initial design and implementation. Single-qubit pure-state environment with fixed initial state `|0⟩`, discrete gate set, fidelity reward, Matplotlib-based Bloch sphere renderer, and history tracking.

    ## References (Suggested Reading)

    * Bloch sphere — standard geometric representation for a qubit.
    * Nielsen, M. A., & Chuang, I. L., *Quantum Computation and Quantum Information* (for unitary gate definitions and single-qubit geometry).

    """

    def __init__(self, target_state, max_steps=20, reward_tolerance=0.99, ffmpeg=False):
        super().__init__()
        self.max_steps = max_steps
        self.target_state = target_state
        self.state = np.array([1, 0], dtype=complex)  # Initial State -> |0>
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension = "mp4" if ffmpeg else "gif"

        # Bloch vector (x, y, z)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Discrete action space
        self.actions = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
                        "RX_pi_2", "RX_pi_4", "RX_-pi_4",
                        "RY_pi_2", "RY_pi_4", "RY_-pi_4",
                        "RZ_pi_2", "RZ_pi_4", "RZ_-pi_4"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_tolerance = reward_tolerance
        if self.reward_tolerance <= 0 or self.reward_tolerance > 1:
            raise ValueError("reward_tolerance must be in (0, 1]")
        self.history = []
        self.steps = 0

    def _state_to_bloch(self, state):
        rho = np.outer(state, np.conj(state))
        x = 2*np.real(rho[0,1])
        y = 2*np.imag(rho[1,0])
        z = np.real(rho[0,0] - rho[1,1])
        return np.array([x, y, z], dtype=np.float32)

    def reset(self):
        self.steps = 0
        self.state = np.array([1, 0], dtype=complex)  # |0>
        self.history = [(self._state_to_bloch(self.state),'None','None')]

        # Default target state (|+>)
        self.target = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

        return self._state_to_bloch(self.state), {}

    def step(self, action):
        gate = self.actions[action]
        if gate in GATES:
            U = GATES[gate]
        elif "RX" in gate:
            U = RX(eval(gate.split("_")[1].replace("pi", "np.pi")))
        elif "RY" in gate:
            U = RY(eval(gate.split("_")[1].replace("pi", "np.pi")))
        elif "RZ" in gate:
            U = RZ(eval(gate.split("_")[1].replace("pi", "np.pi")))
        
        self.state = U @ self.state  # evolve state

        new_obs = self._state_to_bloch(self.state)

        reward = np.abs(np.vdot(self.target_state, self.state))**2
        self.history.append((new_obs, round(reward, 3), gate))
        self.steps += 1
        done = reward > self.reward_tolerance or self.steps >= self.max_steps

        return self._state_to_bloch(self.state), reward, done, {}
    

    def render(self,save_path_without_extension=None, interval=800):
        """
        history: list of bloch vectors for each step
        target_state: 3D bloch vector
        """
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=-60)

        # Add Qiskit-style Bloch sphere labels
        ax.text(0, 0, 1.1, r'$|0\rangle$', fontsize=12, color='black')
        ax.text(0, 0, -1.2, r'$|1\rangle$', fontsize=12, color='black')

        ax.text(0, 1.2, 0, r'$|+i\rangle$', fontsize=12, color='black')
        ax.text(0, -1.4, 0, r'$|-i\rangle$', fontsize=12, color='black')

        ax.text(1.2, 0, 0, r'$|+\rangle$', fontsize=12, color='black')
        ax.text(-1.4, 0, 0, r'$|-\rangle$', fontsize=12, color='black')


        # Sphere (draw once)
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.5, edgecolor='gray', linewidth=0.1)
        
        # Solid lines for X, Y and Z axes
        ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=1)
        ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=1)
        ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=1)

        # Solid Planes for XY, XZ, YZ   
        phi = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(phi), np.sin(phi), 0, color="black", linewidth=0.8)
        ax.plot(np.cos(phi), 0*np.cos(phi), np.sin(phi), color="black", linewidth=0.8)
        ax.plot(0*np.cos(phi), np.cos(phi), np.sin(phi), color="black", linewidth=0.8)


        # Axes limits
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        target_state = self._state_to_bloch(self.target)

        # Target arrow (static)
        target_arrow = ax.quiver(0, 0, 0, target_state[0], target_state[1], target_state[2],
                                color='green', linewidth=2, label='Target')

        # Dynamic prediction arrow (update each frame)
        pred_arrow = ax.quiver(0, 0, 0, self.history[0][0][0], self.history[0][0][1], self.history[0][0][2],
                            color='red', linewidth=2, label='Prediction')

        # Legend (only once)
        ax.legend()

        def update(frame):
            nonlocal pred_arrow
            # remove old arrow
            pred_arrow.remove()
            # draw new arrow
            pred_arrow = ax.quiver(0, 0, 0, self.history[frame][0][0], self.history[frame][0][1], self.history[frame][0][2],
                                color='red', linewidth=2)
            ax.set_title(f"Step {frame} | Reward={self.history[frame][1]} | Gate={self.history[frame][2]}")

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=interval, repeat=False)

        if save_path_without_extension:
            ani.save(f"{save_path_without_extension}.{self.render_extension}", writer=self.writer)
        else:
            plt.show()



