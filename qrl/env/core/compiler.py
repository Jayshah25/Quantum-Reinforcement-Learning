'''
Implementation of CompilerV0 environment
Author: Jay Shah (@Jayshah25)
License: Apache-2.0
'''


from gymnasium import spaces
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import shutil
from .utils import GATES, RX, RY, RZ 
from .base__ import QuantumEnv

class CompilerV0(QuantumEnv):
    """
    Single-qubit quantum gate compilation environment.

    ``CompilerV0`` is a ``gymnasium.Env``-compatible environment that models the
    problem of compiling a target single-qubit unitary using a fixed, discrete
    gate set. The agent incrementally applies quantum gates to build a circuit
    whose resulting unitary approximates a given target operation in SU(2).

    At each step, the agent selects a gate action that left-multiplies the current
    circuit unitary. The episode reward is based on the average gate fidelity
    between the current unitary and the target unitary, encouraging the agent to
    discover short, high-fidelity gate sequences.

    Key properties
    --------------
    - **Action space**: Discrete set of single-qubit gates (Clifford + rotations).
    - **Observation space**: Flattened real and imaginary parts of the current
      ``2×2`` unitary (shape ``(8,)``).
    - **Reward**: Average gate fidelity with respect to the target unitary.
    - **Termination**: Success when fidelity exceeds ``reward_tolerance`` or
      truncation at ``max_steps``.

    Rendering
    ---------
    The ``render()`` method visualizes the compilation process by displaying a
    heatmap of the magnitude of the difference matrix ``|U_target − U|`` over time,
    annotated with the current step, last applied gate, and reward.

    Input Parameters
    ----------
    target : np.ndarray
        Target ``2×2`` unitary matrix in SU(2) to compile towards.
    max_steps : int
        Maximum number of gate applications per episode.
    reward_tolerance : float
        Fidelity threshold for early termination.
    ffmpeg : bool
        Whether to use FFmpeg when saving animations.

    See Also
    --------
    :doc:`tutorials/compiler`
        Step-by-step tutorial on compiling SU(2) unitaries using ``CompilerV0``.
    
    """
    def __init__(self, target_unitary, max_steps=30, reward_tolerance=0.98, ffmpeg=False):
        super().__init__()
        self.max_steps = max_steps
        self.target_unitary = target_unitary  # target is a 2x2 unitary matrix

        assert self.target_unitary.shape == (2, 2), "Target unitary must be a 2x2 matrix."
        assert np.issubdtype(self.target_unitary.dtype, np.complexfloating), "Target matrix must be complex-valued."
        assert np.allclose(self.target_unitary.conj().T @ self.target_unitary, np.eye(2, dtype=complex)), "Target matrix must be unitary."

        # Observation: real+imag flattened 2x2 unitary = 8 floats
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        self.actions = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
                        "RX_pi_2", "RX_pi_4", "RY_pi_2", "RY_pi_4", "RZ_pi_2", "RZ_pi_4"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.history = []

        if reward_tolerance < 0 or reward_tolerance > 1:
            raise ValueError("reward_tolerance must be between 0 and 1")
        self.reward_tolerance = reward_tolerance
        self.steps = 0
        self.U = np.eye(2, dtype=complex)
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension = "mp4" if ffmpeg else "gif"
        if ffmpeg==True and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found on system. Please install ffmpeg or set ffmpeg=False")


    def _unitary_to_obs(self, U):
        """
        Convert a 2×2 unitary matrix into a flat observation vector.

        The unitary is represented by concatenating the flattened real and
        imaginary parts of the matrix.

        Parameters
        ----------
        U : np.ndarray
            Complex ``2×2`` unitary matrix representing the current circuit.

        Returns
        -------
        np.ndarray
            Flattened observation vector of shape ``(8,)`` containing
            ``[Re(U).flatten(), Im(U).flatten()]`` with dtype ``float32``.
        """
        return np.concatenate([U.real.flatten(), U.imag.flatten()]).astype(np.float32)

    def reset(self):
        """
        Reset the environment to the initial compilation state.

        The circuit unitary is reset to the identity matrix, the step counter
        is cleared, and the history buffer is reinitialized.

        Returns
        -------
        observation : np.ndarray
            Flattened observation corresponding to the identity unitary,
            shape ``(8,)``.
        info : dict
            Empty dictionary provided for compatibility with the Gymnasium API.
        """
        self.steps = 0
        self.U = np.eye(2, dtype=complex)
        
        # Random target unitary: sample U3(θ, φ, λ)
        # theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
        # self.target_unitary = (RZ(phi) @ RY(theta) @ RZ(lam))  # general SU(2)
        self.history = [(self.U, 'None', 'None')]
        return self._unitary_to_obs(self.U), {}
    
    def get_reward(self, action):
        """
        Apply a quantum gate action and compute the compilation reward.

        This method left-multiplies the current circuit unitary by the unitary
        corresponding to the selected action and evaluates the average gate
        fidelity with respect to the target unitary.

        Parameters
        ----------
        action : int
            Index of the selected action in ``self.actions``.

        Returns
        -------
        float
            Average gate fidelity between the current unitary and the target
            unitary, defined as
            ``0.5 * |Tr(U_target† · U)|`` for a single-qubit system.
        """
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
        reward = 0.5 * np.abs(np.trace(np.conj(self.target_unitary.T) @ self.U))
        return reward

    def step(self, action):
        """
        Execute one compilation step.

        Applies the selected gate, updates the internal circuit unitary and
        history, computes the reward, and checks termination conditions.

        Parameters
        ----------
        action : int
            Index of the selected action in ``self.actions``.

        Returns
        -------
        observation : np.ndarray
            Updated flattened unitary observation, shape ``(8,)``.
        reward : float
            Average gate fidelity after applying the action.
        done : bool
            True if the episode has terminated due to reaching the fidelity
            threshold or the maximum number of steps.
        info : dict
            Empty dictionary provided for compatibility with the Gymnasium API.
        """
        gate = self.actions[action]
        reward = self.get_reward(action)
        self.steps += 1
        self.history.append((self.U, gate, round(reward, 3)))
        done = reward > self.reward_tolerance or self.steps >= self.max_steps

        return self._unitary_to_obs(self.U), reward, done, {}

    def render(self, save_path_without_extension=None, interval=800):
        """
        Render the compilation process as an animation of the difference matrix.

        The visualization shows the magnitude of the element-wise difference
        ``|U_target - U|`` as a heatmap that evolves over time, along with
        annotations indicating the current step, applied action, and reward.

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

        fig, ax = plt.subplots(figsize=(5, 5))

        # Initial difference
        diff = np.abs(self.target_unitary - self.history[0][0])
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Target - Prediction|")

        def update(step):
            # Compute difference matrix
            diff = np.abs(self.target_unitary - self.history[step][0])
            im.set_array(diff)

            # Update title with fidelity
            ax.set_title(f"Step {step} | Action: {self.history[step][1]} | Reward={self.history[step][2]}")
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), interval=interval, blit=False, repeat=False
        )

        if save_path_without_extension:
            ani.save(f"{save_path_without_extension}.{self.render_extension}", writer=self.writer)
        else:
            plt.show()
