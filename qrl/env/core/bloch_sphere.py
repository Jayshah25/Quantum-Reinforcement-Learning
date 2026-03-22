'''
Implementation of BlochSphereV0 environment

Author: Jay Shah (@Jayshah25)

Contact: jay.shah@qrlqai.com

License: Apache-2.0
'''


from gymnasium import spaces
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import networkx as nx
import shutil
import warnings
from typing import Literal, Optional
from ._base import QuantumEnv
from .utils import GATES, RX, RY, RZ, STATE_LABELS, STATE_BLOCH, STATE_VECTORS, ACTION_NAMES, _TRANSITIONS, _GRAPH_POS


class BlochSphereV0(QuantumEnv):
    """
    Single-qubit Bloch sphere environment for reinforcement learning.

    ``BlochSphereV0`` is a ``gymnasium.Env``-compatible environment where an agent
    controls a single qubit via a discrete set of quantum gates. The qubit state is
    represented internally as a statevector and exposed to the agent as a 3D Bloch
    vector ``(x, y, z)``.

    The objective is to steer the qubit from the fixed initial state ``|0⟩`` to a
    target pure state (default ``|+⟩``) within a limited number of steps by applying
    unitary gate actions.

    Key details
    --------------
    - **Action space**: Discrete set of single-qubit gates (Clifford + common rotations).
    - **Observation space**: Bloch vector ``(x, y, z)``, each component in ``[-1, 1]``.
    - **Reward**: Fidelity ``|⟨target | state⟩|²`` in ``[0, 1]``.
    - **Termination**: Success when reward exceeds ``reward_tolerance`` or truncation
    at ``max_steps``.

    Rendering
    ---------
    The ``render()`` method visualizes the Bloch sphere and the agent’s trajectory,
    showing the current state and target state as arrows in 3D.

    Input Parameters
    ----------
    - **target_state**: Target pure state as a Numpy complex 2-vector, defaults to ``|+⟩``.
    - **max_steps**: Maximum number of steps per episode.
    - **reward_tolerance**: Fidelity threshold for successful termination.
    - **ffmpeg**: If set to True, animations are saved as mp4 videos, else as GIFs. Default is False.
    
    See Also
    --------
    :doc:`tutorials/bloch_sphere`

    """
    def __init__(self, target_state, max_steps=20, reward_tolerance=0.99, ffmpeg=False):
        super().__init__()
        self.max_steps = max_steps
        self.target_state = target_state
        self.state = np.array([1, 0], dtype=complex)  # Initial State -> |0>
        self.writer = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension = "mp4" if ffmpeg else "gif"

        if ffmpeg==True and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found on system. Please install ffmpeg or set ffmpeg=False")

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
        """
        Convert a single-qubit statevector to its Bloch vector representation.

        Parameters
        ----------
        state : np.ndarray
            Complex 2-element statevector |ψ⟩ representing a pure qubit state.

        Returns
        -------
        np.ndarray
            Bloch vector ``(x, y, z)`` as a float32 array of shape ``(3,)``,
            with each component in the range ``[-1, 1]``.
        """
        rho = np.outer(state, np.conj(state))
        x = 2*np.real(rho[0,1])
        y = 2*np.imag(rho[1,0])
        z = np.real(rho[0,0] - rho[1,1])
        return np.array([x, y, z], dtype=np.float32)

    def reset(self):
        """
        Reset the environment to the initial state.

        The qubit is initialized to the computational basis state |0⟩.
        Episode step count and history are cleared.

        Returns
        -------
        observation : np.ndarray
            Initial Bloch vector corresponding to |0⟩, shape ``(3,)``.
        info : dict
            Empty dictionary provided for compatibility with Gymnasium API.
        """
        self.steps = 0
        self.state = np.array([1, 0], dtype=complex)  # |0>
        self.history = [(self._state_to_bloch(self.state),'None','None')]

        # Default target state (|+>)
        self.target = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

        return self._state_to_bloch(self.state), {}
    
    def get_reward(self, action):
        """
        Apply a quantum gate action and compute the resulting reward.

        This method evolves the internal qubit state by applying the unitary
        corresponding to the selected action and evaluates the fidelity with
        respect to the target state.

        Parameters
        ----------
        action : int
            Index of the selected action in ``self.actions``.

        Returns
        -------
        float
            Fidelity between the current state and the target state, defined as
            ``|⟨target | state⟩|²`` and bounded in ``[0, 1]``.
        """
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

        reward = np.abs(np.vdot(self.target_state, self.state))**2

        return reward

    def step(self, action):
        """
        Execute one environment step.

        Applies the selected quantum gate, updates the internal state and history,
        computes the reward, and checks termination conditions.

        Parameters
        ----------
        action : int
            Index of the selected action in ``self.actions``.

        Returns
        -------
        observation : np.ndarray
            Updated Bloch vector of the qubit state, shape ``(3,)``.
        reward : float
            Fidelity-based reward after applying the action.
        done : bool
            True if the episode has terminated due to success or truncation.
        info : dict
            Empty dictionary provided for compatibility with Gymnasium API.
        """
        reward = self.get_reward(action)
        new_obs = self._state_to_bloch(self.state)
        gate = self.actions[action]
        self.history.append((new_obs, round(reward, 3), gate))
        self.steps += 1
        done = reward > self.reward_tolerance or self.steps >= self.max_steps

        return self._state_to_bloch(self.state), reward, done, {}
    

    def render(self, save_path_without_extension=None, interval=800):
        """
        Render the Bloch sphere trajectory as a 3D animation.

        The visualization shows:
        - A translucent Bloch sphere with labeled basis states,
        - The target Bloch vector (green, static),
        - The evolving qubit state trajectory (red, dynamic).

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


class BlochSphereV1(QuantumEnv):

    """
    Single-qubit Bloch sphere environment as a graph problem for reinforcement learning.

    ``BlochSphereV1`` is a ``gymnasium.Env`` compatible environment where an agent
    controls a single qubit via a discrete set of quantum gates. The qubit state is
    exposed to the agent as an integer index corresponding to the discrete states 
    |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩.

    The objective is to steer the qubit from the fixed starting initial state ``|0⟩`` to a
    user defined target pure state (default ``|+⟩``) within a limited number of steps by applying
    unitary gate actions.

    
    The environment is fully compatible with ``ValueIteration`` and ``QValueIteration`` from
    ``qrl.algorithms``.

    Key details
    --------------
    - **Action space**: Discrete set of single-qubit gates (H,X,Z,S).
    - **Observation space**: Integer index corresponding to the Discrete states |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩.
    - **Reward**: Fidelity ``|⟨target | state⟩|²`` in ``[0, 1]``.
    - **Termination**: Success when reward exceeds ``reward_tolerance`` or truncation
    at ``max_steps``.

    Parameters
    ----------
    - **target_state** : int, optional
        Target state index in [0, 5]. Defaults to 2 (|+⟩). The mapping is:
        0 → |0⟩, 1 → |1⟩, 2 → |+⟩, 3 → |-⟩, 4 → |+i⟩, 5 → |-i⟩.
    - **max_steps** : int, optional
        Maximum number of steps per episode. Default is 10.
    - **reward_tolerance** : float, optional
        Fidelity threshold for successful termination. Must be in (0, 1].
        Default is 0.99.
    - **ffmpeg** : bool, optional
        If True, animations are saved as MP4 via ffmpeg, else as GIFs.
        Default is False.

    Raises
    ------
    - **ValueError** : If ``target_state`` is not in [0, 5].
    - **ValueError** : If ``reward_tolerance`` is not in (0, 1].
    - **ValueError** : If ``ffmpeg=True`` but ffmpeg is not installed on the system.
    """
 
 
    def __init__(
        self,
        target_state: int = 2,
        max_steps: int = 10,
        reward_tolerance: float = 0.99,
        ffmpeg: bool = False,
    ) -> None:
        super().__init__()

        # define constants
        # self.STEP_PENALTY = 0.01
        # self.SUCCESS_BONUS = 1.0
 
        if not (0 <= target_state <= 5):
            raise ValueError("target_state must be an integer in [0, 5].")
        if not (0 < reward_tolerance <= 1):
            raise ValueError("reward_tolerance must be in (0, 1].")
        if ffmpeg and shutil.which("ffmpeg") is None:
            raise ValueError("ffmpeg not found. Install it or set ffmpeg=False.")
 
        self.target_state_index = target_state
        self.max_steps          = max_steps
        self.reward_tolerance   = reward_tolerance
        self.writer             = "ffmpeg" if ffmpeg else "pillow"
        self.render_extension   = "mp4"    if ffmpeg else "gif"
        self.fig_array_list     = []
        self.observation_space = spaces.Discrete(6)
        self.action_space      = spaces.Discrete(4)
 
        self._state_index: int  = 0
        self._statevector       = STATE_VECTORS[0].copy()
        self.steps: int         = 0
        self.history: list[int] = []   # sequence of state indices
        self.terminated: bool | None = None
        self.truncated: bool | None = None
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.

        The qubit is placed at state index 0 (|0⟩). Episode step count,
        history, and termination flags are cleared.

        Parameters
        ----------
        seed : int or None, optional
            Random seed passed to the base ``gymnasium.Env`` reset. Default is None.
        options : dict or None, optional
            Additional options passed to the base reset. Default is None.

        Returns
        -------
        observation : int
            Initial state index (always 0, corresponding to |0⟩).
        info : dict
            Dictionary containing ``fidelity``, ``gate`` (``"reset"``), and
            ``bloch_vector`` of the initial state.
        """        
        super().reset(seed=seed)
        self._state_index = 0
        self._statevector = STATE_VECTORS[0].copy()
        self.steps        = 0
        self.history      = [0]
        self.terminated   = False   
        self.truncated    = False   
        return 0, self._info()

    def get_reward(self):
        """
        Compute the reward for the current state and update termination flags.

        Evaluates the fidelity between the current statevector and the target
        statevector. Sets ``self.terminated`` if fidelity meets or exceeds
        ``reward_tolerance``, and ``self.truncated`` if the step limit is reached.

        Returns
        -------
        float
            1.0 if the current state matches the target within ``reward_tolerance``,
            0.0 otherwise.
        """        
        self.terminated = self._fidelity() >= self.reward_tolerance
        self.truncated  = self.steps >= self.max_steps
        return 1.0 if self.terminated else 0.0 

    def step(self, action: int):
        """
        Apply a gate action and advance the episode by one step.

        Applies the unitary gate corresponding to ``action`` to the current
        statevector, updates the discrete state index via the transition table,
        increments the step counter, and appends the new state to history.

        Parameters
        ----------
        action : int
            Index into ``ACTION_NAMES`` selecting the gate to apply.
            0 → H, 1 → X, 2 → Z, 3 → S.

        Returns
        -------
        observation : int
            New discrete state index after applying the gate.
        reward : float
            1.0 if the target is reached within tolerance, 0.0 otherwise.
        terminated : bool
            True if fidelity ≥ ``reward_tolerance``.
        truncated : bool
            True if ``steps`` ≥ ``max_steps``.
        info : dict
            Dictionary containing ``fidelity``, ``gate`` name applied, and
            ``bloch_vector`` of the resulting state.
        """
        gate_name         = ACTION_NAMES[action]
        U                 = GATES[gate_name]
        self._statevector = U @ self._statevector
        self._state_index = int(_TRANSITIONS[self._state_index, action])
        self.steps       += 1
        self.history.append(self._state_index)
 
        reward = self.get_reward()
 
        return self._state_index, float(reward), self.terminated, self.truncated, self._info(gate_name)
 
 
    def _fidelity(self) -> float:
        """
        Compute the fidelity between the current state and the target state.

        Fidelity is defined as ``|⟨target | state⟩|²``, bounded in ``[0, 1]``.
        A value of 1.0 indicates the current state is identical to the target.

        Returns
        -------
        float
            Fidelity between the current statevector and the target statevector,
            in the range ``[0, 1]``.
        """        
        target_sv = STATE_VECTORS[self.target_state_index]
        return float(np.abs(np.vdot(target_sv, self._statevector)) ** 2)
 
    def _info(self, gate: str = "reset") -> dict:
        """
        Construct the info dictionary for the current environment state.

        Parameters
        ----------
        gate : str, optional
            Name of the gate most recently applied. Defaults to ``"reset"``
            when called during environment initialization or reset.

        Returns
        -------
        dict
            A dictionary with three keys:
            - ``fidelity``    : float — current fidelity with the target state.
            - ``gate``        : str   — name of the gate applied in this step.
            - ``bloch_vector``: np.ndarray — Bloch vector ``(x, y, z)`` of the
            current state, shape ``(3,)``.
        """        
        return {
            "fidelity":     self._fidelity(),
            "gate":         gate,
            "state_index":  self._state_index,
        }
 
    # reneder graph

    def _render_graph(self, agent=None, show_true_dynamics: bool = True) -> None:
        """
        Draw the state-transition graph. When agent is provided, adds a second
        panel showing the agent's learned model:
 
          Left  — true environment dynamics
          Right — agent panel:
                    • Node color  : learned state value V(s) or max_a Q(s,a)
                                    on a warm colormap (high = warm, low = cool)
                    • Edge opacity: proportional to empirical visit count
                    • Bold edges  : greedy policy argmax_a Q(s,a)

        Ideally, the learnt target state should have minimal value function.
 
        Layout (both panels mirror the Bloch sphere):
 
                   |0⟩
              |-⟩       |+⟩
             |-i⟩       |+i⟩
                   |1⟩
        """
 
        BG      = "#1a1a1a"   # figure / axes background
        FG      = "#e0e0e0"   # titles, node labels, generic text
        EDGE_FG = "#444440"   # base edge color on dark bg
        LBL_FG  = "#999994"   # edge label color (base graph)
 
        has_agent  = agent is not None 
        if not has_agent:
            raise ValueError("No agent provided.")
        n_panels = (1 if show_true_dynamics else 0) + (1 if has_agent else 0)# number of panels: 1 for true dynamics, 1 for agent1 # number of panels: 1 for true dynamics, 1 for agent
        fig_w = 10 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 8), facecolor=BG) # figure and axes
        if n_panels == 1:
            axes = [axes]
        ax_true  = axes[0] if show_true_dynamics else None
        ax_agent = axes[-1] if has_agent  else None
        for _ax in axes:
            _ax.set_facecolor(BG) # set background color for each axis
 
        # shared graph structure of true environment dynamics and agent
        G = nx.MultiDiGraph() # MultiDiGraph is a directed graph with multiple edges between two nodes
        G.add_nodes_from(range(6)) # add nodes to the graph
        edge_gates: dict[tuple[int, int], list[str]] = {}
        for s in range(6):
            for a, name in enumerate(ACTION_NAMES):
                dst = int(_TRANSITIONS[s, a])
                edge_gates.setdefault((s, dst), []).append(name)
        for (s, d), gnames in edge_gates.items():
            G.add_edge(s, d, label=",".join(gnames))
 
        pos = _GRAPH_POS
 
        # arc midpoint helper to place text labels at the true arc midpoint for each edge
        _graph_center = np.mean(np.array(list(pos.values())), axis=0)
 
        def _arc_midpoint(src, dst, rad=0.10):
            """
            Return the (x, y) midpoint of the arc networkx draws.
 
            Regular edges: quadratic Bezier B(t=0.5) with control point
            offset perpendicularly from the edge midpoint by rad × |P2-P0|.
 
            Self-loops (src == dst): networkx draws a small tangent loop.
            We offset the label outward from the graph center so it sits
            on top of the visible loop, not on the node itself.
            """
            p0 = np.array(pos[src])
            p2 = np.array(pos[dst])
 
            # self-loop
            if src == dst:
                outward = p0 - _graph_center
                norm = np.linalg.norm(outward)
                outward = outward / norm if norm > 1e-9 else np.array([0.0, 1.0])
                return p0 + outward * 0.52
 
            # regular arc 
            mid    = (p0 + p2) / 2.0
            diff   = p2 - p0
            length = np.linalg.norm(diff)
            perp   = np.array([-diff[1], diff[0]]) / length
            pc     = mid + rad * length * perp
            return 0.25 * p0 + 0.5 * pc + 0.25 * p2
 
        def _label_edges(ax, edge_label_map, rad=0.10,
                         font_size=7, font_color=LBL_FG):
            """
            Place text labels at the true arc midpoint for each edge.
            edge_label_map: dict with keys that are either (src, dst) pairs
            or (src, dst, key) triples (networkx MultiDiGraph convention).
            """
            for edge_key, label in edge_label_map.items():
                src, dst = edge_key[0], edge_key[1]
                mx, my = _arc_midpoint(src, dst, rad)
                ax.text(
                    mx, my, label,
                    fontsize=font_size, color=font_color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=BG, ec="none", alpha=0.8),
                )
 
        # helper: draw one graph panel 
        def _draw_base(ax, node_colors, title):
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=11, pad=10, color=FG)
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=1600, node_color=node_colors,
                edgecolors="#888780", linewidths=1.2,
            )
            nx.draw_networkx_labels(
                G, pos, ax=ax, labels=STATE_LABELS, font_size=10, font_color="#111111",
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=EDGE_FG, width=0.8, arrowsize=12,
                arrowstyle="-|>", connectionstyle="arc3,rad=0.10",
                min_source_margin=30, min_target_margin=30,
            )
            _label_edges(
                ax,
                edge_label_map=nx.get_edge_attributes(G, "label"),
                font_size=7, font_color="#888780",
            )
 
        # LEFT PANEL: true dynamics (optional) 
        if show_true_dynamics and ax_true is not None:
            true_colors = []
            for n in range(6):
                if n == self._state_index and n == self.target_state_index:
                    true_colors.append("#F5C518")
                elif n == self._state_index:
                    true_colors.append("#E24B4A")
                elif n == self.target_state_index:
                    true_colors.append("#3B8BD4")
                else:
                    true_colors.append("#D3D1C7")
 
            _draw_base(ax_true, true_colors, "True environment dynamics")
 
            # trajectory overlay
            if len(self.history) > 1:
                traj_edges = list(zip(self.history[:-1], self.history[1:]))
                unique_traj = list(dict.fromkeys(traj_edges))                
                nx.draw_networkx_edges(
                    G, pos, ax=ax_true,
                    edgelist=unique_traj,
                    edge_color="#BA7517", width=2.8, arrowsize=18,
                    arrowstyle="-|>", connectionstyle="arc3,rad=0.10",
                    min_source_margin=30, min_target_margin=30,
                )
 
 
            # left legend
            left_legend = [
                mpatches.Patch(facecolor="#E24B4A", edgecolor="#888780", label="Current"),
                mpatches.Patch(facecolor="#3B8BD4", edgecolor="#888780", label="Target"),
                mpatches.Patch(facecolor="#F5C518", edgecolor="#888780", label="Current = target"),
                mpatches.Patch(facecolor="#D3D1C7", edgecolor="#888780", label="Other"),
            ]
            if len(self.history) > 1:
                left_legend.append(mpatches.Patch(facecolor="#BA7517", label="Trajectory"))
            ax_true.legend(handles=left_legend, loc="lower right", fontsize=8, framealpha=0.85,
                              facecolor="#2a2a2a", edgecolor="#555550", labelcolor=FG)
 
        # RIGHT PANEL: agent's learned model 
        if has_agent:
            # for class ValueIteration from qrl.algorithms.classical
            if hasattr(agent, "_V"):
                values     = agent._V.cpu().numpy().astype(float)
                agent_type = "VI"
            # for class QValueIteration from qrl.algorithms.classical
            elif hasattr(agent, "_Q"):
                values     = agent._Q.max(dim=1).values.cpu().numpy().astype(float)
                agent_type = "QI"
            else:
                warnings.warn("No value function found in agent, using default values.", UserWarning,stacklevel=2)
                values     = np.zeros(6)
                agent_type = "VI"
            counts_sa   = agent._counts.sum(dim=2).cpu().numpy()
            total_steps = int(counts_sa.sum())
            try:
                policy = agent.get_policy().cpu().numpy()
            except Exception:
                policy = np.zeros(6, dtype=int)
            panel_title = f"Agent's learned model  ({total_steps} total steps)"
 
 
            # node colors from value function 
            cmap      = cm.get_cmap("YlOrRd")
            v_min, v_max = values.min(), values.max()
            v_range   = v_max - v_min if v_max > v_min else 1.0
            agent_node_colors = []
            for n in range(6):
                if n == self.target_state_index:
                    agent_node_colors.append("#3B8BD4")   # target always blue
                else:
                    norm_v = (values[n] - v_min) / v_range
                    agent_node_colors.append(cmap(norm_v)) # color the nodes based on the value function
 
            # draw base (gray edges, value-colored nodes) 
            ax_agent.set_aspect("equal")
            ax_agent.axis("off")
            ax_agent.set_title(panel_title, fontsize=9, pad=10, color=FG)
            nx.draw_networkx_nodes(
                G, pos, ax=ax_agent,
                node_size=1600, node_color=agent_node_colors,
                edgecolors="#888780", linewidths=1.2,
            )
            # Value annotations below each node label
            val_sym = "V" if agent_type == "VI" else "Q"
            value_labels = {
                n: f"{STATE_LABELS[n]}\n{val_sym}={values[n]:.2f}" for n in range(6)
            }
            nx.draw_networkx_labels(
                G, pos, ax=ax_agent,
                labels=value_labels, font_size=8, font_color="#111111",
            )
 
            # draw edges with opacity from visit count 
            max_count = counts_sa.max() if counts_sa.max() > 0 else 1.0
            greedy_edges, explored_edges, unexplored_edges = [], [], []
 
            for s in range(6):
                for a, name in enumerate(ACTION_NAMES):
                    dst   = int(_TRANSITIONS[s, a])
                    count = counts_sa[s, a]
                    edge  = (s, dst)
                    if a == policy[s]:
                        greedy_edges.append((edge, name, count))
                    elif count > 0:
                        explored_edges.append((edge, count / max_count))
                    else:
                        unexplored_edges.append(edge)
 
            # Unexplored: very faint
            if unexplored_edges:
                nx.draw_networkx_edges(
                    G, pos, ax=ax_agent,
                    edgelist=unexplored_edges,
                    edge_color="#666660", alpha=0.15, width=0.5,
                    arrowsize=8, arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.10",
                    min_source_margin=30, min_target_margin=30,
                )
 
            # Explored non-greedy: alpha proportional to visit count
            # Draw in buckets by alpha level
            alpha_buckets: dict[float, list] = {}
            for edge, norm_count in explored_edges:
                alpha = round(0.2 + 0.6 * norm_count, 1)
                alpha_buckets.setdefault(alpha, []).append(edge)
            for alpha, edgelist in alpha_buckets.items():
                nx.draw_networkx_edges(
                    G, pos, ax=ax_agent,
                    edgelist=edgelist,
                    edge_color="#aaaaaa", alpha=alpha, width=1.0,
                    arrowsize=12, arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.10",
                    min_source_margin=30, min_target_margin=30,
                )
 
            # Greedy policy: bold teal, labeled with gate name
            greedy_edgelist = [e for e, _, _ in greedy_edges]
            greedy_labels   = {e: name for e, name, _ in greedy_edges}
            if greedy_edgelist:
                nx.draw_networkx_edges(
                    G, pos, ax=ax_agent,
                    edgelist=greedy_edgelist,
                    edge_color="#0F6E56", width=3.0, arrowsize=20,
                    arrowstyle="-|>", connectionstyle="arc3,rad=0.10",
                    min_source_margin=30, min_target_margin=30,
                )
                _label_edges(
                    ax_agent, greedy_labels,
                    font_size=8, font_color="#5DCAA5",
                )
 
            # colorbar for value function 
            # Label: show which quantity is actually displayed
            value_label = "V(s)" if agent_type == "VI" else "max_a Q(s,a)"
 
            # Ticks: min and max only, annotated with the owning state name
            min_state = int(np.argmin(values))
            max_state = int(np.argmax(values))
            def _plain(lbl):
                return lbl.replace("$","").replace("\\rangle",">").replace("{","").replace("}","")
            min_tick = f"{_plain(STATE_LABELS[min_state])}  {v_min:.2f}"
            max_tick = f"{_plain(STATE_LABELS[max_state])}  {v_max:.2f}"
 
            sm = cm.ScalarMappable(
                cmap=cmap,
                norm=mcolors.Normalize(vmin=v_min, vmax=v_max),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_agent, fraction=0.03, pad=0.04)
            cbar.set_label(value_label, fontsize=8, color=FG)
            cbar.set_ticks([v_min, v_max])
            cbar.set_ticklabels([min_tick, max_tick])
            cbar.ax.tick_params(labelsize=7, colors=FG)
            cbar.ax.yaxis.set_tick_params(color=FG)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)
 
            # right legend 
            right_legend = [
                mpatches.Patch(facecolor="#3B8BD4", edgecolor="#888780", label="Target state"),
                mlines.Line2D([], [], color="#0F6E56", linewidth=2.5,  label="Greedy policy"),
                mlines.Line2D([], [], color="#888780", linewidth=1.0,  label="Explored (alpha ∝ visits)"),
                mlines.Line2D([], [], color="#D3D1C7", linewidth=0.5,  label="Unexplored"),
            ]
            ax_agent.legend(handles=right_legend, loc="lower right", fontsize=8, framealpha=0.85,
                              facecolor="#2a2a2a", edgecolor="#555550", labelcolor=FG)
 
        fig.tight_layout()
        fig_array = self._fig_to_array(fig)
        self.fig_array_list.append(fig_array)
        plt.close(fig)
 
    # render: learning animation 

    def _fig_to_array(self, fig):
        """
        Convert a Matplotlib figure to a NumPy RGB array.

        Rasterizes the figure canvas and reshapes the resulting pixel buffer
        into a ``(H, W, 3)`` uint8 array suitable for use as an animation frame.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to convert.

        Returns
        -------
        np.ndarray
            RGB pixel array of shape ``(height, width, 3)``, dtype ``uint8``.
        """        
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
 
    def render(self, save_path_without_extension, interval=600, ffmpeg=False):
        """
        Render accumulated graph frames as an animation and save to disk.

        Assembles the list of graph snapshots captured by ``_render_graph()``
        into a single animation. Each frame corresponds to one call to
        ``_render_graph()``, producing a visual record of the agent's learning
        progression over episodes.

        Parameters
        ----------
        save_path_without_extension : str
            File path (without extension) where the animation will be saved.
            The appropriate extension (``.mp4`` or ``.gif``) is appended
            automatically based on the ``ffmpeg`` argument.
        interval : int, optional
            Delay between frames in milliseconds. Default is 600.
        ffmpeg : bool, optional
            If True, saves the animation as an MP4 using ffmpeg. If False,
            saves as a GIF using Pillow. Default is False.

        Raises
        ------
        ValueError
            If ``_render_graph()`` has not been called and no frames are available.

        Returns
        -------
        None
            This method produces an animation file but does not return a value.
        """
        if not self.fig_array_list:
            raise ValueError("No frames to render. Call _render_graph() first.")

        # Match figure size exactly to the frame pixel dimensions
        h, w = self.fig_array_list[0].shape[:2]
        dpi  = 100
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove all padding
        ax.axis("off")
        im = ax.imshow(self.fig_array_list[0])

        def _update(i):
            im.set_data(self.fig_array_list[i])
            return [im]

        ani = animation.FuncAnimation(
            fig, _update,
            frames=len(self.fig_array_list),
            interval=interval,
            blit=True,
        )
        ext = "mp4" if ffmpeg else "gif"
        ani.save(
            f"{save_path_without_extension}.{ext}",
            writer="ffmpeg" if ffmpeg else "pillow",
            dpi=dpi,
        )
        plt.close(fig)

    @property
    def state_index(self) -> int:
        """Current state index (0-5)."""
        return self._state_index
 
    @property
    def bloch_vector(self) -> np.ndarray:
        """Current Bloch vector (x, y, z)."""
        return STATE_BLOCH[self._state_index].copy()
 
    @staticmethod
    def transition_table() -> np.ndarray:
        """
        Return the deterministic state-transition table for the environment.

        Each entry ``T[s, a]`` gives the next state index when action ``a``
        is taken from state ``s``. Rows correspond to the 6 Bloch sphere
        states and columns to the 4 gate actions (H, X, Z, S).

        Returns
        -------
        np.ndarray
            Integer array of shape ``(6, 4)`` where ``T[s, a] = s'``.
        """

        return _TRANSITIONS.copy()
 
    def __repr__(self) -> str:
        """
        Return a concise string representation of the environment.

        Displays the current state label, target state label, and the
        step count relative to the maximum allowed steps.

        Returns
        -------
        str
            String of the form
            ``BlochSphereV1(state=<label>, target=<label>, steps=<n>/<max>)``.
        """        
        return (
            f"BlochSphereV1("
            f"state={STATE_LABELS[self._state_index]}, "
            f"target={STATE_LABELS[self.target_state_index]}, "
            f"steps={self.steps}/{self.max_steps})"
        )