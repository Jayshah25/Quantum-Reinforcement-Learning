import gymnasium as gym
from gymnasium import spaces
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from base__ import QuantumEnv
from utils import GATES, RX, RY, RZ


class BlochSphereV0(QuantumEnv):
    def __init__(self, target_state, max_steps=20, reward_tolerance=0.99):
        super().__init__()
        self.max_steps = max_steps
        self.target_state = target_state
        self.state = np.array([1, 0], dtype=complex)  # Initial State -> |0>

        # Bloch vector (x, y, z)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Discrete action space
        self.actions = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
                        "RX_pi_2", "RX_pi_4", "RX_-pi_4",
                        "RY_pi_2", "RY_pi_4", "RY_-pi_4",
                        "RZ_pi_2", "RZ_pi_4", "RZ_-pi_4"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_tolerance = reward_tolerance

        self.history = []

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

        reward = np.abs(np.vdot(self.target, self.state))**2
        self.history.append((new_obs, round(reward, 3), gate))
        self.steps += 1
        done = reward > self.reward_tolerance or self.steps >= self.max_steps

        return self._state_to_bloch(self.state), reward, done, {}
    

    def render(self,save_path=None, interval=800):
        """
        history: list of bloch vectors for each step
        target_state: 3D bloch vector
        """
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])

        # Sphere (draw once)
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.5, edgecolor='gray', linewidth=0.1)

        # Axes limits
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

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

        if save_path:
            ani.save(save_path, writer='ffmpeg')
        else:
            plt.show()



if __name__=="__main__":
    # Target vector is |+> = (|0> + |1>)/sqrt(2)
    target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Initialize environment
    env = BlochSphereV0(target_state=target_state)

    # Reset
    obs, _ = env.reset()
    print("Initial Observation (r, theta, phi):", obs)

    # Randomly sample actions
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"After {action} action -> Observation:", obs)
        print("Reward:", reward, "Done:", done)

        if done:
            break

    # Render Bloch sphere
    env.render(save_path="results/core/bloch_sphere.mp4")