from gymnasium import spaces
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from utils import GATES, RX, RY, RZ 
from base__ import QuantumEnv

class CompilerV0(QuantumEnv):
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


if __name__ == "__main__":

    theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
    target = (RZ(phi) @ RY(theta) @ RZ(lam))  # general SU(2)

    # Initialize environment with 1 qubit
    env = CompilerV0(target=target)

    # Reset
    obs, _ = env.reset()
    print("Initial Circuit State:", obs)

    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"After {action} action -> Observation:", obs)
        print("Reward:", reward, "Done:", done)

        if done:
            break

    # Render Bloch sphere
    env.render(save_path="results/core/compilerV0.mp4")