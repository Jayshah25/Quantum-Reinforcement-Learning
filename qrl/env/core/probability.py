import gymnasium as gym
from gymnasium import spaces
import pennylane as qml
from pennylane import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ProbabilityV0(gym.Env):
    """
    Gym environment where the agent learns to construct a quantum circuit
    whose probability distribution matches a given target distribution.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 n_qubits: int,
                 target_distribution: np.ndarray,
                 ansatz=None,**kwargs):
        super(ProbabilityV0, self).__init__()

        assert np.isclose(np.sum(target_distribution), 1.0), \
            "Target distribution must sum to 1."
        self.n_qubits = n_qubits
        self.target_distribution = target_distribution
        self.max_steps = kwargs.get("max_steps", 100)
        self.tolerance = kwargs.get("tolerance", 1e-3)
        self.alpha = kwargs.get("alpha", 0.5)  # weight for KL vs L2
        self.beta = kwargs.get("beta", 0.01)    # step penalty weight

        # Define PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # If no ansatz is provided, define a simple one
        if ansatz is None:
            def default_ansatz(params, wires):
                for i, w in enumerate(wires):
                    qml.RY(params[i], wires=w)
            self.ansatz = default_ansatz
            self.n_params = self.n_qubits
        else:
            self.ansatz = ansatz
            try:
                self.n_params = ansatz.n_params  # If ansatz object has attribute
            except:
                raise ValueError("Please specify ansatz with n_params attribute.")

        # QNode
        @qml.qnode(self.dev)
        def circuit(params):
            self.ansatz(params, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        self.circuit = circuit

        # Spaces
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2**self.n_qubits,), dtype=np.float32)

        # Internal state
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.current_step = 0
        self.history = []

    def step(self, action):
        self.params = self.params + action
        self.current_step += 1

        probs = self.circuit(self.params)

        # Distribution errors
        kl_div = entropy(self.target_distribution + 1e-10, probs + 1e-10)
        l2_error = np.linalg.norm(self.target_distribution - probs, ord=2)

        # Weighted reward
        reward = -(self.alpha * kl_div + (1 - self.alpha) * l2_error)
        reward -= self.beta * self.current_step  # step penalty

        done = False
        if l2_error < 1e-3 or self.current_step >= self.max_steps:
            done = True

        self.history.append(probs)

        return probs, reward, done, {}


    def reset(self):
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.current_step = 0
        self.history = [self.circuit(self.params)]
        return self.history[-1]

    def render(self, mode="human"):
        if mode == "human":
            plt.figure(figsize=(10, 5))
            x = np.arange(2**self.n_qubits)

            plt.bar(x - 0.2, self.target_distribution, width=0.4, label="Target")

            if len(self.history) > 0:
                plt.bar(x + 0.2, self.history[-1], width=0.4, label="Current")

            plt.xlabel("Basis states")
            plt.ylabel("Probability")
            plt.title(f"Step {self.current_step}")
            plt.legend()
            plt.show()

    def animate(self, save_path=None):
        """
        Create an animation showing how the distribution evolves over steps,
        including reward values in the title.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(2**self.n_qubits)
        width = 0.4

        target_bar = ax.bar(x - 0.2, self.target_distribution, width=width, label="Target")
        current_bar = ax.bar(x + 0.2, self.history[0], width=width, label="Evolving")

        ax.set_ylim(0, 1)
        ax.set_xlabel("Basis states")
        ax.set_ylabel("Probability")
        ax.legend()

        # Precompute rewards for all steps in history
        rewards = []
        for probs in self.history:
            kl_div = entropy(self.target_distribution + 1e-10, probs + 1e-10)
            l2_error = np.linalg.norm(self.target_distribution - probs, ord=2)
            reward_val = -(self.alpha * kl_div + (1 - self.alpha) * l2_error)
            rewards.append(reward_val)

        def update(frame):
            probs = self.history[frame]
            for bar, new_height in zip(current_bar, probs):
                bar.set_height(new_height)
            ax.set_title(f"Step {frame} | Reward: {rewards[frame]:.4f}")
            return current_bar

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), blit=False)

        if save_path:
            ani.save(save_path, writer="ffmpeg", fps=2)
        else:
            plt.show()




if __name__ == "__main__":
    n_qubits = 3
    # probs over computational basis of size 2**n
    p_list = [0.2, 0.7, 0.35]
    n = len(p_list)
    probs = np.ones(1)
    for p in p_list:
        probs = np.kron(probs, np.array([1-p, p]))
    # safety normalization
    probs = np.clip(probs, 1e-9, 1.0)
    target_distribution = probs / probs.sum()

    env = ProbabilityV0(
        n_qubits=n_qubits,
        target_distribution=target_distribution,
        alpha=0.7,   # KL vs L2 weight
        beta=0.01,   # step penalty
        max_steps=100
    )

    # Reset environment
    obs = env.reset()

    # Define cost function = negative reward (because we want to maximize reward)
    def cost(params):
        probs = env.circuit(params)

        # KL divergence (target || probs)
        kl_div = np.sum(env.target_distribution * np.log((env.target_distribution + 1e-10) / (probs + 1e-10)))

        # L2 error
        l2_error = np.linalg.norm(env.target_distribution - probs, ord=2)

        # Reward
        reward = -(env.alpha * kl_div + (1 - env.alpha) * l2_error)
        return -reward  # minimize cost = maximize reward


    # Use PennyLane's optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    params = env.params.copy()
    for step in range(env.max_steps):
        params, cost_val = opt.step_and_cost(cost, params)
        probs = env.circuit(params)

        # Save history for rendering
        env.history.append(probs)
        env.params = params  # update env params
        reward = -cost_val
        print(f"Step {step+1}: Reward = {reward:.4f}")

        if reward > -1e-3:  # close to perfect
            break

    # Show final distribution
    # env.render()

    # Animate the full evolution
    env.animate()
