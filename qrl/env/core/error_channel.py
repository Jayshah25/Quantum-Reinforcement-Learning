from gymnasium import spaces
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from base__ import QuantumEnv


class ErrorChannelV0(QuantumEnv):
    """
    Multi-qubit ErrorChannel environment with multiple simultaneous error channels.
    - Each faulty qubit can have its own error type (bitflip, phaseflip, amp_damp).
    - Agent chooses (gate, qubit) to correct.
    """

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


if __name__ == "__main__":
    env = ErrorChannelV0(
        n_qubits=3,
        faulty_errors={0: "bitflip", 2: "amp_damp"},
        noise_prob=0.2,
        max_steps=6,
        seed=42,
    )
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"step={len(env.corrections)} act={env.GATE_ID2NAME[action[0]]}@q{action[1]} "
              f"reward={reward:.3f} errors={info['faulty_errors']}")
    env.render(save_path="results/core/error_channelV0.mp4", interval=700)
