Quickstart
==========

This guide will get you up and running with qrl-qai in just a few minutes. We'll demonstrate
PennyLane integration with qrl-qai, a standalone qrl-qai implementation, and classical
planning with Value Iteration!

Installation Check
------------------

First, make sure you have qrl-qai installed:

.. code-block:: bash

    pip install qrl-qai

Basic Workflow
--------------

The typical qrl-qai workflow follows these steps:

1. **Import** the environment and dependencies
2. **Initialize** the environment with your configuration
3. **Reset** the environment to get initial parameters
4. **Train** using your chosen optimizer
5. **Visualize** the results

Stand Alone Example
-------------------

.. code-block:: python

    #imports
    import numpy as np
    from qrl.env import BlochSphereV0

    # Target vector is |+> = (|0> + |1>)/sqrt(2)
    target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

    # Initialize environment
    # set ffmpeg=True if you have ffmpeg installed to save as mp4, or ffmpeg=False to save as gif
    env = BlochSphereV0(target_state=target_state, max_steps=20, reward_tolerance=0.99, ffmpeg=False)

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
    env.render(save_path_without_extension="bloch_sphere")

**Environment Parameters:**

- ``target_state``: The target pure state as a complex 2-vector
- ``reward_tolerance``: A reward threshold for considering the target state reached (between 0 and 1)
- ``max_steps``: Maximum number of optimization steps allowed
- ``ffmpeg``: If True, saves animations as mp4; if False, saves as gif

**Training Process:**

We randomly sample actions from the action space and step through the environment. The
environment returns the new observation, reward, and done flag after each action. The loop
continues until we either reach the target state (reward > reward_tolerance) or exhaust the
maximum number of steps.

**Visualization:**

After training, ``env.render()`` creates an animated visualization showing how the quantum
state evolved during training. The output will be saved as either a gif or mp4 file depending
on your ffmpeg setting.


Pennylane Integration Example
------------------------------

Here's an example that uses PennyLane's optimizer to train a quantum circuit to match a
uniform probability distribution:

.. code-block:: python

    from pennylane import numpy as np
    import pennylane as qml
    from qrl.env import ProbabilityV0

    # Define the problem
    n_qubits = 2
    target_distribution = np.array([0.25, 0.25, 0.25, 0.25])

    # Initialize environment
    env = ProbabilityV0(
        n_qubits=n_qubits,
        target_distribution=target_distribution,
        alpha=0.7,      # Balance between KL divergence and L2 distance
        beta=0.01,      # Penalty for taking steps
        max_steps=10,   # Maximum training steps
        ffmpeg=False    # Set to True if you have ffmpeg for mp4 output
    )

    # Reset environment to get initial parameters
    params, _ = env.reset()

    # Set up optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    # Training loop
    for step in range(env.max_steps):
        # Optimize parameters and get cost
        params, cost_val = opt.step_and_cost(env.cost_fn, params)
        probs = env.circuit(params)

        # Track progress
        env.history.append(probs)
        env.params = params
        reward = -cost_val
        env.rewards.append(reward)

        print(f"Step {step}: Reward = {reward:.4f}")

        # Stop if we've reached the target
        if reward > -1e-2:
            print("Target reached!")
            break

    # Generate visualization
    env.render(save_path_without_extension="probability_v0")

**Environment Parameters:**

- ``n_qubits``: Number of qubits in the quantum circuit
- ``target_distribution``: The probability distribution we want to learn
- ``alpha``: Weights the trade-off between KL divergence and L2 distance in the reward
- ``beta``: Penalty coefficient for each step taken (encourages efficiency)
- ``max_steps``: Maximum number of optimization steps allowed
- ``ffmpeg``: If True, saves animations as mp4; if False, saves as gif

**Training Process:**

The optimizer adjusts the quantum circuit parameters to maximize reward (minimize cost). The
environment tracks the evolution of probability distributions and can visualize the learning
process.

**Visualization:**

After training, ``env.render()`` creates an animated visualization showing how the quantum
state evolved during training. The output will be saved as either a gif or mp4 file depending
on your ffmpeg setting.


Value Iteration Example
-----------------------

``ValueIteration`` is a classical model-based planning algorithm included in ``qrl.algorithms``.
It builds an empirical transition model from environment interaction and applies the Bellman
optimality operator to compute the optimal policy. It works with any Gymnasium environment
that has discrete observation and action spaces — including qrl-qai quantum environments.

**On FrozenLake-v1:**

.. code-block:: python

    import gymnasium as gym
    from qrl.algorithms.classical import ValueIteration

    TEST_EPISODES = 20
    env      = gym.make("FrozenLake-v1", is_slippery=True)
    test_env = gym.make("FrozenLake-v1", is_slippery=True)
    agent    = ValueIteration(env=env, gamma=0.9)

    iter_no, best_reward = 0, 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)   # explore: seed the empirical model
        agent.value_iteration()          # plan: run Bellman updates to convergence

        reward = sum(agent.play_episode(test_env) for _ in range(TEST_EPISODES))
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if best_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break

**On BlochSphereV1:**

.. code-block:: python

    from qrl.algorithms.classical import ValueIteration
    from qrl.env import BlochSphereV1

    TEST_EPISODES = 20
    env      = BlochSphereV1(target_state=4, max_steps=10, reward_tolerance=0.99)
    test_env = BlochSphereV1(target_state=4, max_steps=10, reward_tolerance=0.99)
    agent    = ValueIteration(env=env, gamma=0.9)

    iter_no, best_reward = 0, 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(50)
        agent.value_iteration()
        env._render_graph(agent=agent)   # collect a graph snapshot for animation

        reward = 0.0
        for _ in range(TEST_EPISODES):
            obs, _ = test_env.reset()
            while True:
                action = agent.select_action(int(obs))
                obs, _, terminated, truncated, _ = test_env.step(action)
                if terminated or truncated:
                    reward += float(terminated)
                    break
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if best_reward >= 1.0:
            print("Solved in %d iterations!" % iter_no)
            break

    env.render(save_path_without_extension="bloch_sphere_value_iteration",
               interval=600, ffmpeg=False)

**Algorithm Parameters:**

- ``env``: A Gymnasium or qrl-qai environment with discrete observation and action spaces
- ``gamma``: Discount factor in [0, 1) — higher values make the agent more far-sighted
- ``play_n_random_steps(n)``: Collects ``n`` random transitions to seed the empirical model
- ``value_iteration()``: Applies Bellman updates until convergence
- ``select_action(state)``: Returns the greedy action via one-step lookahead on V

**Visualization:**

When used with ``BlochSphereV1``, calling ``env._render_graph(agent=agent)`` after each
iteration collects a snapshot of the state-transition graph annotated with the agent's
learned value function and greedy policy. After training, ``env.render()`` assembles all
snapshots into an animated gif.


What's Next?
------------

Now that you've seen the basics, you can:

- Explore other environments.
- Learn about the underlying concepts of the environments.
- Experiment with different optimizers from PennyLane.
- Try ``QValueIteration`` from ``qrl.algorithms.classical`` — it stores Q(s,a) directly,
  making action selection faster and serving as a natural stepping stone toward Q-learning.