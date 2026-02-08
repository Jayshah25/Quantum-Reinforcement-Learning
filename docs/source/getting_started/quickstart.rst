Quickstart
==========

This guide will get you up and running with qrl-qai in just a few minutes. We'll demonstarte pennylane integration with qrl-qai and a standalone qrl-qai implementation!

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

- ``target_state``: The probability distribution we want to learn
- ``reward tolerance``: A reward threshold for considering the target state reached (between 0 and 1)
- ``max_steps``: Maximum number of optimization steps allowed
- ``ffmpeg``: If True, saves animations as mp4; if False, saves as gif

**Training Process:**

We randomly sample actions from the action space and step through the environment. The environment returns the new observation, reward, and done flag after each action. The loop continues until we either reach the target state (reward > reward_tolerance) or exhaust the maximum number of steps.

**Visualization:**

After training, ``env.render()`` creates an animated visualization showing how the quantum state evolved during training. The output will be saved as either a gif or mp4 file depending on your ffmpeg setting.


Pennylane Integration Example
-----------------------------


Here's an example that uses PennyLane optimizer to train a quantum circuit to match a uniform probability distribution:

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

Understanding the Code
----------------------

**Environment Parameters:**

- ``n_qubits``: Number of qubits in the quantum circuit
- ``target_distribution``: The probability distribution we want to learn
- ``alpha``: Weights the trade-off between KL divergence and L2 distance in the reward
- ``beta``: Penalty coefficient for each step taken (encourages efficiency)
- ``max_steps``: Maximum number of optimization steps allowed
- ``ffmpeg``: If True, saves animations as mp4; if False, saves as gif

**Training Process:**

The optimizer adjusts the quantum circuit parameters to maximize reward (minimize cost). The environment tracks the evolution of probability distributions and can visualize the learning process.

**Visualization:**

After training, ``env.render()`` creates an animated visualization showing how the quantum state evolved during training. The output will be saved as either a gif or mp4 file depending on your ffmpeg setting.

What's Next?
------------

Now that you've seen the basics, you can:

- Explore other environments.
- Learn about the underlying concepts of the environments.
- Experiment with different optimizers from PennyLane