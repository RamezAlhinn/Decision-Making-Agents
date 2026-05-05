# Imports:
# --------
import gymnasium as gym

# Step 0: Agenda
# -------
"""
Here we use FrozenLake to answer the following questions
    - What is a discrete environment?
    - What is a static environment?
    - What is a fully observable environment?
    - What is the difference between deterministic and stochastic environment?
"""


# Step 1: Create the environment
# -------
"""
    - What is a discrete environment?
        An environment where states and actions have finite values. Observe the print output when you run the code.

    - What is a static environment?
        Observe that the surrounding does not change as the agent takes actions.

    - What is a fully observable environment?
        Observe the output of obervation. The grid number gives complete information regarding the position of the agent.

    - What is the difference between deterministic and stochastic environment?
        By setting is_slippery=True you get a stochastic environment.
"""

# User description:
# -----------------
stochastic_env = False

env = gym.make("FrozenLake-v1", render_mode="human",
               is_slippery=stochastic_env)

env.reset()

for _ in range(3):
    observation, reward, termincated, truncated, info = env.step(1)
    print(
        f"Observation: {observation}, Reward: {reward}, Terminated: {termincated}, Truncated: {truncated}, Info: {info}")
