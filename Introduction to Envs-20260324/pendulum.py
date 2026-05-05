# Import:
# -------
import gymnasium as gym

# Step 0: Agenda
# -------
"""
We use the Pendulum environment to answer the followig question
    - What is a continuous environment?
        One which takes continuous values for states and actions. Read the documentation and observe the print output when you run the code.
"""
env = gym.make("Pendulum-v1", render_mode="human")

env.reset()

for _ in range(100):
    observation, reward, termincated, truncated, info = env.step([1.0])
    print(
        f"Observation: {observation}, Reward: {reward}, Terminated: {termincated}, Truncated: {truncated}, Info: {info}")
