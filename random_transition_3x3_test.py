import random
import numpy as np


class RandomTransitionGrid3x3Env:
    """Simple 3x3 grid with stochastic action transitions."""

    ACTIONS = {
        0: "Up",
        1: "Down",
        2: "Right",
        3: "Left",
    }

    def __init__(self, transition_prob=0.8, max_steps=20):
        """Initialize the 3x3 stochastic grid environment.

        transition_prob: probability that the intended action is executed.
        max_steps: maximum steps allowed before the episode terminates.
        """
        self.grid_size = 3
        self.transition_prob = transition_prob
        self.max_steps = max_steps
        self.action_space = list(self.ACTIONS.keys())
        self.reset(randomize=True)

    def reset(self, randomize=True):
        """Reset the environment for a new episode.

        If randomize is True, the start and goal positions are chosen randomly.
        Returns the initial state and info dictionary.
        """
        if randomize:
            cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
            self.start = np.array(random.choice(cells), dtype=int)
            self.goal = np.array(random.choice([cell for cell in cells if cell != tuple(self.start)]), dtype=int)
        self.state = self.start.copy()
        self.step_count = 0
        self.done = False
        self.info = {
            "start": tuple(self.start),
            "goal": tuple(self.goal),
        }
        return self.state.copy(), self.info

    def _apply_action(self, action):
        """Compute the next state after executing an action.

        The robot moves one cell in the requested direction if possible.
        If the move would leave the grid, the robot stays in place.
        """
        next_state = self.state.copy()
        if action == 0 and self.state[0] > 0:
            next_state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:
            next_state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:
            next_state[1] += 1
        elif action == 3 and self.state[1] > 0:
            next_state[1] -= 1
        return next_state

    def step(self, action):
        """Take one environment step with stochastic transitions.

        action: intended action chosen by the agent.
        The environment applies the intended action with probability transition_prob.
        Otherwise it applies a random different action.
        Returns (next_state, done, reward, info).
        """
        self.step_count += 1

        if random.random() < self.transition_prob:
            actual_action = action
        else:
            alternatives = [a for a in self.action_space if a != action]
            actual_action = random.choice(alternatives)

        next_state = self._apply_action(actual_action)
        self.state = next_state

        reward = -0.01
        self.done = False

        if np.array_equal(self.state, self.goal):
            reward = 1.0
            self.done = True
        elif self.step_count >= self.max_steps:
            self.done = True

        self.info["actual_action"] = actual_action
        self.info["intended_action"] = action
        self.info["position"] = tuple(self.state)
        return self.state.copy(), self.done, reward, self.info


def run_random_transition_experiment(
    num_episodes=1000,
    transition_prob=0.8,
    max_steps=20,
):
    """Run a batch of episodes and report how often the intended action was followed.

    The agent chooses random intended actions each step. The environment
    may execute a different action based on transition_prob.
    """
    env = RandomTransitionGrid3x3Env(transition_prob=transition_prob, max_steps=max_steps)

    action_followed = {action: [] for action in env.action_space}
    action_counts = {action: 0 for action in env.action_space}
    actual_action_counts = {action: 0 for action in env.action_space}

    for episode in range(num_episodes):
        state, info = env.reset(randomize=True)
        for step in range(max_steps):
            intended_action = random.choice(env.action_space)
            _, done, _, info = env.step(intended_action)
            actual_action = info["actual_action"]

            action_counts[intended_action] += 1
            actual_action_counts[actual_action] += 1
            action_followed[intended_action].append(actual_action == intended_action)

            if done:
                break

    print("\nRandom transition experiment summary")
    print("===================================")
    print(f"Grid: 3x3, Episodes: {num_episodes}, Transition probability: {transition_prob:.2f}\n")

    for action in env.action_space:
        attempts = action_counts[action]
        if attempts == 0:
            probability = 0.0
        else:
            probability = sum(action_followed[action]) / attempts
        print(
            f"{env.ACTIONS[action]:>5}: attempts={attempts:4}, "
            f"followed={sum(action_followed[action]):4}, "
            f"estimated success={probability:.3f}"
        )

    total_attempts = sum(action_counts.values())
    total_success = sum(sum(vals) for vals in action_followed.values())
    print("\nOverall follow rate:", f"{total_success}/{total_attempts} = {total_success/total_attempts:.3f}")
    print("Actual action distribution:")
    for action in env.action_space:
        print(f"  {env.ACTIONS[action]:>5}: {actual_action_counts[action]:4}")
    print("")

    return {
        "action_followed": action_followed,
        "action_counts": action_counts,
        "actual_action_counts": actual_action_counts,
        "transition_prob": transition_prob,
        "total_success": total_success,
        "total_attempts": total_attempts,
    }


if __name__ == "__main__":
    run_random_transition_experiment(num_episodes=1000, transition_prob=0.7, max_steps=20)
