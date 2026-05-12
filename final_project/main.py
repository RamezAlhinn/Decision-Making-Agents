from env import DeliveryRobotEnv
from q_learning import QLearningAgent, train, visualize

# ── Settings ──────────────────────────────────────────────────────────────────
difficulty       = "hard"       # "easy" | "medium" | "hard"

do_train         = True
do_visualize     = True
render_training  = False        # Set True to watch the robot during training (slow)
random_start     = True         # Drop agent at a random non-obstacle cell each episode

q_table_path     = f"q_table_{difficulty}.npy"

# ── Hyperparameters ───────────────────────────────────────────────────────────
no_episodes    = 30_000
alpha          = 0.1      # Learning rate — how fast Q-values are updated
gamma          = 0.99     # Discount factor — how much future rewards matter
epsilon        = 1.0      # Start fully exploring
epsilon_min    = 0.05     # Never drop below 5% exploration
epsilon_decay  = 0.9995   # Multiply epsilon by this after every episode

# ── Run ───────────────────────────────────────────────────────────────────────
env   = DeliveryRobotEnv(difficulty=difficulty, random_start=random_start)
agent = QLearningAgent(
    grid_size     = env.grid_size,
    n_actions     = env.action_space.n,
    alpha         = alpha,
    gamma         = gamma,
    epsilon       = epsilon,
    epsilon_min   = epsilon_min,
    epsilon_decay = epsilon_decay,
)

if do_train:
    print(f"Training on [{difficulty.upper()}] for {no_episodes:,} episodes...\n")
    train(env, agent, no_episodes=no_episodes, render=render_training)
    agent.save(q_table_path)
else:
    agent.load(q_table_path)

if do_visualize:
    # Re-create env without pygame window for visualization (close was called in train)
    env = DeliveryRobotEnv(difficulty=difficulty)
    visualize(agent, env)
    env.close()
