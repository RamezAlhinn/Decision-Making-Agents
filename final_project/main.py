from env import DeliveryRobotEnv
from q_learning import QLearningAgent, train, visualize

# ── Settings ──────────────────────────────────────────────────────────────────
do_train        = True
do_visualize    = True
render_training = False   # Set True to watch the robot during training (slow)
random_start    = True    # Drop agent at a random non-obstacle cell each episode

q_table_path = "q_table.npy"

# ── Hyperparameters ───────────────────────────────────────────────────────────
no_episodes   = 30_000
alpha         = 0.1      # Learning rate
gamma         = 0.99     # Discount factor
epsilon       = 1.0      # Start fully exploring
epsilon_min   = 0.05     # Never drop below 5% exploration
epsilon_decay = 0.9995   # Multiply epsilon by this after every episode

# ── Run ───────────────────────────────────────────────────────────────────────
env   = DeliveryRobotEnv(random_start=random_start)
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
    print(f"Training for {no_episodes:,} episodes...\n")
    train(env, agent, no_episodes=no_episodes, render=render_training)
    agent.save(q_table_path)
else:
    agent.load(q_table_path)

if do_visualize:
    env = DeliveryRobotEnv()
    visualize(agent, env)
    env.close()
