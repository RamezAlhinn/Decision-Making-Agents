# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     render,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        # so it starts in random position each episode
        state, _ = env.reset()

        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            
            if render:
                env.render()

            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or danger-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(danger_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4),
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Convert coordinates to tuples to make indexing and iteration consistent:
        # -----------------------------------------------------------------------
        danger_state_coordinates = [tuple(each_danger) for each_danger in danger_state_coordinates]
        goal_coordinates = tuple(goal_coordinates)

        # Create a mask for terminal states:
        # ----------------------------------
        terminal_state_mask = np.zeros(q_table.shape[:2], dtype=bool)
        terminal_state_mask[goal_coordinates] = True

        for each_danger in danger_state_coordinates:
            terminal_state_mask[each_danger] = True

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = terminal_state_mask.copy()

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9},
                        linewidths=0.5, linecolor="white")

            # Denote Goal and danger states:
            # ----------------------------
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)

            for danger_index, each_danger in enumerate(danger_state_coordinates, start=1):
                ax.text(each_danger[1] + 0.5, each_danger[0] + 0.5, f'D{danger_index}', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

        plt.tight_layout()
        plt.show()

        # Visualize the learned greedy policy:
        # ------------------------------------
        best_actions = np.argmax(q_table, axis=2)
        best_q_values = np.max(q_table, axis=2)
        action_arrows = {
            0: "↑",  # Up
            1: "↓",  # Down
            2: "→",  # Right
            3: "←"   # Left
        }

        _, ax = plt.subplots(figsize=(7, 6))

        sns.heatmap(best_q_values, annot=True, fmt=".2f", cmap="viridis",
                    ax=ax, cbar=True, mask=terminal_state_mask,
                    annot_kws={"size": 9}, linewidths=0.5, linecolor="white")

        for row in range(q_table.shape[0]):
            for col in range(q_table.shape[1]):
                current_state = (row, col)

                if current_state == goal_coordinates:
                    ax.text(col + 0.5, row + 0.5, 'G', color='green',
                            ha='center', va='center', weight='bold', fontsize=16)
                elif current_state in danger_state_coordinates:
                    danger_index = danger_state_coordinates.index(current_state) + 1
                    ax.text(col + 0.5, row + 0.5, f'D{danger_index}', color='red',
                            ha='center', va='center', weight='bold', fontsize=16)
                else:
                    ax.text(col + 0.5, row + 0.25, action_arrows[best_actions[row, col]], color='white',
                            ha='center', va='center', weight='bold', fontsize=18)

        ax.set_title('Learned Greedy Policy and Max Q-value')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
