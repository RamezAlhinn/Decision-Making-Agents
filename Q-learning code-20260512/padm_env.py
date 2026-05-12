# Imports:
# --------
import sys
import pygame
import numpy as np
import gymnasium as gym


# A sample custom environment:
# ----------------------------
class PadmEnv(gym.Env):
    def __init__(self, 
                 grid_size=5, 
                 goal_coordinates=(4, 4),
                 random_initialization=False) -> None:
        
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.danger_states = []
        self.random_initialization = random_initialization

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size*self.grid_size, self.cell_size*self.grid_size))

    # Method 1: .reset()
    # ---------
    def reset(self):
        """
        Everything must be reset
        """
        if self.random_initialization:
            """
            NOTE: Here, the code can also initialize the agent in danger states which can result 
            in immediate termination of the episode. Please be careful with this and take care of 
            this in your implementation.
            """
            self.state = np.array([np.random.choice([0,1,2,3]), np.random.choice([0,1,2,3])])
        else:
            self.state = np.array([0, 0])

        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt((self.state[0]-self.goal[0])**2 + 
                                                (self.state[1]-self.goal[1])**2)

        return self.state, self.info

    # Method 2: Add danger states
    # ---------
    def add_danger_states(self,
                          danger_state_coordinates):
        self.danger_states.append(np.array(danger_state_coordinates))

    # Method 3: .step()
    # ---------
    def step(self, 
             action):
        # Actions:
        # --------
        # Up:
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1

        # Down:
        if action == 1 and self.state[0] < self.grid_size-1:
            self.state[0] += 1

        # Right:
        if action == 2 and self.state[1] < self.grid_size-1:
            self.state[1] += 1

        # Left:
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # Reward:
        # -------
        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward = 10
            self.done = True
        # Check danger-states
        elif True in [np.array_equal(self.state, each_danger) for each_danger in self.danger_states]:
            self.reward = -1
            self.done = True
        else:  # Every other state
            self.reward = 0
            self.done = False

        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt((self.state[0]-self.goal[0])**2 + 
                                                (self.state[1]-self.goal[1])**2)

        return self.state, self.reward, self.done, self.info

    # Method 3: .render()
    # ---------
    def render(self):
        # Code for closing the window:
        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # We make the background White
        self.screen.fill((255, 255, 255))

        # Draw Grid lines:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(
                    y*self.cell_size, x*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid, 1)

        # Draw the Goal-state:
        goal = pygame.Rect(self.goal[1]*self.cell_size, self.goal[0]
                           * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal)

        # Draw the danger-states:
        for each_danger in self.danger_states:
            danger = pygame.Rect(
                each_danger[1]*self.cell_size, each_danger[0]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), danger)

        # Draw the agent:
        agent = pygame.Rect(self.state[1]*self.cell_size, self.state[0]
                            * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 0), agent)

        # Update contents on the window:
        pygame.display.flip()

    # Method 4: .close()
    # ---------
    def close(self):
        pygame.quit()


# Function 1: Create an instance of the environment
# -----------
def create_env(goal_coordinates,
               danger_state_coordinates,
               random_initialization):
    # Create the environment:
    # -----------------------
    env = PadmEnv(goal_coordinates=goal_coordinates,
                  random_initialization=random_initialization)

    for i in range(len(danger_state_coordinates)):
        env.add_danger_states(danger_state_coordinates=danger_state_coordinates[i])

    return env
