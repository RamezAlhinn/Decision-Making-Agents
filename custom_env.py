# Imports:
# --------
import sys
import pygame
import numpy as np
import gymnasium as gym


# Custom Environment:
# -------------------
class MyEnv(gym.Env):
    def __init__(self, grid_size) -> None:
        super().__init__()

        self.state = None
        self.done = False
        self.info = {}
        self.reward = 0
        self.cell_size = 100
        self.grid_size = grid_size
        self.goal = np.array([4, 4])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)

        self.danger_states = []

        # Display:
        # --------
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))


    # Method A:
    # --------
    def add_danger(self, coordinates):
        self.danger_states.append(np.array(coordinates))


    # Method 1:
    # ---------
    def reset(self):
        self.state = np.array([0,0])
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2
            )

        return self.state, self.info

    # Method 2:
    # ---------
    def step(self, action):
        # Up: 0
        if action==0 and self.state[0]>0:
            self.state[0]-=1

        # Down: 1
        if action==1 and self.state[0]<self.grid_size-1:
            self.state[0] += 1


        # Right: 2
        if action==2 and self.state[1]<self.grid_size-1:
            self.state[1]+=1

        # Left: 3
        if action==3 and self.state[1]>0:
            self.state[1]-=1

        # Check termination:
        # ------------------
        # Goal:
        if np.array_equal(self.state, self.goal):
            self.done = True
            self.reward = +10
        # Danger:
        elif True in [np.array_equal(self.state, each_danger) for each_danger in self.danger_states]:
            self.done = True
            self.reward = -20    
        else:
            self.done = False
            self.reward = -0.01


        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2
            )

        return self.state, self.done, self.reward, self.info

    # Method 3:
    # ---------
    def render(self):
        # Code for closing the window
        for event in pygame.event.get():
            if event==pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background:
        # -----------
        self.screen.fill((255,255,255))

        # Draw gridlines:
        for col in range(self.grid_size):
            for row in range(self.grid_size):
                grid = pygame.Rect(col*self.cell_size,
                                   row*self.cell_size,
                                   self.cell_size,
                                   self.cell_size)
                pygame.draw.rect(self.screen,
                                 (200,200,200),
                                 grid,
                                 1)
                
        # Draw goal:
        goal = pygame.Rect(self.goal[1]*self.cell_size,
                    self.goal[0]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        pygame.draw.rect(self.screen,
                        (0,255,0),
                        goal)


        # Add danger states:
        for each_danger in self.danger_states:
            danger = pygame.Rect(each_danger[1]*self.cell_size,
                        each_danger[0]*self.cell_size,
                        self.cell_size,
                        self.cell_size)
            pygame.draw.rect(self.screen,
                            (255,0,0),
                            danger)
            
        # Draw agent:
        agent = pygame.Rect(self.state[1]*self.cell_size,
                    self.state[0]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        pygame.draw.rect(self.screen,
                        (0,0,255),
                        agent)
        

        pygame.time.wait(100)
        pygame.display.flip()


    # Method 4:
    # ---------
    def close(self):
        pygame.quit()


# Run as a script:
# ----------------
if __name__=="__main__":
    for _ in range(100):
        # Create environment:
        # -------------------
        env = MyEnv(grid_size=5)
        env.add_danger(coordinates=(3,1))
        env.add_danger(coordinates=(2,4))

        state, info = env.reset()

        print("Initial_state: ", state, "Distance to goal: ", info["Distance to goal"])

        for _ in range(15):
            # action = int(input("Choose an action: "))
            
            next_step, done, reward, info = env.step(env.action_space.sample())

            env.render()

            print(f"Next-state: {next_step}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

            if done:
                env.close()
                break

        env.close()
