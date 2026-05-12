# Imports:
# --------
import sys
import pygame
import numpy as np
import gymnasium as gym


# Custom Environment: Delivery Robot
# -----------------------------------
# A robot navigates a city grid.
# Phase 1: reach the package (P)
# Phase 2: deliver it to the customer (C)
#
# Two obstacle types:
#   Traffic (X) — you CAN walk in, but episode ends with -15 penalty
#   Wall    (W) — you CANNOT walk in, move is blocked, robot stays
#
# Difficulty modes:
#   "easy"   — 6x6,   2 traffic,  0 walls  → open map, learn basics
#   "medium" — 8x8,   4 traffic,  4 walls  → walls form a barrier to navigate around
#   "hard"   — 10x10, 6 traffic,  8 walls  → walls create corridors, traffic inside them


# Difficulty settings table
# -------------------------
DIFFICULTIES = {
    "easy": {
        "grid_size":    6,
        "max_steps":    60,
        "danger_cells": [
            (1, 2), (3, 1),
        ],
        "wall_cells": [],           # No walls on easy
        "package":  (5, 4),
        "customer": (5, 5),
    },
    "medium": {
        "grid_size":    8,
        "max_steps":    80,
        "danger_cells": [
            (2, 6), (5, 1), (6, 5), (3, 4),
        ],
        "wall_cells": [             # Vertical barrier the robot must go around
            (1, 3), (2, 3), (3, 3), (4, 3),
        ],
        "package":  (7, 6),
        "customer": (7, 7),
    },
    "hard": {
        "grid_size":    10,
        "max_steps":    100,
        "danger_cells": [
            (1, 7), (3, 2), (4, 8), (6, 1), (7, 6), (8, 4),
        ],
        "wall_cells": [             # Two corridors the robot must navigate through
            (2, 3), (3, 3), (4, 3), (5, 3),   # Left wall
            (3, 6), (4, 6), (5, 6), (6, 6),   # Right wall
        ],
        "package":  (2, 8),
        "customer": (9, 9),
    },
}


class DeliveryRobotEnv(gym.Env):
    def __init__(self, difficulty="easy", random_start=False) -> None:
        super().__init__()

        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTIES.keys())}")

        self.difficulty = difficulty
        cfg = DIFFICULTIES[difficulty]

        self.grid_size = cfg["grid_size"]
        self.cell_size = 80
        self.max_steps = cfg["max_steps"]

        self.random_start = random_start

        # Fixed locations
        self.start    = np.array([0, 0])
        self.package  = np.array(cfg["package"])
        self.customer = np.array(cfg["customer"])

        # Episode state
        self.state       = None
        self.has_package = False
        self.done        = False
        self.reward      = 0
        self.info        = {}
        self.step_count  = 0

        # 4 actions: 0=Up, 1=Down, 2=Right, 3=Left
        self.action_space = gym.spaces.Discrete(4)

        # Observation: [row, col, has_package]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.grid_size - 1, self.grid_size - 1, 1]),
            dtype=np.int32
        )

        # Obstacles
        self.danger_states = [np.array(d) for d in cfg["danger_cells"]]
        self.wall_states   = [np.array(w) for w in cfg["wall_cells"]]

        # Display
        pygame.init()
        window_size = self.cell_size * self.grid_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption(f"Delivery Robot  [{difficulty.upper()}]")
        self.font       = pygame.font.SysFont(None, 32)
        self.font_small = pygame.font.SysFont(None, 22)


    # Method 1: reset
    # ---------------
    def reset(self):
        if self.random_start:
            blocked = (
                {tuple(d) for d in self.danger_states} |
                {tuple(w) for w in self.wall_states}   |
                {tuple(self.package), tuple(self.customer)}
            )
            while True:
                row, col = np.random.randint(0, self.grid_size, size=2)
                if (row, col) not in blocked:
                    self.state = np.array([row, col])
                    break
        else:
            self.state = self.start.copy()

        self.has_package = False
        self.done        = False
        self.reward      = 0
        self.step_count  = 0

        self._update_info()
        return self._get_obs(), self.info


    # Method 2: step
    # --------------
    def step(self, action):
        self.step_count += 1

        # Compute where the robot wants to move
        next_state = self.state.copy()
        if action == 0 and self.state[0] > 0:                      # Up
            next_state[0] -= 1
        if action == 1 and self.state[0] < self.grid_size - 1:     # Down
            next_state[0] += 1
        if action == 2 and self.state[1] < self.grid_size - 1:     # Right
            next_state[1] += 1
        if action == 3 and self.state[1] > 0:                      # Left
            next_state[1] -= 1

        # Walls block movement — robot stays in place
        hits_wall = any(np.array_equal(next_state, w) for w in self.wall_states)
        if not hits_wall:
            self.state = next_state

        # Evaluate the position after moving
        # ------------------------------------

        # Hit traffic? (robot walked into danger)
        in_danger = any(np.array_equal(self.state, d) for d in self.danger_states)
        if in_danger:
            self.reward = -15
            self.done   = True

        # Picked up the package?
        elif not self.has_package and np.array_equal(self.state, self.package):
            self.has_package = True
            self.reward      = +5
            self.done        = False

        # Reached customer WITHOUT package — no reward, keep going
        elif not self.has_package and np.array_equal(self.state, self.customer):
            self.reward = -0.01
            self.done   = False

        # Delivered the package!
        elif self.has_package and np.array_equal(self.state, self.customer):
            self.reward = +20
            self.done   = True

        # Bumped into a wall — small extra penalty for wasting a step
        elif hits_wall:
            self.reward = -0.05
            self.done   = False

        # Normal step
        else:
            self.reward = -0.01
            self.done   = False

        # Out of steps?
        if not self.done and self.step_count >= self.max_steps:
            self.reward = -5
            self.done   = True

        self._update_info()
        return self._get_obs(), self.done, self.reward, self.info


    # Method 3: render
    # ----------------
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))

        # Grid lines
        for col in range(self.grid_size):
            for row in range(self.grid_size):
                cell = pygame.Rect(col * self.cell_size, row * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), cell, 1)

        # Walls (dark gray) — drawn first so other items appear on top
        for wall in self.wall_states:
            rect = pygame.Rect(wall[1] * self.cell_size, wall[0] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (70, 70, 70), rect)
            label = self.font.render("W", True, (180, 180, 180))
            self.screen.blit(label, (rect.x + 26, rect.y + 24))

        # Traffic (red)
        for danger in self.danger_states:
            rect = pygame.Rect(danger[1] * self.cell_size, danger[0] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (220, 50, 50), rect)
            label = self.font.render("X", True, (255, 255, 255))
            self.screen.blit(label, (rect.x + 28, rect.y + 24))

        # Package (yellow) — only if not yet picked up
        if not self.has_package:
            rect = pygame.Rect(self.package[1] * self.cell_size, self.package[0] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 210, 0), rect)
            label = self.font.render("P", True, (0, 0, 0))
            self.screen.blit(label, (rect.x + 28, rect.y + 24))

        # Customer (green)
        rect = pygame.Rect(self.customer[1] * self.cell_size, self.customer[0] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (50, 190, 50), rect)
        label = self.font.render("C", True, (255, 255, 255))
        self.screen.blit(label, (rect.x + 28, rect.y + 24))

        # Robot (blue) — R when empty, R+ when carrying
        rect = pygame.Rect(self.state[1] * self.cell_size, self.state[0] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (30, 100, 220), rect)
        symbol = "R+" if self.has_package else "R"
        label = self.font.render(symbol, True, (255, 255, 255))
        self.screen.blit(label, (rect.x + 20, rect.y + 24))

        # HUD
        hud = self.font_small.render(
            f"[{self.difficulty.upper()}]  Steps: {self.step_count}/{self.max_steps}"
            f"   Phase: {self.info.get('phase', '')}",
            True, (50, 50, 50)
        )
        self.screen.blit(hud, (6, 4))

        # Legend
        legend_items = [
            ((70, 70, 70),   "W = Wall (blocks)"),
            ((220, 50, 50),  "X = Traffic (danger)"),
            ((255, 210, 0),  "P = Package"),
            ((50, 190, 50),  "C = Customer"),
            ((30, 100, 220), "R = Robot"),
        ]
        legend_x = self.grid_size * self.cell_size - 160
        for i, (color, text) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(legend_x, 6 + i * 18, 12, 12))
            lbl = self.font_small.render(text, True, (30, 30, 30))
            self.screen.blit(lbl, (legend_x + 16, 4 + i * 18))

        wait_time = {"easy": 150, "medium": 120, "hard": 80}[self.difficulty]
        pygame.time.wait(wait_time)
        pygame.display.flip()


    # Method 4: close
    # ---------------
    def close(self):
        pygame.quit()


    # Helpers
    # -------
    def _get_obs(self):
        return np.array([self.state[0], self.state[1], int(self.has_package)], dtype=np.int32)

    def _update_info(self):
        target = self.customer if self.has_package else self.package
        self.info["phase"]              = "delivering" if self.has_package else "fetching package"
        self.info["steps_remaining"]    = self.max_steps - self.step_count
        self.info["distance_to_target"] = np.sqrt(
            (self.state[0] - target[0])**2 + (self.state[1] - target[1])**2
        )


# Run as a script (random agent demo):
# -------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes",   default=3, type=int)
    args = parser.parse_args()

    env = DeliveryRobotEnv(difficulty=args.difficulty)

    print(f"\nDifficulty : {args.difficulty.upper()}")
    print(f"Grid size  : {env.grid_size}x{env.grid_size}")
    print(f"Max steps  : {env.max_steps}")
    print(f"Traffic    : {len(env.danger_states)} cells")
    print(f"Walls      : {len(env.wall_states)} cells\n")

    for episode in range(args.episodes):
        obs, info = env.reset()
        print(f"--- Episode {episode + 1} ---")
        print(f"Start | State: {obs} | Phase: {info['phase']} | Distance: {info['distance_to_target']:.2f}")

        for step in range(env.max_steps):
            action = env.action_space.sample()
            obs, done, reward, info = env.step(action)
            env.render()

            print(f"Step {step+1:3d} | {['Up  ','Down','Right','Left'][action]} | "
                  f"State: {obs} | Reward: {reward:+6.2f} | "
                  f"Phase: {info['phase']:<18} | "
                  f"Dist: {info['distance_to_target']:.2f} | "
                  f"Steps left: {info['steps_remaining']}")

            if done:
                if reward == +20:
                    result = "DELIVERED! SUCCESS"
                elif reward == -15:
                    result = "HIT TRAFFIC! FAILED"
                elif reward == -5:
                    result = "OUT OF STEPS! FAILED"
                else:
                    result = "FAILED"
                print(f">>> {result}\n")
                break

        env.close()
