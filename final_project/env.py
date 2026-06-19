# =============================================================================
# env.py — Delivery Robot Environment
# =============================================================================
#
# Theme: A robot navigates a 10x10 city grid to pick up a package (P) and
# deliver it to a customer (C).  The task is split into two phases:
#   Phase 1 — navigate to the package and pick it up
#   Phase 2 — carry it to the customer
#
# The environment is fully deterministic and tabular, making it well-suited
# for Q-learning with a discrete state space of size 10×10×2 (row, col,
# has_package) and 4 actions (Up, Down, Right, Left).
#
# Special state types — 20 individual cells, 6 distinct mechanics:
# ----------------------------------------------------------------
#   Symbol  Name       Description                                   Reward
#   ------  ---------  -------------------------------------------   ------
#   W       Wall       Blocks movement; robot stays in place         −0.05
#   X       Danger     Traffic jam; entering ends the episode        −15
#   T       Teleport   Warps robot to a fixed exit cell              −2
#   Z       Trap       Construction site; consumes an extra step     −3
#   B       Bonus      Fuel station; one-time pickup                 +3
#   P       Package    Sub-goal; must be collected before delivery   +5
#   C       Customer   Terminal goal; only valid when carrying P     +20
#
# Every normal step costs −0.01 to encourage short paths.
# Timeout after 150 steps ends the episode with a −10 penalty.
#
# Observation space: Box([row, col, has_package])  — shape (3,), dtype int32
# Action space:      Discrete(4)  — 0=Up, 1=Down, 2=Right, 3=Left
# =============================================================================

import sys
import pygame
import numpy as np
import gymnasium as gym


# Map configuration
# -----------------
GRID_SIZE = 10
MAX_STEPS = 150

# (row, col) coordinates — all checked in step() in priority order
WALL_CELLS    = [(1, 2), (2, 2), (3, 2), (4, 2),   # Left barrier
                 (5, 6), (6, 6), (7, 6), (8, 6)]   # Right barrier

DANGER_CELLS  = [(2, 5), (4, 7), (6, 3), (8, 1)]   # Traffic jams — episode ends

TELEPORT_PAIRS = {                                  # key: entry cell → value: exit cell
    (1, 7): (7, 2),
    (4, 4): (8, 8),
}

TRAP_CELLS    = [(3, 8), (7, 4)]                    # Construction sites — costs extra step

BONUS_CELLS   = [(0, 5), (6, 9)]                    # Fuel stations — one-time +3

PACKAGE_CELL  = (2, 8)                              # Pick up here first
CUSTOMER_CELL = (9, 9)                              # Deliver here to win

START_CELL    = (0, 0)                              # Robot always starts here


class DeliveryRobotEnv(gym.Env):
    def __init__(self, random_start=False) -> None:
        super().__init__()

        self.grid_size = GRID_SIZE
        self.cell_size = 80
        self.max_steps = MAX_STEPS
        self.random_start = random_start

        # Fixed locations
        self.start    = np.array(START_CELL)
        self.package  = np.array(PACKAGE_CELL)
        self.customer = np.array(CUSTOMER_CELL)

        # Pre-convert special cells to numpy arrays for fast comparison
        self.wall_states     = [np.array(w) for w in WALL_CELLS]
        self.danger_states   = [np.array(d) for d in DANGER_CELLS]
        self.teleport_map    = {k: np.array(v) for k, v in TELEPORT_PAIRS.items()}
        self.trap_states     = [np.array(t) for t in TRAP_CELLS]
        self.bonus_states    = [np.array(b) for b in BONUS_CELLS]

        # Episode state (initialised properly in reset)
        self.state            = None
        self.has_package      = False
        self.collected_bonus  = set()   # tracks which bonus cells are already spent
        self.done             = False
        self.reward           = 0
        self.info             = {}
        self.step_count       = 0

        # Action space: 0=Up 1=Down 2=Right 3=Left
        self.action_space = gym.spaces.Discrete(4)

        # Observation: [row, col, has_package]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.grid_size - 1, self.grid_size - 1, 1]),
            dtype=np.int32
        )

        # Display
        pygame.init()
        window_size = self.cell_size * self.grid_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Delivery Robot")
        self.font       = pygame.font.SysFont(None, 32)
        self.font_small = pygame.font.SysFont(None, 22)


    # Method 1: reset
    # ---------------
    def reset(self):
        all_special = (
            {tuple(w) for w in self.wall_states}   |
            {tuple(d) for d in self.danger_states} |
            set(TELEPORT_PAIRS.keys())              |
            {tuple(t) for t in self.trap_states}   |
            {tuple(b) for b in self.bonus_states}  |
            {PACKAGE_CELL, CUSTOMER_CELL}
        )

        if self.random_start:
            while True:
                row, col = np.random.randint(0, self.grid_size, size=2)
                if (row, col) not in all_special:
                    self.state = np.array([row, col])
                    break
        else:
            self.state = self.start.copy()

        self.has_package     = False
        self.collected_bonus = set()
        self.done            = False
        self.reward          = 0
        self.step_count      = 0

        self._update_info()
        return self._get_obs(), self.info


    # Method 2: step
    # --------------
    def step(self, action):
        self.step_count += 1

        # ── Move ──────────────────────────────────────────────────────────────
        next_state = self.state.copy()
        if action == 0 and self.state[0] > 0:
            next_state[0] -= 1                          # Up
        if action == 1 and self.state[0] < self.grid_size - 1:
            next_state[0] += 1                          # Down
        if action == 2 and self.state[1] < self.grid_size - 1:
            next_state[1] += 1                          # Right
        if action == 3 and self.state[1] > 0:
            next_state[1] -= 1                          # Left

        hits_wall = any(np.array_equal(next_state, w) for w in self.wall_states)
        if not hits_wall:
            self.state = next_state

        # ── Evaluate new position ─────────────────────────────────────────────

        # 1. Wall bump — wasted step
        if hits_wall:
            self.reward = -0.05
            self.done   = False

        # 2. Danger — episode ends immediately
        elif any(np.array_equal(self.state, d) for d in self.danger_states):
            self.reward = -15
            self.done   = True

        # 3. Teleport — warped to exit cell
        elif tuple(self.state) in self.teleport_map:
            self.state  = self.teleport_map[tuple(self.state)].copy()
            self.reward = -2
            self.done   = False

        # 4. Trap — wastes an extra step
        elif any(np.array_equal(self.state, t) for t in self.trap_states):
            self.step_count += 1   # extra step consumed
            self.reward      = -3
            self.done        = False

        # 5. Bonus — one-time fuel station reward
        elif any(np.array_equal(self.state, b) for b in self.bonus_states):
            key = tuple(self.state)
            if key not in self.collected_bonus:
                self.collected_bonus.add(key)
                self.reward = +3
            else:
                self.reward = -0.01   # already collected, normal step
            self.done = False

        # 6. Package pickup (Phase 1 → Phase 2)
        elif not self.has_package and np.array_equal(self.state, self.package):
            self.has_package = True
            self.reward      = +5
            self.done        = False

        # 7. Customer without package — no reward
        elif not self.has_package and np.array_equal(self.state, self.customer):
            self.reward = -0.01
            self.done   = False

        # 8. Delivery — terminal success
        elif self.has_package and np.array_equal(self.state, self.customer):
            self.reward = +20
            self.done   = True

        # 9. Normal move
        else:
            self.reward = -0.01
            self.done   = False

        # ── Timeout ───────────────────────────────────────────────────────────
        if not self.done and self.step_count >= self.max_steps:
            self.reward = -10
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

        # Walls — dark gray
        for w in self.wall_states:
            r = pygame.Rect(w[1]*self.cell_size, w[0]*self.cell_size,
                            self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (70, 70, 70), r)
            self.screen.blit(self.font.render("W", True, (180,180,180)),
                             (r.x+26, r.y+24))

        # Danger — red
        for d in self.danger_states:
            r = pygame.Rect(d[1]*self.cell_size, d[0]*self.cell_size,
                            self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (220, 50, 50), r)
            self.screen.blit(self.font.render("X", True, (255,255,255)),
                             (r.x+28, r.y+24))

        # Teleport entries — purple
        for entry, exit_rc in self.teleport_map.items():
            r = pygame.Rect(entry[1]*self.cell_size, entry[0]*self.cell_size,
                            self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (140, 60, 200), r)
            self.screen.blit(self.font.render("T", True, (255,255,255)),
                             (r.x+28, r.y+24))
            # Exit marker — lighter purple
            r2 = pygame.Rect(exit_rc[1]*self.cell_size, exit_rc[0]*self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (190, 140, 230), r2)
            self.screen.blit(self.font_small.render("t", True, (255,255,255)),
                             (r2.x+30, r2.y+28))

        # Traps — orange
        for t in self.trap_states:
            r = pygame.Rect(t[1]*self.cell_size, t[0]*self.cell_size,
                            self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (230, 140, 30), r)
            self.screen.blit(self.font.render("Z", True, (255,255,255)),
                             (r.x+28, r.y+24))

        # Bonus — cyan
        for b in self.bonus_states:
            if tuple(b) not in self.collected_bonus:
                r = pygame.Rect(b[1]*self.cell_size, b[0]*self.cell_size,
                                self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 190, 200), r)
                self.screen.blit(self.font.render("B", True, (255,255,255)),
                                 (r.x+28, r.y+24))

        # Package — yellow (only before pickup)
        if not self.has_package:
            r = pygame.Rect(self.package[1]*self.cell_size, self.package[0]*self.cell_size,
                            self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 210, 0), r)
            self.screen.blit(self.font.render("P", True, (0,0,0)),
                             (r.x+28, r.y+24))

        # Customer — green
        r = pygame.Rect(self.customer[1]*self.cell_size, self.customer[0]*self.cell_size,
                        self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (50, 190, 50), r)
        self.screen.blit(self.font.render("C", True, (255,255,255)),
                         (r.x+28, r.y+24))

        # Robot — blue
        r = pygame.Rect(self.state[1]*self.cell_size, self.state[0]*self.cell_size,
                        self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (30, 100, 220), r)
        symbol = "R+" if self.has_package else "R"
        self.screen.blit(self.font.render(symbol, True, (255,255,255)),
                         (r.x+20, r.y+24))

        # HUD
        hud = self.font_small.render(
            f"Steps: {self.step_count}/{self.max_steps}   "
            f"Phase: {self.info.get('phase','')}   "
            f"Bonus collected: {len(self.collected_bonus)}",
            True, (50, 50, 50)
        )
        self.screen.blit(hud, (6, 4))

        # Legend
        legend_items = [
            ((70,  70,  70),  "W = Wall"),
            ((220, 50,  50),  "X = Danger"),
            ((140, 60,  200), "T = Teleport"),
            ((230, 140, 30),  "Z = Trap"),
            ((0,   190, 200), "B = Bonus"),
            ((255, 210, 0),   "P = Package"),
            ((50,  190, 50),  "C = Customer"),
            ((30,  100, 220), "R = Robot"),
        ]
        legend_x = self.grid_size * self.cell_size - 160
        for i, (color, text) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(legend_x, 6 + i*18, 12, 12))
            self.screen.blit(self.font_small.render(text, True, (30,30,30)),
                             (legend_x+16, 4 + i*18))

        pygame.time.wait(100)
        pygame.display.flip()


    # Method 4: close
    # ---------------
    def close(self):
        pygame.quit()


    # Helpers
    # -------
    def _get_obs(self):
        return np.array([self.state[0], self.state[1], int(self.has_package)],
                        dtype=np.int32)

    def _update_info(self):
        target = self.customer if self.has_package else self.package
        self.info["phase"]              = "delivering" if self.has_package else "fetching"
        self.info["steps_remaining"]    = self.max_steps - self.step_count
        self.info["distance_to_target"] = float(np.sqrt(
            (self.state[0] - target[0])**2 + (self.state[1] - target[1])**2
        ))
        self.info["bonus_collected"]    = len(self.collected_bonus)


# Run as a script (random agent demo):
# --------------------------------------
if __name__ == "__main__":
    env = DeliveryRobotEnv()

    print("Delivery Robot — 10x10 map")
    print(f"Special cells: {len(WALL_CELLS)} walls, {len(DANGER_CELLS)} danger, "
          f"{len(TELEPORT_PAIRS)} teleports, {len(TRAP_CELLS)} traps, "
          f"{len(BONUS_CELLS)} bonus, 1 package, 1 customer\n")

    for episode in range(3):
        obs, info = env.reset()
        print(f"--- Episode {episode+1} ---")
        total = 0
        for _ in range(env.max_steps):
            action = env.action_space.sample()
            obs, done, reward, info = env.step(action)
            env.render()
            total += reward
            if done:
                break
        print(f"Total reward: {total:.2f}  |  Phase: {info['phase']}  |  "
              f"Bonus: {info['bonus_collected']}\n")

    env.close()
