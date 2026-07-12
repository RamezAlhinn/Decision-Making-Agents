# =============================================================================
# env.py — Delivery Robot Environment
# =============================================================================
#
# Theme
# -----
# A robot navigates an 8x8 city grid to complete a two-phase delivery task:
#   Phase 1 — find the package (P) and pick it up
#   Phase 2 — carry the package to the customer (C)
#
# The environment is fully deterministic: the same action in the same state
# always produces the same next state and reward.  This makes it ideal for
# tabular Q-learning, where the agent builds an exact Q-table indexed by
# (row, col, has_package).
#
# Special states — 11 individual cells, 4 distinct mechanics
# ----------------------------------------------------------
#   Symbol  Name      What happens when the robot enters            Reward
#   ------  --------  ------------------------------------------   -------
#   W       Wall      Movement blocked; robot stays in place        −0.05
#   X       Danger    Traffic jam; episode ends immediately         −15
#   B       Bonus     Fuel station; one-time reward, then normal    +3
#   P       Package   Sub-goal; robot picks it up and enters Ph.2  +5
#   C       Customer  Terminal goal; only counts if robot has P     +20
#
# Every normal step costs −0.01 to push the agent toward shorter paths.
# Running out of steps (100 max) ends the episode with −10.
#
# State space : 8 × 8 × 2 = 128 discrete states
#               (row 0-7, col 0-7, has_package 0/1)
# Action space: Discrete(4) — 0=Up, 1=Down, 2=Right, 3=Left
# =============================================================================

import sys
import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------------------
# Map layout  (8 × 8 grid, all coordinates are (row, col))
# ---------------------------------------------------------------------------
GRID_SIZE = 8
MAX_STEPS = 100

#   4 walls forming an L-shaped barrier the robot must navigate around
WALL_CELLS   = [(1, 3), (2, 3), (3, 3), (3, 4)]

#   3 danger cells (traffic jams) scattered across the map
DANGER_CELLS = [(2, 1), (4, 5), (6, 2)]

#   2 bonus cells (fuel stations) — each gives +3 exactly once per episode
BONUS_CELLS  = [(0, 5), (5, 7)]

PACKAGE_CELL  = (1, 6)   # robot must reach here first
CUSTOMER_CELL = (7, 7)   # final delivery destination
START_CELL    = (0, 0)   # robot always starts here


class DeliveryRobotEnv(gym.Env):
    """
    8×8 delivery robot environment for tabular Q-learning.

    How to run
    ----------
    Standalone demo (random agent, opens a pygame window)::

        python env.py

    From training code::

        env = DeliveryRobotEnv(headless=True)   # no window — fast training
        obs, info = env.reset()
        obs, done, reward, info = env.step(action)
        env.close()

    Parameters
    ----------
    random_start : bool
        If True the robot is placed at a random free cell each episode.
        Used during experiments for better state coverage.
    headless : bool
        If True all pygame calls are skipped — required for experiment runs
        where no display is available or speed is critical.
    """

    def __init__(self, random_start=False, headless=False) -> None:
        """Initialise grid layout, action/observation spaces, and pygame (if not headless)."""
        super().__init__()

        self.grid_size    = GRID_SIZE
        self.cell_size    = 90
        self.max_steps    = MAX_STEPS
        self.random_start = random_start
        self.headless     = headless

        self.start    = np.array(START_CELL)
        self.package  = np.array(PACKAGE_CELL)
        self.customer = np.array(CUSTOMER_CELL)

        # Convert to numpy arrays once for fast comparison in step()
        self.wall_states   = [np.array(c) for c in WALL_CELLS]
        self.danger_states = [np.array(c) for c in DANGER_CELLS]
        self.bonus_states  = [np.array(c) for c in BONUS_CELLS]

        # Episode state — reset() sets these to their real starting values
        self.state           = None
        self.has_package     = False
        self.collected_bonus = set()
        self.done            = False
        self.reward          = 0
        self.info            = {}
        self.step_count      = 0

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([GRID_SIZE - 1, GRID_SIZE - 1, 1]),
            dtype=np.int32
        )

        if not self.headless:
            import pygame
            self.pygame     = pygame
            pygame.init()
            side = self.cell_size * self.grid_size
            self.screen     = pygame.display.set_mode((side, side))
            self.font       = pygame.font.SysFont(None, 36)
            self.font_small = pygame.font.SysFont(None, 22)
            pygame.display.set_caption("Delivery Robot")


    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------
    def reset(self):
        """Reset the episode and return the initial observation (obs, info)."""
        # Build the set of cells the robot cannot start on
        blocked = (
            {tuple(w) for w in self.wall_states}   |
            {tuple(d) for d in self.danger_states} |
            {tuple(b) for b in self.bonus_states}  |
            {PACKAGE_CELL, CUSTOMER_CELL}
        )

        if self.random_start:
            while True:
                row, col = np.random.randint(0, self.grid_size, size=2)
                if (row, col) not in blocked:
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


    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------
    def step(self, action):
        """
        Apply action and return (obs, done, reward, info).

        Action encoding: 0=Up, 1=Down, 2=Right, 3=Left.
        Priority order: wall bump → danger → bonus → package pickup →
        customer delivery → normal step → timeout.
        """
        self.step_count += 1

        # 1. Compute intended next position
        next_pos = self.state.copy()
        if action == 0 and self.state[0] > 0:                       # Up
            next_pos[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:    # Down
            next_pos[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:    # Right
            next_pos[1] += 1
        elif action == 3 and self.state[1] > 0:                     # Left
            next_pos[1] -= 1

        # 2. Check for wall — robot stays in place with a small penalty
        hits_wall = any(np.array_equal(next_pos, w) for w in self.wall_states)
        if hits_wall:
            self.reward = -0.05
            self.done   = False
        else:
            self.state = next_pos

            # 3. Evaluate the new cell
            if any(np.array_equal(self.state, d) for d in self.danger_states):
                # Danger — episode ends with a large penalty
                self.reward = -15
                self.done   = True

            elif any(np.array_equal(self.state, b) for b in self.bonus_states):
                # Bonus — first visit gives +3, later visits are normal steps
                key = tuple(self.state)
                if key not in self.collected_bonus:
                    self.collected_bonus.add(key)
                    self.reward = +3
                else:
                    self.reward = -0.01
                self.done = False

            elif not self.has_package and np.array_equal(self.state, self.package):
                # Package pickup — start Phase 2
                self.has_package = True
                self.reward      = +5
                self.done        = False

            elif self.has_package and np.array_equal(self.state, self.customer):
                # Successful delivery
                self.reward = +20
                self.done   = True

            elif not self.has_package and np.array_equal(self.state, self.customer):
                # Reached customer without the package — keep going
                self.reward = -0.01
                self.done   = False

            else:
                # Normal step
                self.reward = -0.01
                self.done   = False

        # 4. Timeout
        if not self.done and self.step_count >= self.max_steps:
            self.reward = -10
            self.done   = True

        self._update_info()
        return self._get_obs(), self.done, self.reward, self.info


    # -----------------------------------------------------------------------
    # render  (pygame)
    # -----------------------------------------------------------------------
    def render(self):
        """Draw the current grid state to the pygame window. No-op in headless mode."""
        if self.headless:
            return
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        self.screen.fill((245, 245, 245))
        cs = self.cell_size

        # Grid lines
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pg.draw.rect(self.screen, (210, 210, 210),
                             pg.Rect(c*cs, r*cs, cs, cs), 1)

        def draw_cell(row, col, color, label, text_color=(255,255,255)):
            rect = pg.Rect(col*cs, row*cs, cs, cs)
            pg.draw.rect(self.screen, color, rect)
            surf = self.font.render(label, True, text_color)
            self.screen.blit(surf, surf.get_rect(center=rect.center))

        for w in self.wall_states:
            draw_cell(w[0], w[1], (70, 70, 70), "W", (180,180,180))
        for d in self.danger_states:
            draw_cell(d[0], d[1], (210, 50, 50), "X")
        for b in self.bonus_states:
            if tuple(b) not in self.collected_bonus:
                draw_cell(b[0], b[1], (0, 185, 195), "B")
        if not self.has_package:
            draw_cell(self.package[0], self.package[1], (240, 200, 0), "P", (0,0,0))
        draw_cell(self.customer[0], self.customer[1], (50, 180, 50), "C")
        draw_cell(self.state[0], self.state[1], (30, 100, 210),
                  "R+" if self.has_package else "R")

        # HUD
        hud = self.font_small.render(
            f"Steps: {self.step_count}/{self.max_steps}   "
            f"Phase: {'delivering' if self.has_package else 'fetching'}   "
            f"Bonus: {len(self.collected_bonus)}/2",
            True, (40, 40, 40)
        )
        self.screen.blit(hud, (4, 4))

        # Legend (bottom strip)
        items = [
            ((70,70,70),    "W Wall"),
            ((210,50,50),   "X Danger"),
            ((0,185,195),   "B Bonus"),
            ((240,200,0),   "P Package"),
            ((50,180,50),   "C Customer"),
            ((30,100,210),  "R Robot"),
        ]
        lx = 4
        for color, text in items:
            pg.draw.rect(self.screen, color, pg.Rect(lx, cs*self.grid_size-20, 12, 12))
            surf = self.font_small.render(text, True, (30,30,30))
            self.screen.blit(surf, (lx+15, cs*self.grid_size-22))
            lx += surf.get_width() + 28

        pg.time.wait(120)
        pg.display.flip()


    # -----------------------------------------------------------------------
    # close
    # -----------------------------------------------------------------------
    def close(self):
        """Shut down the pygame window. Always call this after training is done."""
        if not self.headless:
            self.pygame.quit()


    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    def _get_obs(self):
        return np.array([self.state[0], self.state[1], int(self.has_package)],
                        dtype=np.int32)

    def _update_info(self):
        target = self.customer if self.has_package else self.package
        self.info["phase"]           = "delivering" if self.has_package else "fetching"
        self.info["steps_remaining"] = self.max_steps - self.step_count
        self.info["dist_to_target"]  = float(np.linalg.norm(self.state - target))
        self.info["bonus_collected"] = len(self.collected_bonus)


# ---------------------------------------------------------------------------
# Quick random-agent demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = DeliveryRobotEnv()
    print(f"Map: {GRID_SIZE}x{GRID_SIZE}  |  "
          f"{len(WALL_CELLS)} walls, {len(DANGER_CELLS)} danger, "
          f"{len(BONUS_CELLS)} bonus, 1 package, 1 customer\n")

    for ep in range(3):
        obs, info = env.reset()
        total = 0
        for _ in range(env.max_steps):
            obs, done, reward, info = env.step(env.action_space.sample())
            env.render()
            total += reward
            if done:
                break
        print(f"Episode {ep+1}: reward={total:.2f}  phase={info['phase']}  "
              f"bonus={info['bonus_collected']}")
    env.close()
