"""
environment.py
==============
Partially observable grid-maze environment, compatible with Gymnasium.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridMazeEnv(gym.Env):
    """
    A partially observable random-maze navigation environment.

    Parameters
    ----------
    maze_size : int
        Width and height of the square grid (default 15). Must be odd.
    obs_radius : int
        Radius of the square local observation window (default 2 → 5×5).
    max_steps : int
        Maximum steps before the episode is forcibly ended (default 400).
    render_mode : str or None
        Not yet implemented.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        maze_size: int = 15,
        obs_radius: int = 2,
        max_steps: int = 400,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if maze_size % 2 == 0:
            maze_size += 1

        self.maze_size = maze_size
        self.obs_radius = obs_radius
        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_side = 2 * obs_radius + 1
        self.obs_dim = 3 * obs_side * obs_side

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)
        self._deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # State variables
        self.maze: np.ndarray | None = None
        self.agent_pos: list[int] | None = None
        self.goal_pos: list[int] | None = None
        self.steps: int = 0
        self._visited: set[tuple[int, int]] | None = None
        self._prev_dist: float = 0.0

    # ------------------------------------------------------------------
    # Maze generation (iterative recursive backtracking)
    # ------------------------------------------------------------------

    def _generate_maze(self) -> np.ndarray:
        """Build a random perfect maze. 1 = wall, 0 = open."""
        maze = np.ones((self.maze_size, self.maze_size), dtype=np.int8)
        start = (1, 1)
        maze[start] = 0
        stack = [start]

        while stack:
            r, c = stack[-1]
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            self.np_random.shuffle(directions)

            advanced = False
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 < nr < self.maze_size - 1
                    and 0 < nc < self.maze_size - 1
                    and maze[nr, nc] == 1
                ):
                    maze[r + dr // 2, c + dc // 2] = 0
                    maze[nr, nc] = 0
                    stack.append((nr, nc))
                    advanced = True
                    break

            if not advanced:
                stack.pop()

        return maze

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.maze = self._generate_maze()
        open_cells = list(zip(*np.where(self.maze == 0)))

        if len(open_cells) < 2:
            raise RuntimeError("Maze generation produced fewer than 2 open cells.")

        # Choose start and goal far apart
        n = len(open_cells)
        best_dist = -1
        best_pair = (0, 1)
        for _ in range(min(100, n * n)):
            i, j = self.np_random.integers(0, n, size=2)
            if i == j:
                continue
            r1, c1 = open_cells[i]
            r2, c2 = open_cells[j]
            d = abs(r1 - r2) + abs(c1 - c2)
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)

        self.agent_pos = list(open_cells[best_pair[0]])
        self.goal_pos = list(open_cells[best_pair[1]])

        self.steps = 0
        self._visited = {tuple(self.agent_pos)}
        self._prev_dist = self._manhattan_dist()

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        dr, dc = self._deltas[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        moved = False
        if (
            0 <= nr < self.maze_size
            and 0 <= nc < self.maze_size
            and self.maze[nr, nc] == 0
        ):
            self.agent_pos = [nr, nc]
            moved = True

        self.steps += 1
        reached_goal = self.agent_pos == self.goal_pos

        reward = -0.01  # per-step penalty
        if not moved:
            reward -= 0.05  # wall collision

        if reached_goal:
            reward += 1.0
        else:
            curr_dist = self._manhattan_dist()
            reward += 0.01 * (self._prev_dist - curr_dist)
            self._prev_dist = curr_dist

            pos_key = tuple(self.agent_pos)
            if pos_key not in self._visited:
                reward += 0.005
                self._visited.add(pos_key)

        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        info = {
            "success": reached_goal,
            "steps": self.steps,
            "distance_to_goal": self._manhattan_dist(),
            "cells_visited": len(self._visited),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _manhattan_dist(self) -> float:
        return float(
            abs(self.agent_pos[0] - self.goal_pos[0])
            + abs(self.agent_pos[1] - self.goal_pos[1])
        )

    def _get_obs(self) -> np.ndarray:
        """
        Return flattened observation: walls, agent, goal channels.
        Out-of-bounds cells are treated as walls (1.0).
        """
        r, c = self.agent_pos
        rad = self.obs_radius
        side = 2 * rad + 1

        # Pre-allocate arrays
        walls = np.ones((side, side), dtype=np.float32)
        agent_ch = np.zeros((side, side), dtype=np.float32)
        goal_ch = np.zeros((side, side), dtype=np.float32)

        # Determine window boundaries
        r_min = max(0, r - rad)
        r_max = min(self.maze_size, r + rad + 1)
        c_min = max(0, c - rad)
        c_max = min(self.maze_size, c + rad + 1)

        # Map global maze slice to local window slice
        lr_min = r_min - (r - rad)
        lr_max = lr_min + (r_max - r_min)
        lc_min = c_min - (c - rad)
        lc_max = lc_min + (c_max - c_min)

        # Fill channels using array slicing
        walls[lr_min:lr_max, lc_min:lc_max] = self.maze[r_min:r_max, c_min:c_max].astype(np.float32)

        # Agent marker
        agent_lr = rad
        agent_lc = rad
        agent_ch[agent_lr, agent_lc] = 1.0

        # Goal marker if visible
        gr, gc = self.goal_pos
        if r - rad <= gr <= r + rad and c - rad <= gc <= c + rad:
            goal_lr = gr - (r - rad)
            goal_lc = gc - (c - rad)
            goal_ch[goal_lr, goal_lc] = 1.0

        return np.stack([walls, agent_ch, goal_ch], axis=0).flatten()