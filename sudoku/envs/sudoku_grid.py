# System imports.
import os
import pygame
import numpy as np
import gymnasium as gym
from sklearn.metrics import mean_squared_error

# Local imports.
from utils.display import plot_sudoku
from configs.environment import Config

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)



class Discrete(object):
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.random.randint(0, self.shape)


class Box(object):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(
            low=self.low, high=self.high, size=self.shape
        ).astype(self.dtype)



class SudokuEnv(gym.Env):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {'render_modes': ["human", "ansi", "rgb_array"], "render_fps": 60}

    def __init__(self, grid, full_grid, render_mode="human"):
        self.grid = None
        self.is_completed = None
        self.initial_grid = grid
        self.full_grid = full_grid
        self.window_size = 512  # The size of the PyGame window

        # Actions
        self.action_space = Discrete(Config.N_ACTIONS)

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(
            low=1, high=9, shape=Config.OBS_SHAPE, dtype=np.int8
        )

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.window = None
        self.clock = None


    def _next_observation(self):
        return self.grid


    def _take_action(self, action):
        # On récupère les informations de l'action
        row_idx = np.floor(action / (9 * 9)).astype(np.int8)
        col_idx = np.floor(action % (9 * 9) / 9).astype(np.int8)
        value = action % 9 + 1

        # On met à jour la grille
        if self.grid[row_idx, col_idx, 0] < 1:
            self.grid[row_idx, col_idx, 0] = value

        self.is_completed = np.all(self.grid > 0)

        self.mse = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )
        if self.mse < self.mse_min:
            self.mse_min = self.mse


    def _compute_reward(self):
        reward = 0.
        if self.is_completed:
            reward += 100

        for k in range(1, 10):
            mask = self.grid[:, :, 0] == k
            # Check row unicity
            reward -= np.sum(np.sum(mask, axis=0) > 1)
            # Check col unicity
            reward -= np.sum(np.sum(mask, axis=1) > 1)
            # Check submatrix unicity
            for i in range(3):
                for j in range(3):
                    mask = self.grid[3*i:3*(i+1), 3*j:3*(j+1), 0] == k
                    reward -= np.sum(mask > 1)

        reward -= self.mse

        return reward


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        terminated = (self.is_completed) or (self.current_step == Config.MAX_STEPS)
        reward = self._compute_reward()
        obs = self._next_observation()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, {}


    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.is_completed = False
        self.grid = self.initial_grid

        self.mse_min = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )
        obs = self._next_observation()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}


    def render(self):
        if self.render_mode == "ansi":
            rows = []
            for i, row in enumerate(self.grid):
                if i % 3 == 0 and i > 0:
                    rows.append("+".join(["-" * 9] * 3))
                cols = []
                for col in row.reshape(3, 3):
                    cols.append("  ".join([str(el) for el in col]))
                rows.append(" " + " | ".join(cols))
            print(f"\n".join(rows))

        else:
            self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        grid = self.grid[:, :, 0]
        case_width = self.window_size // 9

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)
        for i in range(1, 9):
            # Lignes horizontales
            pygame.draw.line(
                canvas, BLACK,
                (0, i * case_width), (self.window_size, i * case_width),
                2 if i % 3 == 0 else 1
            )
            # Lignes verticales
            pygame.draw.line(
                canvas, BLACK,
                (i * case_width, 0), (i * case_width, self.window_size),
                2 if i % 3 == 0 else 1
            )

        # Remplissage des chiffres
        police = pygame.font.Font(None, 36)
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    texte = police.render(str(grid[i][j]), True, BLACK)
                    canvas.blit(
                        texte, (j * case_width + 20, i * case_width + 15)
                    )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()