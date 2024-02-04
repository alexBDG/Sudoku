# System imports.
import pygame
import numpy as np
import gymnasium as gym
from sklearn.metrics import mean_squared_error

# Local imports.
from ..configs.environment import Config

BLACK = (0, 0, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)



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
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 1
    }

    def __init__(self, grid, full_grid, render_mode="human"):
        self.grid = None
        self.is_completed = None
        self.initial_grid = grid
        self.full_grid = full_grid
        self.window_size = 512  # The size of the PyGame window

        # Actions
        self.action_space = Discrete(Config.N_ACTIONS)
        self._action_row_idx = 0
        self._action_col_idx = 0
        self._action_value = None
        self._action_filled_new_case = None

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
        self._action_value = None
        if action == 0 and self._action_row_idx > 0:
            self._action_row_idx -= 1
        elif action == 1 and self._action_row_idx < 8:
            self._action_row_idx += 1
        elif action == 2 and self._action_col_idx > 0:
            self._action_col_idx -= 1
        elif action == 3 and self._action_col_idx < 8:
            self._action_col_idx += 1
        else:
            self._action_value = (action + 1) - 4

        # On met à jour la grille
        self._action_filled_new_case = False
        if (self._action_value is not None and
            self.grid[self._action_row_idx, self._action_col_idx, 0] < 1):
            self._action_filled_new_case = True
            self.grid[self._action_row_idx, self._action_col_idx, 0] = self._action_value

        self.is_completed = np.all(self.grid > 0)

        self.mse = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )
        if self.mse < self.mse_min:
            self.mse_min = self.mse


    def _compute_reward(self, terminated=False):
        reward = 0.

        # Bonus for new cases
        if self._action_filled_new_case:
            reward += 1

        # if terminated:
        #     reward += np.sum(self.grid > 0) - np.sum(self.initial_grid > 0)

        if self.is_completed:
            reward += 100

        # # Malus for non unique values
        # for k in range(1, 10):
        #     mask = self.grid[:, :, 0] == k
        #     # Check row unicity
        #     reward -= np.sum(np.sum(mask, axis=0) > 1)
        #     # Check col unicity
        #     reward -= np.sum(np.sum(mask, axis=1) > 1)
        #     # Check submatrix unicity
        #     for i in range(3):
        #         for j in range(3):
        #             mask = self.grid[3*i:3*(i+1), 3*j:3*(j+1), 0] == k
        #             reward -= np.sum(mask > 1)

        # # Malus for error
        # reward -= self.mse

        return reward


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        terminated = (self.is_completed) or (self.current_step == Config.MAX_STEPS)
        reward = self._compute_reward(terminated)
        obs = self._next_observation()

        return obs, reward, terminated, False, {}


    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.is_completed = False
        self.grid = self.initial_grid.copy()

        self.mse_min = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )
        obs = self._next_observation()

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
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        pygame.font.init()
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

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

        # Local cursor
        x_i = self._action_col_idx * case_width + 1
        y_i = self._action_row_idx * case_width + 1
        x_i1 = (self._action_col_idx + 1) * case_width - 1
        y_i1 = (self._action_row_idx + 1) * case_width - 1
        pygame.draw.line(canvas, RED, (x_i, y_i), (x_i, y_i1), 2)
        pygame.draw.line(canvas, RED, (x_i1, y_i), (x_i1, y_i1), 2)
        pygame.draw.line(canvas, RED, (x_i, y_i), (x_i1, y_i), 2)
        pygame.draw.line(canvas, RED, (x_i, y_i1), (x_i1, y_i1), 2)

        # Remplissage des chiffres
        grid = self.grid[:, :, 0]
        initial_grid = self.initial_grid[:, :, 0]
        police = pygame.font.Font(None, 36)
        for i in range(9):
            for j in range(9):
                if initial_grid[i][j] > 0:
                    texte = police.render(str(initial_grid[i][j]), True, BLACK)
                elif grid[i][j] > 0:
                    texte = police.render(str(grid[i][j]), True, GREY)
                if initial_grid[i][j] + grid[i][j] > 0:
                    canvas.blit(
                        texte, (j * case_width + 20, i * case_width + 15)
                    )
        if self._action_value is not None:
            police = pygame.font.Font(None, 18)
            texte = police.render(str(self._action_value), True, RED)
            canvas.blit(
                texte, (
                    self._action_col_idx * case_width + 4,
                    self._action_row_idx * case_width + 4
                )
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()