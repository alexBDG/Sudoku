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



def get_unvalid_cases(grid):
    """
    Check the validity of a Sudoku by computing count of unvalid values.
    * Unicity of row values
    * Unicity of column values
    * Unicity of subgrid values
    A subgrid is a subset of the grid, of size (n^(1/2), n^(1/2)).

    Parameters
    ----------
    grid : ndarray
        Input Sudoku grid, shape of (n, n).

    Returns
    -------
    int
        Number of unvalid values
    """
    n = grid.shape[0]
    m = np.square(n)
    unvalid = 0
    for k in range(1, n + 1):
        mask = grid == k
        # Check row unicity
        unvalid += np.sum(np.sum(mask, axis=0) > 1)
        # Check col unicity
        unvalid += np.sum(np.sum(mask, axis=1) > 1)
        # Check submatrix unicity
        for i in range(m):
            for j in range(m):
                mask = grid[m*i:m*(i+1), m*j:m*(j+1)] == k
                unvalid += np.sum(mask > 1)
    return unvalid


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
        "render_fps": 4
    }

    def __init__(self, grid, full_grid, render_mode="human"):
        self.grid = None
        self.is_completed = None
        self.is_unvalid = False
        self.initial_grid = grid
        self.full_grid = full_grid
        self.window_size = 513  # The size of the PyGame window

        # Actions
        self.action_space = Discrete(Config.N_ACTIONS)
        self._action_row_idx = None
        self._action_col_idx = None
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
        self._action_row_idx = np.floor(action / (9 * 9)).astype(np.int8)
        self._action_col_idx = np.floor(action % (9 * 9) / 9).astype(np.int8)
        self._action_value = action % 9 + 1

        # On met à jour la grille
        self._action_filled_new_case = False
        if self.grid[self._action_row_idx, self._action_col_idx, 0] < 1:
            self._action_filled_new_case = True
            self.grid[self._action_row_idx, self._action_col_idx, 0] = self._action_value

        self.mse = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )
        if self.mse < self.mse_min:
            self.mse_min = self.mse


    def _compute_reward(self):
        reward = 0

        # Bonus for new cases
        if self._action_filled_new_case and not self.is_unvalid:
            reward += 1

        if self.is_completed and not self.is_unvalid:
            reward += 100

        # # Malus for error
        # reward -= self.mse

        return reward


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.is_completed = np.all(self.grid > 0)

        self.is_unvalid = False
        if self._action_filled_new_case:
            self.is_unvalid = get_unvalid_cases(self.grid[:, :, 0]) > 0

        terminated = (
            self.is_completed or
            self.current_step == Config.MAX_STEPS or
            self.is_unvalid
        )
        reward = self._compute_reward()
        obs = self._next_observation()

        return obs, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
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
        if self.current_step == 0 or self._action_filled_new_case:
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
                (self.window_size, self.window_size + 100)
            )
        pygame.font.init()
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        case_width = self.window_size // 9

        canvas = pygame.Surface((self.window_size, self.window_size + 100))
        canvas.fill(WHITE)
        for i in range(1, 9):
            # Horizontal lines
            pygame.draw.line(
                canvas, BLACK,
                (0, i * case_width), (self.window_size, i * case_width),
                2 if i % 3 == 0 else 1
            )
            # Vertical lines
            pygame.draw.line(
                canvas, BLACK,
                (i * case_width, 0), (i * case_width, self.window_size),
                2 if i % 3 == 0 else 1
            )
        # Last line
        pygame.draw.line(
            canvas, BLACK,
            (0, self.window_size), (self.window_size, self.window_size), 2
        )

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

        # Step counter
        police = pygame.font.Font(None, 36)
        texte = police.render(f"Step: {self.current_step}", True, RED)
        canvas.blit(
            texte, (self.window_size // 2 - 35, self.window_size + 45)
        )

        if self.is_unvalid:
            draw_warning(
                canvas, self.window_size // 3, self.window_size + 45,
                radius=25
            )
            draw_warning(
                canvas, self.window_size // 3 * 2, self.window_size + 45,
                radius=25
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


def draw_warning(canvas, x, y, radius=25):
    # Circle
    pygame.draw.circle(canvas, RED, (x, y), radius)

    # Cross
    cross_lenght = 50
    pygame.draw.line(
        canvas, RED,
        (x - cross_lenght // 2, y),
        (x + cross_lenght // 2, y), 2
    )
    pygame.draw.line(
        canvas, RED,
        (x, y - cross_lenght // 2),
        (x, y + cross_lenght // 2), 2
    )