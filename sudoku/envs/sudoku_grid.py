# System imports.
import pygame
import numpy as np
import gymnasium as gym
from sklearn.metrics import mean_squared_error

# Local imports.
from ..configs import settings

BLACK = (0, 0, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)



def check_grid_validity(grid):
    """
    Check the validity of a Sudoku by computing count of unvalid values.
    * Unicity of row values
    * Unicity of column values
    * Unicity of subgrid values
    A subgrid is a subset of the grid, of size (\sqrt{n}, \sqrt{n}).

    (Performances: 313 µs ± 102 µs per loop, mean ± std. dev. of 100 runs, 100
    loops each)

    Parameters
    ----------
    grid : ndarray
        Input Sudoku grid, shape of (n, n).

    Returns
    -------
    bool
        Either the grid is valid or not.
    """
    n = grid.shape[0]
    m = np.sqrt(n).astype(int)
    is_valid = True
    # Check row unicity
    for row in grid:
        row = row[row > 0]
        is_valid = is_valid & (np.unique(row).shape[0] == row.shape[0])
    # Check col unicity
    for col in grid.T:
        col = col[col > 0]
        is_valid = is_valid & (np.unique(col).shape[0] == col.shape[0])
    # Check submatrix unicity
    for i in range(m):
        for j in range(m):
            subgrid = grid[m*i:m*(i+1), m*j:m*(j+1)]
            subgrid = subgrid[subgrid > 0]
            is_valid = is_valid & (np.unique(subgrid).shape[0] == subgrid.shape[0])
    return is_valid


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
        "render_fps": 10
    }

    def __init__(self, grid, full_grid, render_mode="human"):
        self.grid = None
        self.is_completed = None
        self.is_unvalid = False
        self.initial_grid = grid
        self.full_grid = full_grid
        self.window_size = 513  # The size of the PyGame window

        # Actions
        self.action_space = Discrete(settings.N_ACTIONS)
        self._action_row_idx = 0
        self._action_col_idx = 0
        self._action_value = None
        self._action_filled_new_case = None

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(
            low=1, high=9, shape=settings.OBS_SHAPE, dtype=np.int8
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
            self.is_unvalid = not check_grid_validity(self.grid[:, :, 0])

        terminated = (
            self.is_completed or
            self.current_step == settings.MAX_STEPS or
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

        # Step counter
        police = pygame.font.Font(None, 36)
        texte = police.render(f"Step: {self.current_step}", True, RED)
        canvas.blit(
            texte, (self.window_size // 2 - 35, self.window_size + 45)
        )

        if self.is_unvalid:
            draw_warning(
                canvas, self.window_size // 4, self.window_size + 45,
                radius=25
            )
            draw_warning(
                canvas, self.window_size // 4 * 3, self.window_size + 45,
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


def draw_warning(canvas, x, y, radius=25, width=5):
    # Circle
    pygame.draw.circle(canvas, RED, (x, y), radius=radius, width=width)

    # Cross
    pygame.draw.line(
        canvas, RED,
        (x - radius / np.sqrt(2), y + radius / np.sqrt(2)),
        (x + radius / np.sqrt(2), y - radius / np.sqrt(2)),
        width=width
    )
    pygame.draw.line(
        canvas, RED,
        (x - radius / np.sqrt(2), y - radius / np.sqrt(2)),
        (x + radius / np.sqrt(2), y + radius / np.sqrt(2)),
        width=width
    )