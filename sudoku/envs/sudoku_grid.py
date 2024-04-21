# System imports.
import pygame
import numpy as np
import pandas as pd
import gymnasium as gym

# Local imports.
from ..configs import settings

# Set seed
np.random.seed(0)

BLACK = (0, 0, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)



class SudokuGenerator(object):
    """
    Generate Sudoku grids.

    References
    ----------
    `3 million Sudoku puzzles with ratings 
    <https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings>`
    """

    def __init__(self, step_mode="train"):
        # Either `"train"` or `"test"`
        self.step_mode = step_mode
        # Load data
        df = pd.read_csv(settings.sudoku_path)

        # Filter by difficulty
        df = df[df["difficulty"] <= settings.min_difficulty]
        df.reset_index(drop=True, inplace=True)

        # Filter low difficulty grid
        self.puzzle = df["puzzle"].values
        self.solution = df["solution"]
        # self.clues = df["clues"]

        # Index
        self.n = 0
        self.n_max = self.puzzle.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.n_max:
            raise StopIteration
        if self.step_mode == "train":
            self.n += 1

        # Transform the grid from str to 3d array
        grid = self.puzzle[self.n]
        grid = grid.replace(".", "0")
        grid = np.array([tuple(map(int, grid[n*9:n*9+9])) for n in range(0, 9)])
        grid = grid.astype("float32")

        # Transform the solution from str to 2d array
        solu = self.solution[self.n]
        solu = solu.replace(".", "0")
        solu = np.array([tuple(map(int, solu[n*9:n*9+9])) for n in range(0, 9)])
        solu = solu.astype("float32")
        return grid, solu


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
        self._grid_shape = shape[:-1] + (shape[-1] - 1,)  # Remove idx channels
        self._idx_shape = shape[:-1] + (1,)  # Remove data channels
        self.dtype = dtype

    def sample(self):
        # Create a grid for index, filled with 0 and 1 (1 is for the cursor
        # position)
        idx = np.zeros(self._idx_shape, dtype=self.dtype).flatten()
        idx[0] = 1
        np.random.shuffle(idx)
        idx = idx.reshape(self._idx_shape)

        # Create a grid for data
        data = np.random.randint(
            low=self.low, high=self.high+1, size=self._grid_shape,
        ).astype(self.dtype)

        return np.concatenate([data, idx], axis=-1)


class SudokuEnv(gym.Env):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "step_modes": ["train", "test"],
        "render_fps": 30
    }

    def __init__(self, render_mode="human", step_mode="train",
                 dtype=np.float32):
        self.cumulative_reward = None
        self.is_completed = None
        self.is_unvalid = False
        self.dtype = dtype
        self.window_size = 513  # The size of the PyGame window
        self.grid = None
        self.initial_grid = None
        self.initial_solu = None
        self.empty_cases = None
        self.unfilled_cases = None
        self.step_mode = step_mode

        # Initiate the grid generator
        self.grid_generator = SudokuGenerator(self.step_mode)

        # Actions
        self.action_space = Discrete(settings.N_ACTIONS)
        self._action_value = None
        self._action_row_idx = None
        self._action_col_idx = None
        self._action_filled_new_case = None

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(
            low=1, high=9, shape=settings.OBS_SHAPE, dtype=self.dtype
        )

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.window = None
        self.clock = None


    def _init_grid(self):
        # Get the new grid
        grid, solu = next(self.grid_generator)
        self.grid = grid.copy()
        self.initial_grid = grid
        self.solution_grid = solu
        self.empty_cases = self.unfilled_cases = np.sum(grid <= 0)


    def _next_observation(self):
        obs = np.expand_dims(self.grid, axis=2)
        # Max normalization
        obs = obs / 9.
        return obs


    def _take_action(self, action):
        # Get action's informations
        self._action = action
        self._action_value = None

        # Get current row/col indexes
        self._action_row_idx = np.floor(self._action / (9 * 9)).astype(np.int8)
        self._action_col_idx = np.floor(self._action % (9 * 9) / 9).astype(np.int8)
        self._action_value = action % 9 + 1
        row_idx = self._action_row_idx
        col_idx = self._action_col_idx

        # Updating the grid
        self.is_unvalid = False
        self._action_filled_new_case = False
        if self.grid[row_idx, col_idx] <= 0:
            self._action_filled_new_case = True

            # Check validity
            if True:#self.step_mode == "train":
                self.is_unvalid = (
                    self.solution_grid[row_idx, col_idx] != self._action_value
                )
            else:
                tmp_grid = self.grid.copy()
                tmp_grid[row_idx, col_idx] = self._action_value
                self.is_unvalid = not check_grid_validity(tmp_grid)

            if not self.is_unvalid:
                self.grid[row_idx, col_idx] = self._action_value


    def _compute_reward(self):
        reward = 0

        # Malus for bad value
        if self.is_unvalid:
            reward -= 1
        # Bonus for new cases
        elif self._action_filled_new_case:
            reward += 1
        else:
            reward -= 1

        if self.is_completed and not self.is_unvalid:
            reward += 100

        self.cumulative_reward += reward

        self._action_reward = reward
        return reward


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.unfilled_cases = np.sum(self.grid <= 0)
        self.is_completed = self.unfilled_cases < 1

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
        self.cumulative_reward = 0
        self.is_completed = False
        self._init_grid()

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
        if self._action_row_idx is not None or self._action_col_idx is not None:
            row_idx = self._action_row_idx
            col_idx = self._action_col_idx
            x_i = col_idx * case_width + 2
            y_i = row_idx * case_width + 2
            x_i1 = (col_idx + 1) * case_width - 4
            y_i1 = (row_idx + 1) * case_width - 4
            pygame.draw.line(canvas, RED, (x_i, y_i), (x_i, y_i1), 4)
            pygame.draw.line(canvas, RED, (x_i1, y_i), (x_i1, y_i1), 4)
            pygame.draw.line(canvas, RED, (x_i, y_i), (x_i1, y_i), 4)
            pygame.draw.line(canvas, RED, (x_i, y_i1), (x_i1, y_i1), 4)

            police = pygame.font.Font(None, 14)
            text = police.render(f"{self._action_value:.0f}", True, GREY)
            canvas.blit(
                text, (col_idx * case_width + 10, (row_idx + 1) * case_width - 15)
            )

        # Filling digit
        police = pygame.font.Font(None, 36)
        police_solution = pygame.font.Font(None, 14)
        for i in range(9):
            for j in range(9):
                if self.initial_grid[i][j] > 0:
                    text = police.render(
                        f"{self.initial_grid[i][j]:.0f}", True, BLACK
                    )
                elif self.grid[i][j] > 0:
                    text = police.render(f"{self.grid[i][j]:.0f}", True, GREY)
                if self.initial_grid[i][j] + self.grid[i][j] > 0:
                    canvas.blit(
                        text, (j * case_width + 20, i * case_width + 15)
                    )

                text = police_solution.render(
                    f"{self.solution_grid[i][j]:.0f}", True, RED
                )
                canvas.blit(
                    text, (j * case_width + 10, i * case_width + 10)
                )

        # Step counter
        police = pygame.font.Font(None, 36)
        text = police.render(f"Step: {self.current_step}", True, RED)
        canvas.blit(
            text, (self.window_size // 2 - 35, self.window_size + 15)
        )
        text = police.render(
            (
                f"Filled: {self.empty_cases - self.unfilled_cases}/"
                f"{self.empty_cases}"
            ), True, RED
        )
        canvas.blit(
            text, (self.window_size // 2 - 35, self.window_size + 45)
        )
        text = police.render(f"Reward: {self.cumulative_reward}", True, RED)
        canvas.blit(
            text, (self.window_size // 2 - 35, self.window_size + 75)
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
            # The following line copies our drawings from `canvas` to the
            # visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined
            # framerate. The following line will automatically add a delay to
            # keep the framerate stable.
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