# System imports.
import os
import pygame
import numpy as np
import pandas as pd
import gymnasium as gym
import pygame_chart as pyc

# Local imports.
try:
    from ..configs.settings import MAX_STEPS
    from ..configs.settings import sudoku_path
    from ..configs.settings import min_difficulty
except ImportError:
    MAX_STEPS = 1000
    sudoku_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs", "data", "sudoku-3m.csv"
    )
    min_difficulty = 0.

# Set seed
np.random.seed(0)

BLACK = (0, 0, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Environment variables
N_ACTIONS = 9 + 4
OBS_SHAPE = (9, 9, 2)



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
        df = pd.read_csv(sudoku_path)

        # Filter by difficulty
        df = df[df["difficulty"] <= min_difficulty]
        df.reset_index(drop=True, inplace=True)

        # Filter low difficulty grid
        self.puzzle = df["puzzle"].values
        self.solution = df["solution"]
        # self.clues = df["clues"]

        # Index - first one for test only
        self.n = 0 if self.step_mode == "test" else 1
        self.n_max = self.puzzle.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.n_max:
            raise StopIteration
        if self.step_mode == "train":
            self.n += 1

        # Transform the grid from str to 2d array
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
    name = "sudoku"

    def __init__(self, render_mode="human", step_mode="train",
                 dtype=np.float32):
        self.all_rewards = []
        self.cumulative_reward = None
        self.is_completed = False
        self.is_valid = False
        self.is_exact = False
        self.is_empty = False
        self.is_value = False
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
        self.action_space = Discrete(N_ACTIONS)
        self._action_value = None

        # Observations
        self.observation_space = Box(
            low=1, high=9, shape=OBS_SHAPE, dtype=self.dtype
        )

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # PyGames
        self.window = None
        self.clock = None


    def _init_grid(self):
        # Create a grid for index, filled with 0 and 1 (1 is for the cursor
        # position)
        idx_shape = OBS_SHAPE[:-1] + (1,)
        idx = np.zeros(idx_shape, dtype=self.dtype).flatten()
        idx[0] = 1
        np.random.shuffle(idx)
        idx = idx.reshape(idx_shape)

        # Get the new grid
        grid, solu = next(self.grid_generator)
        self.initial_grid = grid
        self.solution_grid = solu
        self.empty_cases = self.unfilled_cases = np.sum(grid <= 0)

        self.grid = np.concatenate([np.expand_dims(grid, axis=2), idx], axis=-1)


    def _next_observation(self):
        # Max normalization
        obs = self.grid / 9.
        return obs


    def _take_action(self, action):
        # Get action's informations
        self._action = action
        self._action_value = None
        self.is_value = False

        # Get current row/col indexes
        row_idx = np.arange(9)[self.grid[:, :, -1].sum(1).astype(bool)]
        col_idx = np.arange(9)[self.grid[:, :, -1].sum(0).astype(bool)]

        # Apply cursor changing or value setting
        if action == 0 and row_idx > 0:
            # Up
            self.grid[:, :, -1] = np.vstack([
                self.grid[1:, :, -1], self.grid[0, :, -1]
            ])
        elif action == 1 and row_idx < 8:
            # Down
            self.grid[:, :, -1] = np.vstack([
                self.grid[-1, :, -1], self.grid[:-1, :, -1]
            ])
        elif action == 2 and col_idx < 8:
            # Right
            self.grid[:, :, -1] = np.hstack([
                self.grid[:, -1, -1].reshape(-1, 1), self.grid[:, :-1, -1]
            ])
        elif action == 3 and col_idx > 0:
            # Left
            self.grid[:, :, -1] = np.hstack([
                self.grid[:, 1:, -1], self.grid[:, 0, -1].reshape(-1, 1)
            ])
        elif action >= 4:
            self.is_value = True
            self._action_value = (action + 1) - 4

        # Updating the grid
        self.is_valid = False
        self.is_exact = False
        self.is_empty = self.is_value and self.grid[row_idx, col_idx, 0] <= 0
        if self.is_empty:

            # Check validity
            tmp_grid = self.grid[:, :, 0].copy()
            tmp_grid[row_idx, col_idx] = self._action_value
            self.is_valid = check_grid_validity(tmp_grid)
            self.is_exact = (
                self.solution_grid[row_idx, col_idx] == self._action_value
            )

            if self.is_valid:
                self.grid[row_idx, col_idx, 0] = self._action_value


    def _compute_reward(self, terminated):
        reward = 0

        # If the perfect grid is found
        if terminated and self.is_completed:
            reward += 1000
        # If time out
        elif terminated:
            reward -= 1000

        # For all movements
        if not self.is_value:
            reward =- 1
        # If the case was already filled
        elif not self.is_empty:
            reward -= 1
        # If a correct value is found
        elif self.is_exact:
            reward += 100
        # If an incorrect value is found but is correct
        elif self.is_valid:
            reward += 10
        # If an incorrect value is found and is not correct
        # elif not self.is_valid:
        #     reward -= 10  # <== This is already a game breaking condition

        self.cumulative_reward += reward
        self.all_rewards.append(reward)

        self._action_reward = reward
        return reward


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.unfilled_cases = np.sum(self.grid[:, :, 0] <= 0)
        self.is_completed = self.unfilled_cases < 1

        terminated = (
            # Sudoku is finished
            self.is_completed or
            # Time out
            self.current_step == MAX_STEPS or
            # Filled case does not agreed with Sudoku rules
            self.is_empty and not self.is_valid
            # Try to fill a filled case
            # self.is_value and not self.is_empty
        )
        reward = self._compute_reward(terminated)
        obs = self._next_observation()

        return obs, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cumulative_reward = 0
        self.all_rewards = []
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
                (2 * self.window_size, self.window_size + 100)
            )
        pygame.font.init()
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        case_width = self.window_size // 9

        canvas = pygame.Surface((2 * self.window_size, self.window_size + 100))
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
        pygame.draw.line(
            canvas, BLACK,
            (self.window_size, 0), (self.window_size, self.window_size), 2
        )

        # Local cursor
        row_idx = int(np.arange(9)[self.grid[:, :, -1].sum(1).astype(bool)])
        col_idx = int(np.arange(9)[self.grid[:, :, -1].sum(0).astype(bool)])
        x_i = col_idx * case_width + 2
        y_i = row_idx * case_width + 2
        x_i1 = (col_idx + 1) * case_width - 4
        y_i1 = (row_idx + 1) * case_width - 4
        pygame.draw.line(canvas, RED, (x_i, y_i), (x_i, y_i1), 4)
        pygame.draw.line(canvas, RED, (x_i1, y_i), (x_i1, y_i1), 4)
        pygame.draw.line(canvas, RED, (x_i, y_i), (x_i1, y_i), 4)
        pygame.draw.line(canvas, RED, (x_i, y_i1), (x_i1, y_i1), 4)
        if self._action_value is not None:
            police = pygame.font.Font(None, 14)
            text = police.render(f"{self._action_value:.0f}", True, GREY)
            canvas.blit(
                text, (col_idx * case_width + 10, (row_idx + 1) * case_width - 15)
            )

        # Filling digit
        grid = self.grid[:, :, 0]
        police = pygame.font.Font(None, 36)
        police_solution = pygame.font.Font(None, 14)
        for i in range(9):
            for j in range(9):
                if self.initial_grid[i][j] > 0:
                    text = police.render(
                        f"{self.initial_grid[i][j]:.0f}", True, BLACK
                    )
                elif grid[i][j] > 0:
                    text = police.render(f"{grid[i][j]:.0f}", True, GREY)
                if self.initial_grid[i][j] + grid[i][j] > 0:
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

        if self.is_empty and not self.is_valid:
            draw_warning(
                canvas, self.window_size // 4, self.window_size + 45,
                radius=25
            )
            draw_warning(
                canvas, self.window_size // 4 * 3, self.window_size + 45,
                radius=25
            )

        # Add chart
        figure = pyc.Figure(
            canvas,
            self.window_size+25, 25,
            self.window_size-50, self.window_size-50,
            bg_color=WHITE
        )
        figure.line("", [0, 1], [0, 1], color=WHITE)
        figure.add_title("Reward evolution")
        figure.add_legend()
        figure.add_xaxis_label("Steps")
        figure.add_yaxis_label("Reward (log10)")
        if self.current_step > 0:
            x = np.arange(len(self.all_rewards)).tolist()
            y = np.array(self.all_rewards)
            sign_y = y / np.abs(y)
            figure.scatter(
                "Step Reward",
                x, (np.log10(y * sign_y) * sign_y).tolist(),
            )
            y = np.cumsum(self.all_rewards)
            sign_y = y / np.abs(y)
            figure.line(
                "Cumulative Reward",
                x, (np.log10(y * sign_y) * sign_y).tolist(),
            )
        figure.draw()
        canvas.blit(figure.background, (self.window_size+25, 25))

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


def play(fps=None, store=False):
    game = SudokuEnv(render_mode="human")

    if isinstance(fps, int):
        game.metadata["render_fps"] = fps

    if store:
        try:
            from .utils import EpisodeBuffer
        except ImportError:
            from utils import EpisodeBuffer

    # game loop
    while True:
        state, _ = game.reset()
        done = False
        if store: episode_buffer = EpisodeBuffer(name=game.name)
        while True:
            game.render()

            # Collect user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    elif event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_1 or event.key == pygame.K_KP1:
                        action = 4
                    elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                        action = 5
                    elif event.key == pygame.K_3 or event.key == pygame.K_KP3:
                        action = 6
                    elif event.key == pygame.K_4 or event.key == pygame.K_KP4:
                        action = 7
                    elif event.key == pygame.K_5 or event.key == pygame.K_KP5:
                        action = 8
                    elif event.key == pygame.K_6 or event.key == pygame.K_KP6:
                        action = 9
                    elif event.key == pygame.K_7 or event.key == pygame.K_KP7:
                        action = 10
                    elif event.key == pygame.K_8 or event.key == pygame.K_KP8:
                        action = 11
                    elif event.key == pygame.K_9 or event.key == pygame.K_KP9:
                        action = 12

                    if store: episode_buffer.store_frame(state)

                    state, reward, done, _, _ = game.step(action)
                    if store: episode_buffer.store_effect(action, reward, done)

            if done:
                if store: episode_buffer.save()
                break



if __name__ == '__main__':
    play()