# System imports.
import numpy as np
from sklearn.metrics import mean_squared_error

# Local imports.
from configs.environment import Config



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


class SudokuEnv(object):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid, full_grid):
        self.grid = None
        self.is_completed = None
        self.initial_grid = grid
        self.full_grid = full_grid

        # Actions
        self.action_space = Discrete(Config.N_ACTIONS)

        # Contient les valeurs de paramètres des cinq précédentes estimations
        self.observation_space = Box(
            low=1, high=9, shape=Config.OBS_SHAPE, dtype=np.int8
        )


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


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        done = (self.is_completed) or (self.current_step == Config.MAX_STEPS)

        # Fonction de reward
        if done and self.current_step < Config.MAX_STEPS:
            # Si done==True après Config.MAX_STEPS/2 étapes, alors reward=5
            reward = 10 - self.mse
        else:
            reward = - self.mse

        obs = self._next_observation()

        return obs, reward, done, {}


    def reset(self):
        self.current_step = 0
        self.is_completed = False
        self.grid = self.initial_grid

        self.mse_min = mean_squared_error(
            self.full_grid.flatten(), self.grid.flatten()
        )

        return self._next_observation()


    def render(self, mode="human", close=False):
        if mode == "human":
            rows = []
            for i, row in enumerate(self.grid):
                if i % 3 == 0 and i > 0:
                    rows.append("-" * 29)
                cols = []
                for col in row.reshape(3, 3):
                    cols.append("  ".join([str(el) for el in col]))
                rows.append(" " + " | ".join(cols))
            print(f"\n".join(rows))

        elif mode == "rgb_array":
            return self.grid
