# System imports.
import numpy as np
from sklearn.metrics import mean_squared_error

# Local imports.
from configs.environment import Config




class Discrete(object):
    def __init__(self, shape, dtype=np.int8):
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.randint(0, self.shape).astype(self.dtype)


class Box(object):
    def __init__(self, shape, dtype=np.float16):
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.rand(
            self.shape[0], self.shape[1], self.shape[2]
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
        self.observation_space = Box(shape=Config.OBS_SHAPE, dtype=np.int8)


    def _next_observation(self):
        return self.grid


    def _take_action(self, action):
        # On récupère les informations de l'action
        row_idx, col_idx, value = action

        # On met à jour la grille
        self.grid[row_idx, col_idx] = value + 1

        self.is_completed = np.all(self.grid > 0)

        self.mse = mean_squared_error(self.full_grid, self.grid)
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
            reward = 10*(Config.MAX_STEPS-self.current_step+1)/Config.MAX_STEPS
        else:
            reward = 0

        obs = self._next_observation()

        return obs, reward, done, {}


    def reset(self):
        self.is_completed = False
        self.grid = self.initial_grid


    def render(self, mode='human', close=False):
        grid_str = f"{'-' * 27}".join([
            " | ".join([
                "  ".join([str(el) for el in col]) for col in row.reshape(3, 3)
            ]) for row in self.grid
        ])
        print(grid_str)
