# Local imports.
from sudoku.envs.sudoku_grid import SudokuEnv
from sudoku.core.q_linear import Linear
from sudoku.core.q_schedule import LinearSchedule
from sudoku.core.q_schedule import LinearExploration
from sudoku.configs import settings
from sudoku.configs.sudoku_samples import GRID
from sudoku.configs.sudoku_samples import FULL_GRID



if __name__ == "__main__":
    # make env
    env = SudokuEnv(GRID, FULL_GRID)

    # exploration strategy
    exp_schedule = LinearExploration(
        env, settings.eps_begin, settings.eps_end, settings.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(
        settings.lr_begin, settings.lr_end, settings.lr_nsteps
    )

    # train model
    model = Linear(env, settings)
    model.run(exp_schedule, lr_schedule)