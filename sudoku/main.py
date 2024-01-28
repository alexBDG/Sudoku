# System imports.
from envs.sudoku_grid import SudokuEnv

from core.q_linear import Linear
from core.q_schedule import LinearSchedule
from core.q_schedule import LinearExploration

from configs.dqn import config
from configs.sudoku_samples import GRID
from configs.sudoku_samples import FULL_GRID



if __name__ == "__main__":
    # make env
    env = SudokuEnv(GRID, FULL_GRID)

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(
        config.lr_begin, config.lr_end, config.lr_nsteps
    )

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)