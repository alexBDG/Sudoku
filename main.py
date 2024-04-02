# Local imports.
from sudoku.envs.sudoku_grid import SudokuEnv
from sudoku.core.q_nature import NatureQN
from sudoku.core.q_schedule import LinearSchedule
from sudoku.core.q_schedule import LinearExploration
from sudoku.configs import settings



if __name__ == "__main__":
    # make env
    env = SudokuEnv()

    # exploration strategy
    exp_schedule = LinearExploration(
        env, settings.eps_begin, settings.eps_end, settings.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(
        settings.lr_begin, settings.lr_end, settings.lr_nsteps
    )

    # train model
    # model = Linear(env, settings)
    model = NatureQN(env, settings)
    model.run(exp_schedule, lr_schedule)