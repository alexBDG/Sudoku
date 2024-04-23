# System imports.
import argparse

# Local imports.
from sudoku.envs.snake_grid import SnakeEnv
from sudoku.envs.sudoku_grid import SudokuEnv
from sudoku.core.q_nature import NatureQN
from sudoku.core.q_schedule import LinearSchedule
from sudoku.core.q_schedule import LinearExploration
from sudoku.configs import settings



def get_parsed_arguments():
    parser = argparse.ArgumentParser(prog="Sudoku")
    parser.add_argument(
        "--game", help=(
            "Which game to play."
        ), default="sudoku", choices=["sudoku", "snake"]
    )
    parser.add_argument(
        "--render_mode", help=(
            "Compute the render frames as specified by render_mode during the "
            "initialization of the environment."
        ), default="rgb_array", choices=["human", "ansi", "rgb_array"]
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_parsed_arguments()

    # make env
    if args.game == "sudoku":
        env = SudokuEnv(render_mode=args.render_mode)
    else:
        env = SnakeEnv(render_mode=args.render_mode)

    # exploration strategy
    exp_schedule = LinearExploration(
        env, settings.eps_begin, settings.eps_end, settings.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(
        settings.lr_begin, settings.lr_end, settings.lr_nsteps
    )

    # train model
    model = NatureQN(env, settings)
    model.run(exp_schedule, lr_schedule)