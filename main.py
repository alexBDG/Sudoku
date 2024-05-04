# System imports.
import argparse



def get_parsed_arguments():
    parser = argparse.ArgumentParser(prog="Sudoku")
    parser.add_argument(
        "--game", help=(
            "Which game to play."
        ), default="sudoku", choices=["sudoku", "snake"]
    )
    parser.add_argument(
        "--render-mode", help=(
            "Compute the render frames as specified by render_mode during the "
            "initialization of the environment."
        ), default="rgb_array", choices=["human", "ansi", "rgb_array"]
    )
    parser.add_argument(
        "--playing-mode", help=(
            "How to play the game, using the algorithm or manualy."
        ), default="learning", choices=["learning", "manual"]
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_parsed_arguments()

    if args.playing_mode == "manual":
        if args.game == "sudoku":
            from sudoku.envs.sudoku_grid import play
        elif args.game == "snake":
            from sudoku.envs.snake_grid import play
        play(fps=4, store=True)
        quit()

    # Local imports.
    from sudoku.envs.snake_grid import SnakeEnv
    from sudoku.envs.sudoku_grid import SudokuEnv
    from sudoku.core.q_nature import NatureQN
    from sudoku.core.q_schedule import LinearSchedule
    from sudoku.core.q_schedule import LinearExploration
    from sudoku.configs import settings

    # make env
    if args.game == "sudoku":
        env = SudokuEnv(render_mode=args.render_mode)
    elif args.game == "snake":
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