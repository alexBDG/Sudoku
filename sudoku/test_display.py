# System imports.
import os

# Local imports.
from utils.display import plot_sudoku
from configs.linear import config
from configs.sudoku_samples import GRID
from configs.sudoku_samples import FULL_GRID


if __name__ == "__main__":
    path = os.path.join(config.output_path, "test_grid.png")
    plot_sudoku(GRID.reshape(9, 9), path)