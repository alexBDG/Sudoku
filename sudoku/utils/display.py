# System imports.
import numpy as np
import matplotlib.pyplot as plt


def plot_sudoku(grille, filename=None):
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.imshow(np.ones((9, 9)), cmap='Blues')
    axs.set_xticks([])
    axs.set_yticks([])

    for i in range(1, 9):
        linewidth = 2 if i % 3 == 0 else 0.5
        axs.axhline(i - 0.5, color='black', linewidth=linewidth)
        axs.axvline(i - 0.5, color='black', linewidth=linewidth)

    for i in range(9):
        for j in range(9):
            if grille[i][j] != 0:
                axs.text(
                    j, i+0.05, str(grille[i][j]),
                    ha='center', va='center', fontsize=18
                )

    for axis in ['top','bottom','left','right']:
        axs.spines[axis].set_linewidth(2)

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()
