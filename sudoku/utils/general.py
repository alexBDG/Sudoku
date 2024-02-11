# System imports.
import logging
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt


def export_plot(ys, ylabel, filename):
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(logging.WARNING) 
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class Summarize(object):
    def __init__(self, file_path: str, total: int) -> None:
        self.file_path = file_path
        self.total = total
        self.step = 0
        self.episode = 0
        self._idx_steps = np.full((self.total), fill_value=np.nan)
        self._idx_episodes = np.full((self.total), fill_value=np.nan)
        self._step_loss = np.full((self.total), fill_value=np.nan)
        self._step_reward = np.full((self.total), fill_value=np.nan)
        self._episode_reward = np.full((self.total), fill_value=np.nan)
        self._step_epsilon = np.full((self.total), fill_value=np.nan)
        self._step_learning_rate = np.full((self.total), fill_value=np.nan)

    def update(self, n: int, reward: float, loss: float, learning_rate: float,
               epsilon: float) -> None:
        self._idx_steps[self.step] = self.step
        self._step_loss[self.step] = loss
        self._step_reward[self.step] = reward
        self._step_epsilon[self.step] = epsilon
        self._step_learning_rate[self.step] = learning_rate
        self.step += n

    def update_episode(self, episode: int, reward: float) -> None:
        self._idx_episodes[self.episode] = self.episode
        self._episode_reward[self.episode] = reward
        self.episode += episode

    def plot(self) -> None:
        fig, axs = plt.subplots(
            3, 1, constrained_layout=True, figsize=(2*6.8, 2*4.6)
        )

        # Filter only filled values
        idx = ~np.isnan(self._idx_episodes)

        # Episodes
        axs[0].scatter(self._idx_episodes[idx], self._episode_reward[idx])
        axs[0].set_ylabel("Total Reward")
        axs[0].set_xlabel("Episodes")

        # Filter only filled values
        idx = ~np.isnan(self._idx_steps)

        # Reward/Loss
        axs[1].scatter(
            self._idx_steps[idx], self._step_reward[idx],
            c="red"
        )
        axs[1].set_ylabel("Average Reward", color="red")
        axs[1].set_xlabel("Steps")
        axs[1].tick_params(axis="y", labelcolor="red")
        axs[1].spines['left'].set_color('red')
        ax1 = axs[1].twinx()
        ax1.scatter(
            self._idx_steps[idx], self._step_loss[idx],
            c="blue"
        )
        ax1.set_yscale("log")
        ax1.spines['right'].set_color('blue')
        ax1.set_ylabel("Loss", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Epsilon/Learning rate
        axs[2].scatter(
            self._idx_steps[idx], self._step_epsilon[idx],
            c="red"
        )
        axs[2].set_ylabel("Epsilon", color="red")
        axs[2].set_xlabel("Steps")
        axs[2].tick_params(axis='y', labelcolor="red")
        axs[2].spines['left'].set_color('red')
        ax2 = axs[2].twinx()
        ax2.scatter(
            self._idx_steps[idx], self._step_learning_rate[idx],
            c="blue"
        )
        ax2.spines['right'].set_color('blue')
        ax2.set_ylabel("Learning Rate", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        fig.savefig(self.file_path)
        plt.close(fig)
