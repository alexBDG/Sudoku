# System imports.
import logging
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import LinearLocator



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
        empty_array = np.hstack([0, np.full((self.total), fill_value=np.nan)])
        # Steps
        self.step = 0
        self._idx_step = empty_array.copy()
        self._step_loss = empty_array.copy()
        self._step_reward = empty_array.copy()
        self._step_epsilon = empty_array.copy()
        self._step_learning_rate = empty_array.copy()
        # Episodes
        self.episode = 0
        self._idx_episode = empty_array.copy()
        self._episode_reward = empty_array.copy()
        # Evaluations
        self.evaluation = 0
        self.scores_eval = []
        self._evaluation_reward = empty_array.copy()

    def update_step(self, n: int, reward: float, loss: float,
                    learning_rate: float, epsilon: float) -> None:
        self.step += n
        self._idx_step[self.step] = self.step
        self._step_loss[self.step] = loss
        self._step_reward[self.step] = reward
        self._step_epsilon[self.step] = epsilon
        self._step_learning_rate[self.step] = learning_rate

    def update_episode(self, n: int, reward: float) -> None:
        self.episode += n
        self._idx_episode[self.step] = self.episode
        self._episode_reward[self.step] = reward

    def update_evaluation(self, n: int, reward: float) -> None:
        self.evaluation += n
        self.scores_eval.append(reward)
        self._evaluation_reward[self.step] = reward

    def _step_to_episode(self) -> np.ndarray:
        nan_mask = ~np.isnan(self._idx_episode)
        xp = np.arange(self._idx_episode.shape[0])[nan_mask]
        fp = self._idx_episode[nan_mask]

        def get_episode(step: np.ndarray, xp: np.ndarray=xp,
                        fp: np.ndarray=fp) -> np.ndarray:
            episode = np.interp(step, xp, fp)
            return episode

        return get_episode

    def _episode_to_step(self) -> np.ndarray:
        nan_mask = ~np.isnan(self._idx_episode)
        fp, xp = np.unique(self._idx_episode[nan_mask], return_index=True)

        def get_step(episode: np.ndarray, xp: np.ndarray=xp,
                     fp: np.ndarray=fp) -> np.ndarray:
            step = np.interp(episode, xp, fp)
            return step

        return get_step

    def plot(self) -> None:
        fig, axs = plt.subplots(
            3, 1, constrained_layout=True, figsize=(2*6.8, 2*4.6)
        )

        # Episodes
        # Filter only filled values
        idx = ~np.isnan(self._episode_reward)
        axs[0].scatter(
            self._idx_step[idx], self._episode_reward[idx],
            c="tab:blue", label="Training", s=5.
        )
        axs[0].set_ylabel("Total Reward")
        axs[0].set_xlabel("Episodes")

        # Filter only filled values
        idx = ~np.isnan(self._evaluation_reward)
        axs[0].plot(
            self._idx_step[idx], self._evaluation_reward[idx],
            color="tab:red", ls="-", marker="x", label="Evaluation", lw=3.
        )

        # axs[0].set_xscale(
        #     "function",
        #     functions=(self._step_to_episode(), self._episode_to_step())
        # )
        # axs[0].margins(x=0.)

        # Filter only filled values
        idx = ~np.isnan(self._idx_step)

        # Reward/Loss
        axs[1].scatter(
            self._idx_step[idx], self._step_reward[idx],
            c="tab:red", s=5.
        )
        axs[1].set_ylabel("Average Reward", color="tab:red")
        axs[1].set_xlabel("Steps")
        axs[1].tick_params(axis="y", labelcolor="tab:red")
        axs[1].spines['left'].set_color('tab:red')
        ax1 = axs[1].twinx()
        ax1.scatter(
            self._idx_step[idx], self._step_loss[idx],
            c="tab:blue", s=5.
        )
        ax1.set_yscale("log")
        ax1.spines['right'].set_color('tab:blue')
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        # axs[1].margins(x=0.)

        # Epsilon/Learning rate
        axs[2].scatter(
            self._idx_step[idx], self._step_epsilon[idx],
            c="tab:red", s=5.
        )
        axs[2].set_ylabel("Epsilon", color="tab:red")
        axs[2].set_xlabel("Steps")
        axs[2].tick_params(axis='y', labelcolor="tab:red")
        axs[2].spines['left'].set_color('tab:red')
        ax2 = axs[2].twinx()
        ax2.scatter(
            self._idx_step[idx], self._step_learning_rate[idx],
            c="tab:blue", s=5.
        )
        ax2.spines['right'].set_color('tab:blue')
        ax2.set_ylabel("Learning Rate", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        # axs[2].margins(x=0.)

        fig.savefig(self.file_path)
        plt.close(fig)
