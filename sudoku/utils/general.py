# System imports.
import logging
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt



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
        # Steps
        self.step = 0
        self._idx_step = np.full((self.total+1), fill_value=np.nan)
        self._step_loss = np.full((self.total+1), fill_value=np.nan)
        self._step_reward = np.full((self.total+1), fill_value=np.nan)
        self._step_epsilon = np.full((self.total+1), fill_value=np.nan)
        self._step_learning_rate = np.full((self.total+1), fill_value=np.nan)
        # Episodes
        self.episode = 0
        self._idx_episode = np.full((self.total+1), fill_value=np.nan)
        self._episode_reward = np.full((self.total+1), fill_value=np.nan)
        # Evaluations
        self.evaluation = 0
        self.scores_eval = []
        self._idx_evaluation = np.full((self.total+1), fill_value=np.nan)
        self._evaluation_reward = np.full((self.total+1), fill_value=np.nan)

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
        self._idx_episode[self.episode] = self.episode
        self._episode_reward[self.episode] = reward

    def update_evaluation(self, n: int, reward: float) -> None:
        self.evaluation += n
        self.scores_eval.append(reward)
        self._idx_evaluation[self.episode] = self.episode
        self._evaluation_reward[self.episode] = reward

    def plot(self) -> None:
        fig, axs = plt.subplots(
            3, 1, constrained_layout=True, figsize=(2*6.8, 2*4.6)
        )

        # Filter only filled values
        idx = ~np.isnan(self._idx_episode)

        # Episodes
        axs[0].scatter(
            self._idx_episode[idx], self._episode_reward[idx],
            c="tab:blue", label="Training", s=5.
        )
        axs[0].set_ylabel("Total Reward")
        axs[0].set_xlabel("Episodes")

        # Filter only filled values
        idx = ~np.isnan(self._idx_evaluation)
        axs[0].plot(
            self._idx_evaluation[idx], self._evaluation_reward[idx],
            color="tab:red", ls="-", marker="x", label="Evaluation", lw=3.
        )
        axs[0].legend(loc="upper left")

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
        ax2.set_yscale("log")
        ax2.spines['right'].set_color('tab:blue')
        ax2.set_ylabel("Learning Rate", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        fig.savefig(self.file_path)
        plt.close(fig)
