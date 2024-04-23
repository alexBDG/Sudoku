# System imports.
import os
import warnings
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from collections import deque
from moviepy.editor import VideoFileClip

# Local imports.
from ..utils.general import Summarize
from ..utils.general import get_logger
from ..utils.replay_buffer import ReplayBuffer



class QN(object):
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # build model
        self.build()


    def build(self):
        """
        Build model
        """
        pass


    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)


    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass


    def initialize(self):
        """
        Initialize variables if necessary
        """
        pass


    def get_best_action(self, state):
        """
        Returns best action according to the network

        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError


    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]


    def update_target_params(self):
        """
        Update params of Q' with params of Q
        """
        pass


    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -1.
        self.max_reward = -1.
        self.std_reward = -1.

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = 0


    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards, initial=0.)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.state_history
        )
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()
        summarize = Summarize(
            file_path=self.config.summarize_output,
            total=self.config.nsteps_train,
        )

        episodes = 0 # time control of nb of elasped episodes
        t = last_eval = last_record = 0 # time control of nb of steps
        summarize.update_evaluation(0, reward=self.evaluate())

        pbar = tqdm(total=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state, _ = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, _, _ = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                pbar.update(1)
                summarize.update_step(
                    1,
                    reward=reward,
                    loss=loss_eval,
                    learning_rate=lr_schedule.epsilon,
                    epsilon=exp_schedule.epsilon,
                )
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    self.update_averages(
                        rewards, max_q_values, q_values, summarize.scores_eval
                    )
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    pbar.set_description(f"#{episodes} episodes")
                    if len(rewards) > 0:
                        pbar.set_postfix(
                            Loss=f"{loss_eval:.2E}",
                            AvgR=f"{self.avg_reward:.2f}",
                            MaxR=f"{np.max(rewards):.2f}",
                            eps=f"{exp_schedule.epsilon:.2E}",
                            Grads=f"{grad_eval:.2f}",
                            MaxQ=f"{self.max_q:.2f}",
                            lr=f"{lr_schedule.epsilon:.2E}"
                        )

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    pbar.set_description("Populating the memory")

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            episodes += 1
            summarize.update_episode(1, reward=total_reward)
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                summarize.update_evaluation(1, reward=self.evaluate())

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record(t=t, step_mode="test")
                summarize.plot()

        # last words
        pbar.close()
        self.logger.info("- Training done.")
        self.save()
        summarize.update_evaluation(1, reward=self.evaluate())
        summarize.plot()


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.state_history
        )
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state, _ = env.reset()
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, _, _ = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = f"Average reward: {avg_reward:04.2f} +/- {sigma_reward:04.2f}"
            self.logger.info(msg)

        return avg_reward


    def record(self, t=None, step_mode="train"):
        """
        Re create an env and record a video for one episode

        Args:
            t: (int) nths step
        """

        # Define a new instance of the same environment
        env = self.env.__class__(render_mode="rgb_array", step_mode=step_mode)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            env = gym.wrappers.RecordVideo(
                env, video_folder=self.config.record_path,
                episode_trigger=lambda x: True, name_prefix=f"step-{t}"
            )

        self.evaluate(env, 1)


    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record(t="start", step_mode="test")
            # Save as GIF
            video_name = os.path.join(
                self.config.record_path, f"step-start-episode-0"
            )
            videoClip = VideoFileClip(f"{video_name}.mp4")
            videoClip.write_gif(f"{video_name}.gif")

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record(t="end", step_mode="test")
            # Save as GIF
            video_name = os.path.join(
                self.config.record_path, f"step-end-episode-0"
            )
            videoClip = VideoFileClip(f"{video_name}.mp4")
            videoClip.write_gif(f"{video_name}.gif")
