# System imports.
import os
import time
import numpy as np



class EpisodeBuffer(object):
    def __init__(self, name) -> None:
        self.obs = None
        self.action = []
        self.reward = []
        self.done = []
        self.steps = 0
        self.name = name
        self.time_stamp = time.time()

    def store_frame(self, frame):
        obs = np.expand_dims(frame, axis=0)
        if self.steps == 0:
            self.obs = obs
        else:
            self.obs = np.vstack([self.obs, obs])
        self.steps += 1

    def store_effect(self, action, reward, done):
        self.action = np.append(self.action, action)
        self.reward = np.append(self.reward, reward)
        self.done = np.append(self.done, done)

    def save(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        total_reward = float(np.sum(self.reward))
        filename = (
            f"{self.name} t={self.time_stamp:.0f}s "
            f"reward={total_reward:.0f} steps={self.steps}"
        )
        np.savez(
            os.path.join("data", filename),
            obs=self.obs, action=self.action, reward=self.reward, done=self.done
        )
