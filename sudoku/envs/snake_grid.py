# System imports.
import pygame
import numpy as np
from enum import Enum
import gymnasium as gym
import pygame_chart as pyc
from collections import namedtuple

# Set seed
np.random.seed(0)



class Direction(Enum):
    NONE = 0
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

    @classmethod
    def get_action(self, idx):
        if idx == 0: return self.NONE
        elif idx == 1: return self.RIGHT
        elif idx == 2: return self.LEFT
        elif idx == 3: return self.UP
        elif idx == 4: return self.DOWN

    @classmethod
    def get_idx(self, action):
        if action == self.NONE: return 0
        elif action == self.RIGHT: return 1
        elif action == self.LEFT: return 2
        elif action == self.UP: return 3
        elif action == self.DOWN: return 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Environment variables
N_ACTIONS = 1 + 4
OBS_SHAPE = (18, 18, 1)


class Discrete(object):
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.random.randint(0, self.shape)


class Box(object):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        # Create a grid for data
        data = np.random.randint(
            low=self.low, high=self.high+1, size=self.shape,
        ).astype(self.dtype)

        return data


class SnakeEnv(gym.Env):
    """A fonction approximation environment for OpenAI gym"""
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "step_modes": ["train", "test"],
        "render_fps": 30
    }
    name = "snake"

    def __init__(self, render_mode="human", step_mode="train",
                 dtype=np.float32):
        self.window_width = 513  # The size of the PyGame window
        self.window_heigh = 513  # The size of the PyGame window
        self.step_mode = step_mode
        self.dtype = dtype

        self.current_step = 0
        self.all_rewards = []
        self.cumulative_reward = 0
        self.food = None
        self._init_game_state()
        self._place_food()

        # Actions
        self.action_space = Discrete(N_ACTIONS)

        # Observations
        self.observation_space = Box(
            low=-1, high=1, shape=OBS_SHAPE, dtype=self.dtype
        )

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.window = None
        self.clock = None


    def _init_game_state(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(int(OBS_SHAPE[0]/2), int(OBS_SHAPE[0]//2))
        self.snake = [
            self.head,
            Point(self.head.x-1, self.head.y),
            Point(self.head.x-2, self.head.y)
        ]

        self.food = None


    def _place_food(self):
        x = np.random.randint(0, OBS_SHAPE[0])
        y = np.random.randint(0, OBS_SHAPE[1])
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def _next_observation(self):
        grid = np.zeros(OBS_SHAPE)
        # Place snake
        for pt in self.snake:
            if (
                (0 <= pt.x and pt.x < OBS_SHAPE[0]) and
                (0 <= pt.y and pt.y < OBS_SHAPE[1])
            ):
                grid[pt.x, pt.y] = 1
        # Place food
        grid[self.food.x, self.food.y] = -1
        return grid


    def _compute_reward(self, terminated):
        reward = 0
        if self.head == self.food:
            reward += 1

        if terminated:
            reward -= 1

        self.cumulative_reward += reward
        self.all_rewards.append(reward)

        return reward


    def step(self, action):
        self.current_step += 1

        # 2. move
        self._move(action) # update the head

        # 3. check if game over
        terminated = False
        if self._is_collision():
            terminated = True

        reward = self._compute_reward(terminated)

        # Remove last snake part or add new food
        if self.head == self.food:
            self._place_food()
        else:
            self.snake.pop()

        obs = self._next_observation()

        # 6. return game over and score
        return obs, reward, terminated, False, {}


    def _is_collision(self):
        # hits boundary
        if (
            (self.head.x < 0) or (OBS_SHAPE[0] <= self.head.x) or
            (self.head.y < 0) or (OBS_SHAPE[1] <= self.head.y)
        ):
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False


    def _move(self, action):
        if isinstance(action, (int, np.int64)):
            action = Direction.get_action(action)

        previous_direction = self.direction
        if action != Direction.NONE:
            self.direction = action

        x = self.head.x
        y = self.head.y

        # # Cannot go back, only left, right and forward (in the snake referencial)
        # if (
        #     (previous_direction == Direction.RIGHT) and
        #     (self.direction == Direction.LEFT)
        # ) or (
        #     (previous_direction == Direction.UP) and
        #     (self.direction == Direction.DOWN)
        # ):
        #     return 0

        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)
        self.snake.insert(0, self.head)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.all_rewards = []
        self.cumulative_reward = 0
        self._init_game_state()
        self._place_food()

        obs = self._next_observation()

        return obs, {}


    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (2 * self.window_width, self.window_heigh + 100)
            )
            pygame.display.set_caption('Snake')
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        case_width = self.window_width // OBS_SHAPE[0]
        case_heigh = self.window_heigh // OBS_SHAPE[1]

        canvas = pygame.Surface(
            (2 * self.window_width, self.window_heigh + 100)
        )
        canvas.fill(BLACK)

        for pt in self.snake:
            rect = pygame.Rect(
                pt.x*case_width, pt.y*case_heigh, case_width, case_heigh
            )
            pygame.draw.rect(canvas, BLUE1, rect)
            rect = pygame.Rect(
                pt.x*case_width+4, pt.y*case_heigh+4, 12, 12
            )
            pygame.draw.rect(canvas, BLUE2, rect)

        rect = pygame.Rect(
            self.food.x*case_width, self.food.y*case_heigh,
            case_width, case_heigh
        )
        pygame.draw.rect(canvas, RED, rect)

        font = pygame.font.SysFont('arial', 25)
        text = font.render(f"Score: {self.cumulative_reward}", True, WHITE)
        canvas.blit(text, [0, 0])

        # Separation line
        pygame.draw.line(
            canvas, WHITE,
            (0, self.window_heigh), (self.window_width, self.window_heigh),
            self.window_width % OBS_SHAPE[0]
        )
        pygame.draw.line(
            canvas, WHITE,
            (self.window_width, 0), (self.window_width, self.window_heigh),
            self.window_width % OBS_SHAPE[0]
        )

        # Add chart
        figure = pyc.Figure(
            canvas,
            self.window_width+25, 25,
            self.window_width-50, self.window_heigh-50,
            bg_color=WHITE
        )
        figure.line("", [0, 1], [0, 1], color=WHITE)
        figure.add_title("Reward evolution")
        figure.add_legend()
        figure.add_xaxis_label("Steps")
        figure.add_yaxis_label("Reward")
        if self.current_step > 0:
            x = np.arange(len(self.all_rewards)).tolist()
            y = np.array(self.all_rewards)
            figure.scatter("Step Reward", x, y.tolist())
            y = np.cumsum(self.all_rewards)
            figure.line("Cumulative Reward", x, y.tolist())
        figure.draw()
        canvas.blit(figure.background, (self.window_width+25, 25))

        draw_arrows(
            canvas, action=self.direction,
            offset=(self.window_width/2, self.window_heigh+50)
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the
            # visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined
            # framerate. The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def draw_arrows(canvas, action, offset):
        arrow_points_up = [
            (0, -20), (10, 0), (5, 0), (5, 20), (-5, 20), (-5, 0), (-10, 0)
        ]
        arrow_points_right = [
            (20, 0), (0, -10), (0, -5), (-20, -5), (-20, 5), (0, 5), (0, 10)
        ]
        arrow_points_left = [
            (-20, 0), (0, -10), (0, -5), (20, -5), (20, 5), (0, 5), (0, 10)
        ]
        arrow_points_down = [
            (0, 20), (10, 0), (5, 0), (5, -20), (-5, -20), (-5, 0), (-10, 0)
        ]
        pygame.draw.polygon(
            canvas, RED if action == Direction.RIGHT else WHITE,
            [(x + offset[0] + 25, y + offset[1]) for (x, y) in arrow_points_right]
        )
        pygame.draw.polygon(
            canvas, RED if action == Direction.LEFT else WHITE,
            [(x + offset[0] + 75, y + offset[1]) for (x, y) in arrow_points_left]
        )
        pygame.draw.polygon(
            canvas, RED if action == Direction.UP else WHITE,
            [(x + offset[0] - 25, y + offset[1]) for (x, y) in arrow_points_up]
        )
        pygame.draw.polygon(
            canvas, RED if action == Direction.DOWN else WHITE,
            [(x + offset[0] - 75, y + offset[1]) for (x, y) in arrow_points_down]
        )


def play(fps=None, store=False):
    game = SnakeEnv(render_mode="human")

    if isinstance(fps, int):
        game.metadata["render_fps"] = fps

    if store:
        try:
            from .utils import EpisodeBuffer
        except ImportError:
            from utils import EpisodeBuffer

    # game loop
    while True:
        state, _ = game.reset()
        if store: episode_buffer = EpisodeBuffer(name=game.name)
        while True:
            game.render()
            if store: episode_buffer.store_frame(state)

            # Collect user input
            action = Direction.NONE
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        action = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        action = Direction.DOWN
                    elif event.key == pygame.K_q:
                        quit()

            state, reward, done, _, _ = game.step(action)
            if store:
                episode_buffer.store_effect(
                    Direction.get_idx(action), reward, done
                )
            if done:
                if store: episode_buffer.save()
                break



if __name__ == '__main__':
    play()