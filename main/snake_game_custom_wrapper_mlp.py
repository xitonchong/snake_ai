import math
import time # For debugging.

import gym
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN # gym.discrete.Discrete(4)
        
        self.observation_space = gym.spaces.Box( #gym.spaces.Box
            low=-1, high=1,
            shape=(self.game.board_size, self.game.board_size),
            dtype=np.float32
        ) # 0: empty, 0.5: snake body, 1: snake head, -1: food

        self.board_size = board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs
    
    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)
            # return obs, reward * 0.1, self.done, info

            # Linear penalty decay.
            reward = info["snake_size"] - self.grid_size # (-max_growth, 0)
            return obs, reward * 0.1, self.done, info
        
        elif info["food_obtained"]: # food eaten
            # Reward on num_steps between getting food.
            reward = math.exp((self.grid_size - self.reward_step_counter) / self.grid_size) # (0, e)
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"] # No upper limit might enable the agent to master shorter scenario faster and more firmly.
            else:
                reward = - 1 / info["snake_size"]
            # print(reward*0.1)
            # time.sleep(1)

        # max_score: 144e - 1 = 390
        # min_score: -141 

        # Linear:
        # max_score: 288
        # min_score: -141

        reward = reward * 0.1 # Scale reward
        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: 0; SnakeBODY: 0.5; SnakeHEAD: 1; FOOD: -1;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)
        obs[tuple(self.game.snake[0])] = 1.0            
        obs[tuple(self.game.food)] = -1.0
        return obs
