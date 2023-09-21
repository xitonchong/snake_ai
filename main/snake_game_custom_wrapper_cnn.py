import math 

import gym 
import numpy as np 

from snake_game import SnakeGame 

class SnakeEnv(gym.Env): 
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True): 
        super().__init__()
        self.game =  SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode) 
        self.game.reset() 

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(4) 

        self.observation_space = gym.spaces.Box( 
            low=0, high=255, 
            shape=(84, 84, 3), 
            dtype=np.uint8
        )

        self.board_size = board_size 
        self.grid_size = board_size ** 2 # max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake) 
        self.max_growth = self.grid_size - self.init_snake_size 

        self.done = False 

        if limit_step: 
            self.step_limit = self.grid_size * 4 # more than enough steps to get the food. 
        else: 
            self.step_limit = 1e9 
        self.reward_step_counter = 0 


    def reset(self): 
        self.game.reset() 

        self.done = False 
        self.reward_step_counter = 0 

        obs = self._generate_observation() 
        return obs
    

    def step(self, action): 
        self.done, info = self.game.step(action) #
        obs = self._generate_observation() 

        reward = 0.0 
        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size: # snake fills up the entire board. game over. 
            reward = self.max_growth * 0.1 # victory reward 
            self.done = True 
            if not self.silent_mode: 
                self.game.sound_victory.play() 
            return obs, reward, self.done, info 
        
        if self.reward_step_counter > self.step_limit: # step limit reached, game over 
            self.reward_step_counter = 0 
            self.done = True 

        if self.done: # snake bumps into wall or itself, episode is over
            # game over penalty is based on snake size 
            reward = -math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth)
            reward = reward * 0.1
            return obs, reward, self.done, info 
        
        else: 
            # give a tiny reward/penalty to the agent based on whether it is heading 
            # towards the 
            # not competing with game over penalty or the food eaten reward
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else: 
                reward = -1 / info["snake_size"]
            
        return obs, reward, self.done, info 
    

    def render(self): 
        self.game.render() 


    def get_action_mask(self): 
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # check if the action is against the current direction of the snake or is ending the game 
    def _check_action_validity(self, action): 
        current_direction = self.game.direction 
        snake_list = self.game.snake 
        row, col = snake_list[0]
        if action == 0 : # UP
            if current_direction == "DOWN":
                return False 
            else: 
                row -= 1
            
        elif action == 1: #LEFT 
            if current_direction == "RIGHT":
                return False 
            else: 
                col -= 1 
        
        elif action == 2: #RIGHT 
            if current_direction == "LEFT": 
                return False 
            else: 
                col += 1

        elif action == 3: # DOWN
            if current_direction == "UP": 
                return False 
            else: 
                row += 1

        # check if sanke collided with itself or the wall. Note that the tail of the snake would be 
        if (row, col) == self.game.food: 
            game_over = (
                (row, col) in snake_list # the snake won't pop the last cell if it ate food. 
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
        
    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self): 
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 500, len(self.game.snake), dtype=np.uint8)

        # stack single layer into 3-channel-image. 
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # set the snake head to green and the tail to blue 
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # set the food to red
        obs[self.game.food] = [0, 0, 255]

        # enlarge the observation to 84x84, 7*12
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs
        