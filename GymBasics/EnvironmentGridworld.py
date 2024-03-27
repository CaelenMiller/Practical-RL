import gym
from gym import spaces
import numpy as np



class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N) # N is the number of actions

        # Example for using image as input (observation space):
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        # Initialize state or observation
        self.state = None


    def step(self, action):
        """
        Apply the action to the environment and step the environment forward.
        - action: the action to be executed in the environment
        Returns:
        - observation (Np Array): agent's observation of the current environment
        - reward (float) : amount of reward returned after previous action
        - done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        - info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Execute one time step within the environment
        # Update self.state
        # Calculate reward
        # Determine if the episode is done
        # Optionally include additional info
        return observation, reward, done, info
    

    def reset(self):
        """
        Reset the state of the environment to an initial state and returns an initial observation.
        Returns:
        - observation (object): the initial observation.
        """
        # Reset the state of the environment to an initial state
        # Return the initial observation
        return observation
    

    def render(self, mode='human', close=False):
        """
        Render the environment.
        The set of supported modes varies per environment. (And some environments do not support rendering at all.)
        """
        # Implement rendering logic here
        pass


    def close(self):
        """
        Perform any necessary cleanup.
        """
        # Close and clean up resources
        pass


import pygame
import sys

class Display:
    def __init__(self, initial_grid, size=(500, 500)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.size = size
        self.grid = initial_grid

        self.clock = pygame.time.Clock()
        self.FPS = 3

        # Generate a pygame color for each of the integers in the array
        self.colors = [
            pygame.Color(50, 50, 50),  # 0
            pygame.Color(192, 192, 192),  # 1
        ]
        self.agent_color = pygame.Color(255,50,50)
        # Determine the size of the rectangles needed to fit all the squares on the screen
        self.rect_size = min(self.size) // max(self.grid.shape)

    def render(self, grid, agent_pos=[0,0]):
        self.grid = grid
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill((0, 0, 0))  # Fill the background with black

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Determine the color based on the integer in the grid
                color = self.colors[self.grid[i, j]]
                # Draw the rectangle
                pygame.draw.rect(self.screen, color, 
                                 (j*self.rect_size, i*self.rect_size, self.rect_size, self.rect_size))
                
        pygame.draw.rect(self.screen, self.agent_color, 
                                 (agent_pos[0]*self.rect_size + 3, agent_pos[1]*self.rect_size + 3, self.rect_size - 6, self.rect_size - 6))

        pygame.display.flip()
