import gym
from gym import error, spaces, utils
from gym.utils import seeding 

class NycMaze1(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        print('Environment initialized')

        
    def step(self):
        print('Step successful!')
    def reset(self):
        print('Environment reset')
