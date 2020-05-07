import gym

class nycmaze1(gym.Env):

    def __init__(self):
        print('Environment initialized')
    def step(self):
        print('Step successful!')
    def reset(self):
        print('Environment reset')

    def build_maze(self):
      raise NotImplementedError

    def get_image(self):
        pass
        