from gym.envs.registration import register

register(id='nycmaze1-v1',
    entry_point='custom_env_dir.envs:NycMaze1',
)
