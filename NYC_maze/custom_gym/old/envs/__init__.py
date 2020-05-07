from gym.envs.registration import register

register(id='nycmaze1-v0',
    entry_point='envs.custom_env_dir:nycmaze1'
)