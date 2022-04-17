import gym
from omegaconf import OmegaConf
from salina.agents.gyma import AutoResetGymAgent, NoAutoResetGymAgent
from salina import get_class, get_arguments, instantiate_class

class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    '''
    This agent will be used to perform the exploit part of the training.
    '''
    def __init__(self, cfg: OmegaConf):
        super().__init__(
            get_class(cfg.env),
            get_arguments(cfg.env),
            n_envs=cfg.algorithm.number_environments
        )
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env
    
    def is_action_space_continuous(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def get_observation_size(self):
        if isinstance(self.observation_space, gym.spaces.Box):
            return self.observation_space.shape[0]
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            return self.observation_space.n
        else:
            ValueError("Incorrect space type")

    def get_action_size(self):
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        else:
            ValueError("Incorrect space type")



class AutoResetEnvAgent(AutoResetGymAgent):
    '''
    This agent will be used to perform the explore part of the training.
    '''
    def __init__(self, cfg: OmegaConf):
        super().__init__(
            get_class(cfg.env),
            get_arguments(cfg.env),
            n_envs=cfg.algorithm.number_environments
        )
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_action_space_continuous(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def get_observation_size(self):
        if isinstance(self.observation_space, gym.spaces.Box):
            return self.observation_space.shape[0]
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            return self.observation_space.n
        else:
            ValueError("Incorrect space type")

    def get_action_size(self):
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        else:
            ValueError("Incorrect space type")