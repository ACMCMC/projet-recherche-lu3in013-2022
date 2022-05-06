import gym
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf
from gym_agents import AutoResetGymAgent, NoAutoResetGymAgent
from salina import get_class, get_arguments, instantiate_class
import my_gym

def make_gym_env(env_name):
    return gym.make(env_name)

class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    '''
    This agent will be used to perform the exploit part of the training.
    '''
    def __init__(self, cfg: OmegaConf):
        super().__init__(
            make_env_fn=get_class(cfg.env),
            make_env_args=get_arguments(cfg.env),
            n_envs=cfg.algorithm.number_environments
        )
        env = instantiate_class(cfg.env)
        env.seed(cfg.algorithm.stochasticity_seed)
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
            make_env_fn=get_class(cfg.env),
            make_env_args=get_arguments(cfg.env),
            n_envs=cfg.algorithm.number_environments
        )
        env = instantiate_class(cfg.env)
        env.seed(cfg.algorithm.stochasticity_seed)
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