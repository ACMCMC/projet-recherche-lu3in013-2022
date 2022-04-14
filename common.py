
import multiprocessing
from typing import Dict

import gym
import my_gym
from omegaconf import OmegaConf
from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent
from salina.agent import Agent
import torch

def get_cumulated_reward(cumulated_rewards_dict: Dict[Agent, torch.Tensor], agent: Agent):
    cumulated_rewards_of_agent = cumulated_rewards_dict[agent]
    return torch.mean(cumulated_rewards_of_agent)

class EnvAgent(AutoResetGymAgent):
    def __init__(self, cfg: OmegaConf):
        super().__init__(
            get_class(cfg.env),
            get_arguments(cfg.env),
            n_envs=cfg.algorithm.number_environments
        )
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del(env)

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
