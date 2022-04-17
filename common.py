from typing import Dict

import gym
import my_gym
from salina.agent import Agent
import torch

def get_cumulated_reward(cumulated_rewards_dict: Dict[Agent, torch.Tensor], agent: Agent):
    cumulated_rewards_of_agent = cumulated_rewards_dict[agent]
    return torch.mean(cumulated_rewards_of_agent)