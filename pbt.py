import time
from turtle import forward
import numpy as np
import copy # used for multiprocessing

import gym
from gym.wrappers import TimeLimit
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

import salina
from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent, NRemoteAgent
from salina.agents.asynchronous import AsynchronousAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent
from omegaconf import DictConfig, OmegaConf

class A2CAgent(TemporalAgent):
    def __init__(self, observation_size, hidden_layer_size, action_size):
        self.action_model = torch.nn.Sequential(
        torch.nn.Linear(observation_size, hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, action_size)
        )

    def forward(self, time, **kwargs):
        observation = self.get(('env/observation', time))
        scores = self.action_model(observation)
        probabilities = torch.softmax(scores, dim=-1)

def make_env(cfg) -> gym.Env:
    # We set a timestep limit on the environment of max_episode_steps
    return TimeLimit(gym.make(cfg.env.env_name), cfg.env.max_episode_steps)

def train(cfg):
    # Create the required number of agents
    population = []
    for i in range(cfg.algorithm.population_size):
        environment = make_env(cfg)
        # observation_size: the number of features of the observation (in Pendulum-v1, it is 3)
        observation_size = environment.observation_space.shape[0]
        # hidden_layer_size: the number of neurons in the hidden layer
        hidden_layer_size = cfg.algorithm.neural_network.hidden_layer_size
        # action_size: the number of parameters to output as actions (in Pendulum-v1, it is 1)
        action_size = environment.action_space.shape[0]
        agent = A2CAgent(observation_size, hidden_layer_size, action_size)


@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    # Create the workspace
    pass