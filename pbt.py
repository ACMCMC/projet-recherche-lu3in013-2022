import multiprocessing
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

class A2CAgent(salina.TAgent):
    # TAgent != TemporalAgent, TAgent is only an extension of the Agent interface to say that this agent accepts the current timestep parameter in the forward method
    r'''This agent implements an Advantage Actor-Critic agent (A2C)'''
    def __init__(self, observation_size, hidden_layer_size, action_size, stochastic=True):
        super().__init__()
        self.action_model = torch.nn.Sequential(
            torch.nn.Linear(observation_size, hidden_layer_size),
            torch.nn.ReLU(), # -> min(0, val)
            torch.nn.Linear(hidden_layer_size, action_size)
        )
        self.critic_model = torch.nn.Sequential(
            torch.nn.Linear(observation_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, 1) # Because we only want one output feature: the score of the action taken
        )

        self.stochastic = stochastic

    def forward(self, time, **kwargs):
        observation = self.get(('env/env_obs', time))
        scores = self.action_model(observation)
        probabilities = torch.softmax(scores, dim=-1)
        critic = self.critic_model(observation).squeeze(-1) # squeeze() removes the last dimension of the tensor

        if self.stochastic:
            action = torch.distributions.Categorical(probabilities).sample() # Create a statictical distribution, and take a sample from there (this will usually return the action we think is the best, but at times, it will return a different one)
        else:
            action = probabilities.argmax(1)

        self.set(('action', time), action)
        self.set(('action_probabilities', time), probabilities)
        self.set(('critic', time), critic)

class EnvironmentAgent(NoAutoResetGymAgent):
    def __init__(self, cfg, env):
        super().__init__(cfg, n_envs=1) # TODO: What is the number of environments?
        self.env = env

def make_env(cfg) -> gym.Env:
    # We set a timestep limit on the environment of max_episode_steps
    # We can also add a seed to the environment here
    return TimeLimit(gym.make(cfg.env.env_name), cfg.env.max_episode_steps)

def create_population(cfg):
    # We'll run this number of agents in parallel
    n_cpus = multiprocessing.cpu_count()

    # Create the required number of agents
    population = []
    for i in range(cfg.algorithm.population_size):
        # TODO: We can change the hyperparameters here (they're stored in cfg)
        #       We could also create another wrapper for the A2C agent which is in charge of changing them
        environment = make_env(cfg)
        # observation_size: the number of features of the observation (in Pendulum-v1, it is 3)
        observation_size = environment.observation_space.shape[0]
        # hidden_layer_size: the number of neurons in the hidden layer
        hidden_layer_size = cfg.algorithm.neural_network.hidden_layer_size
        # action_size: the number of parameters to output as actions (in Pendulum-v1, it is 1)
        action_size = environment.action_space.shape[0]
        # The agent that we'll train will use the A2C algorithm
        a2c_agent = A2CAgent(observation_size, hidden_layer_size, action_size)
        # To generate the observations, we need a gym agent, which will provide the values in 'env/env_obs'
        environment_agent = EnvironmentAgent(cfg, environment)
        temporal_agent = TemporalAgent(Agents(environment_agent, a2c_agent))
        temporal_agent.seed(cfg.algorithm.stochasticity_seed)
        async_agent = AsynchronousAgent(temporal_agent)
        population.append(async_agent)

    return population

def train(cfg, population):
    for epoch in range(cfg.algorithm.max_epochs):
        scores = []

        for agent in population:
            agent(time=0, stop_variable='env/done')

        agents_running = True
        while agents_running:
            agents_running = any([agent.is_running() for agent in population])
        
        # They have all finished executing
        print('Finished epoch {}'.format(epoch))


@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    population = create_population(cfg)
    train(cfg, population)

if __name__ == '__main__':
    main()