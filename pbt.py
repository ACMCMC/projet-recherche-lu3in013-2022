import multiprocessing
import time
from turtle import forward
from typing import Union
from typing_extensions import Self
import numpy as np
import copy # used for multiprocessing

import gym
from gym.wrappers import TimeLimit
import copy
import time
import omegaconf

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

def sort_performance(agents_list):
    pass

def select_pbt(portion, agents_list):
    random_index = torch.distributions.Uniform(0, portion * len(agents_list)).sample()
    return agents_list[random_index]

class EnvAgent(GymAgent):
    def __init__(self, cfg: OmegaConf):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.number_environments
        )
        self.env = instantiate_class(cfg.algorithm.env)

    def get_observation_size(self):
        if self.observation_space.isinstance(gym.spaces.Box):
            return self.observation_space.shape[0]
        elif self.observation_space.isinstance(gym.spaces.Discrete):
            return self.observation_space.n
        else:
            ValueError("Incorrect space type")

    def get_action_size(self):
        if self.action_space.isinstance(gym.spaces.Box):
            return self.action_space.shape[0]
        elif self.action_space.isinstance(gym.spaces.Discrete):
            return self.action_space.n
        else:
            ValueError("Incorrect space type")

class A2CAgent(salina.TAgent):
    # TAgent != TemporalAgent, TAgent is only an extension of the Agent interface to say that this agent accepts the current timestep parameter in the forward method
    r'''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''

    def __init__(self, parameters, observation_size, hidden_layer_size, action_size, stochastic=True):
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

        self.observation_size = observation_size
        self.hidden_layer_size = hidden_layer_size
        self.action_size = action_size
        self.stochastic = stochastic
        self.params = omegaconf.DictConfig(content=parameters)

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
    
    def compute_critic_loss(self, reward, done, critic) -> Union[float, float]:
        # Compute temporal difference
        target = reward[1:] + self.discount_factor * critic[1:].detach() * (1 - done[1:].float())
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()
        return critic_loss, td
    
    def compute_a2c_loss(self, action_probs, action, td):
        action_logp = _index(action_probs, action)
        a2c_loss = action_logp[:-1] * td.detach()
        return a2c_loss.mean()
    
    def get_hyperparameter(self, param_name):
        return self.params.param_name

    def set_hyperparameter(self, param_name, value):
        self.params.param_name = value

    def clone(self):
        new = A2CAgent(self.parameters, self.observation_size, self.hidden_layer_size, self.action_size, self.stochastic)
        return new

class A2CParameterizedAgent(salina.TAgent):
    # TAgent != TemporalAgent, TAgent is only an extension of the Agent interface to say that this agent accepts the current timestep parameter in the forward method
    r'''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''


    def __init__(self, parameters, observation_size, hidden_layer_size, action_size, mutation_rate, stochastic=True):
        super().__init__()

        self.a2c_agent = A2CAgent(parameters=simplified_parameters, observation_size=observation_size, hidden_layer_size=hidden_layer_size, action_size=action_size, stochastic=stochastic)

        self.mutation_rate = mutation_rate
        
        simplified_parameters = omegaconf.DictConfig(content={}) # The A2C Agent only sees a dictionary of (param_name, param_value) entries
        self.params_metadata = omegaconf.DictConfig(content={}) # This wrapper will store the metadata for the parameters of the A2C agent, so it knows how to change them when needed
        for param in parameters:
            generated_val = torch.distributions.Uniform(parameters[param].min, parameters[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            self.params_metadata.param = {'min': parameters[param].min, 'max': parameters[param].max}
            simplified_parameters.param = generated_val
        
    def mutate_hyperparameters(self):
        r'''This function mutates, randomly, all the hyperparameters of this agent, according to the mutation rate'''
        for param in self.params:
            # We'll generate a completely random value, and mutate the original one according to the mutation rate.
            old_val = self.a2c_agent.get_hyperparameter(param)
            generated_val = torch.distributions.Uniform(self.params[param].min, self.params[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            mutated_val = (1.0 - self.mutation_rate) * old_val + self.mutation_rate * generated_val # For example, 0.8 * old_val + 0.2 * mutated_val
            self.a2c_agent.set_hyperparameter(param, mutated_val)
    
    def get_agent(self):
        return self.a2c_agent

    def copy(self, other: Self):
        self.a2c_agent = other.get_agent().clone()


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
        a2c_agent = A2CAgent(cfg.algorithm.hyperparameters, observation_size, hidden_layer_size, action_size, cfg.algorithm.mutation_rate)
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