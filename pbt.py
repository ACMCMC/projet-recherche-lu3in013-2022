import multiprocessing
import time
from turtle import forward
from typing import Dict, List, Union
from typing_extensions import Self
import numpy as np
import copy # used for multiprocessing
import random

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


class EnvAgent(GymAgent):
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
    '''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''

    def __init__(self, parameters, observation_size, hidden_layer_size, action_size, stochastic=True, std_param=None):
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
        if std_param is None:
            self.std_param = nn.parameter.Parameter(torch.randn(action_size,1)) # TODO: What is this? Should we copy it too?
        else:
            self.std_param = std_param.clone()

    def forward(self, time, **kwargs):
        input = self.get(("env/env_obs", time))
        mean = torch.tanh(self.action_model(input))
        dist = torch.distributions.Normal(mean, torch.nn.Softplus()(self.std_param))
        # dist = Normal(mean, torch.exp(self.std_param))
        self.set(("entropy", time), dist.entropy())
        if self.stochastic:
            action = dist.sample()
        else : 
            action = mean 
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", time), action)
        self.set(("action_logprobs", time), logp_pi)

    def compute_critic_loss(self, reward, done, critic) -> Union[float, float]:
        # Compute temporal difference
        target = reward[1:] + self.discount_factor * critic[1:].detach() * (1 - done[1:].float())
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()
        return critic_loss, td
    
    @staticmethod
    def _index_3d_tensor_with_2d_tensor(tensor_3d: torch.Tensor, tensor_2d: torch.Tensor):
        # TODO: What is the purpose of this function?
        x, y, z = tensor_3d.size()
        t = tensor_3d.reshape(x*y, z)
        tt = tensor_2d.reshape(x*y)
        v = t[torch.arrange(x*y), tt]
        v = v.reshape(x, y)
        return v
    
    def compute_a2c_loss(self, action_probs: torch.Tensor, action: torch.Tensor, td: float) -> float:
        action_logp = self._index_3d_tensor_with_2d_tensor(action_probs, action)
        a2c_loss = action_logp[:-1] * td.detach()
        return a2c_loss.mean()

    def get_hyperparameter(self, param_name):
        return self.params[param_name]

    def set_hyperparameter(self, param_name, value):
        self.params[param_name] = value

    def clone(self):
        new = A2CAgent(self.params, self.observation_size, self.hidden_layer_size, self.action_size, self.stochastic, std_param=self.std_param)
        return new

class A2CParameterizedAgent(salina.TAgent):
    # TAgent != TemporalAgent, TAgent is only an extension of the Agent interface to say that this agent accepts the current timestep parameter in the forward method
    '''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''


    def __init__(self, parameters, observation_size, hidden_layer_size, action_size, mutation_rate, stochastic=True):
        super().__init__()

        self.mutation_rate = mutation_rate
        
        simplified_parameters = omegaconf.DictConfig(content={}) # The A2C Agent only sees a dictionary of (param_name, param_value) entries
        self.params_metadata = omegaconf.DictConfig(content={}) # This wrapper will store the metadata for the parameters of the A2C agent, so it knows how to change them when needed
        for param in parameters:
            generated_val = torch.distributions.Uniform(parameters[param].min, parameters[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            self.params_metadata.param = {'min': parameters[param].min, 'max': parameters[param].max}
            simplified_parameters.param = generated_val

        self.a2c_agent = A2CAgent(parameters=simplified_parameters, observation_size=observation_size, hidden_layer_size=hidden_layer_size, action_size=action_size, stochastic=stochastic)
        
    def mutate_hyperparameters(self):
        '''This function mutates, randomly, all the hyperparameters of this agent, according to the mutation rate'''
        for param in self.params_metadata:
            # We'll generate a completely random value, and mutate the original one according to the mutation rate.
            old_val = self.a2c_agent.get_hyperparameter(param)
            generated_val = torch.distributions.Uniform(self.params_metadata[param].min, self.params_metadata[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            # TODO: if > 0.5, 0.8
            # raise Error()
            discriminator = torch.distributions.Uniform(0, 1).sample().item()
            if discriminator > 0.5:
                mutation_rate = 1.0 - self.mutation_rate
            else:
                mutation_rate = self.mutation_rate
            mutated_val = (1.0 - mutation_rate) * old_val + mutation_rate * generated_val # For example, 0.8 * old_val + 0.2 * mutated_val
            self.a2c_agent.set_hyperparameter(param, mutated_val)
    
    def get_agent(self):
        return self.a2c_agent

    def compute_critic_loss(self, reward, done, critic):
        return self.a2c_agent.compute_critic_loss(reward, done, critic)
    
    def compute_a2c_loss(self, action_probs, action, td):
        return self.a2c_agent.compute_a2c_loss(action_probs, action)

    def copy(self, other: Self):
        self.a2c_agent = other.get_agent().clone()

    def get_cumulated_reward(self):
        return self.a2c_agent.get_cumulated_reward()

    def __call__(self, workspace, t, **kwargs):
        return self.a2c_agent(time=t, workspace=workspace, kwargs=kwargs)


def make_env(**kwargs) -> gym.Env:
    # We set a timestep limit on the environment of max_episode_steps
    # We can also add a seed to the environment here
    return TimeLimit(gym.make(kwargs['env_name']), kwargs['max_episode_steps'])

def create_population(cfg):
    # We'll run this number of agents in parallel
    n_cpus = multiprocessing.cpu_count()

    # Create the required number of agents
    population = []
    workspaces = {} # A dictionary of the workspace of each agent
    for i in range(cfg.algorithm.population_size):
        environment = make_env(**cfg.env)
        # observation_size: the number of features of the observation (in Pendulum-v1, it is 3)
        observation_size = environment.observation_space.shape[0]
        # hidden_layer_size: the number of neurons in the hidden layer
        hidden_layer_size = cfg.algorithm.neural_network.hidden_layer_size
        # action_size: the number of parameters to output as actions (in Pendulum-v1, it is 1)
        action_size = environment.action_space.shape[0]

        # TODO: Is this the right way to do it? Should the environment be passed as a parameter?
        workspace = Workspace()
        # raise Error()

        # The agent that we'll train will use the A2C algorithm
        a2c_agent = A2CParameterizedAgent(cfg.algorithm.hyperparameters, observation_size, hidden_layer_size, action_size, cfg.algorithm.mutation_rate)
        # To generate the observations, we need a gym agent, which will provide the values in 'env/env_obs'
        environment_agent = EnvAgent(cfg)
        temporal_agent = TemporalAgent(Agents(environment_agent, a2c_agent))
        temporal_agent.seed(cfg.algorithm.stochasticity_seed)
        population.append(temporal_agent)
        
        workspaces[temporal_agent] = workspace
        
        async_agent = AsynchronousAgent(temporal_agent) # TODO: Implement async operation

    return population, workspaces

def get_cumulated_reward(workspace):
    crewards = workspace['env/cumulated_reward']
    done = workspace['env/done']
    return torch.mean(crewards[done]) # TODO: Should we get the mean of the cumulated rewards?
    
def sort_performance(agents_list: List[TemporalAgent], agents_workspaces: Dict[TemporalAgent, Workspace]):
    agents_list.sort(key=lambda agent: get_cumulated_reward(agents_workspaces[agent]), reverse=True) # TODO: Is this the right way to access the agent?

def select_pbt(portion, agents_list):
    random_index = torch.distributions.Uniform(0, portion * len(agents_list)).sample().item()
    return agents_list[int(random_index)]

def train(cfg, population: List[Agent], workspaces: Dict[Agent, Workspace]):
    for epoch in range(cfg.algorithm.max_epochs):
        for agent in population:
            workspace = workspaces[agent]
            agent(time=0, stop_variable='env/done', workspace=workspace)

        # They have all finished executing
        print('Finished epoch {}'.format(epoch))

        # We sort the agents by their performance
        sort_performance(population, agents_workspaces=workspaces)
        
        print('Cumulated rewards at epoch {}: {}'.format(epoch, [get_cumulated_reward(workspaces[agent]) for agent in population]))

        for bad_agent in population[-1 * int(cfg.algorithm.pbt_portion * len(population)) : ]:
            # Select randomly one agent to replace the current one
            agent_to_copy = select_pbt(cfg.algorithm.pbt_portion, population)
            print('Copying agent with creward = {} into agent with creward {}'.format(get_cumulated_reward(workspaces[agent_to_copy]), get_cumulated_reward(workspaces[bad_agent])))
            bad_agent.agent[1].copy(agent_to_copy.agent[1])
            bad_agent.agent[1].mutate_hyperparameters()


@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    population, workspaces = create_population(cfg)
    train(cfg, population, workspaces)

if __name__ == '__main__':
    main()
