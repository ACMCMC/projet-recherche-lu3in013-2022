from copy import deepcopy
import multiprocessing
from typing import Dict, List

import gym
import hydra
import matplotlib.pyplot as plt
import torch
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf
from salina import (Agent, Workspace, get_arguments, get_class,
                    instantiate_class)
from salina.agents import Agents, TemporalAgent
from salina.agents.asynchronous import AsynchronousAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.logger import TFLogger
from a2c import A2CParameterizedAgent, CriticAgent, create_a2c_agents

from common import get_cumulated_reward
from env import AutoResetEnvAgent, NoAutoResetEnvAgent
from plot import CrewardsLogger, Logger, plot_hyperparams
from torch import nn

from utils import load_model

def make_env(**kwargs) -> gym.Env:
    # We set a timestep limit on the environment of max_episode_steps
    # We can also add a seed to the environment here
    return TimeLimit(gym.make(kwargs['env_name']), kwargs['max_episode_steps'])

def visualize_performances(workspaces: List[Workspace]):
    # We visualize the performances of the agents
    fig, ax = plt.subplots()
    for workspace in workspaces:
        visualize_performance(ax, workspace)
    ax.set(xlabel='timestep', ylabel='creward',
       title='Evolution of crewards')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

def visualize_performance(axes, workspace: Workspace):
    axes.plot(workspace['env/cumulated_reward'].mean(dim=1))



    
def sort_performance(agents_list: List[TemporalAgent], crewards: Dict[TemporalAgent, torch.Tensor]):
    agents_list.sort(key=lambda agent: crewards[agent], reverse=True)

def select_pbt(portion, agents_list):
    random_index = torch.distributions.Uniform(0, portion * len(agents_list)).sample().item()
    return agents_list[int(random_index)]

def _index_3d_2d(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensor using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v

class PBTAgent:
    '''
    This class contains all the necessary agents for one member of the population.
    '''
    def __init__(self, cfg: OmegaConf) -> None:
        # 1) Start by creating the environment agents
        self.train_env_agent = AutoResetEnvAgent(cfg)
        self.eval_env_agent = NoAutoResetEnvAgent(cfg)

        # 2) Create all the agents that we'll use for training and evaluating
        self.train_agent, self.eval_agent, self.action_agent, self.critic_agent = create_a2c_agents(cfg, self.train_env_agent, self.eval_env_agent)

        # 3) Create the tcritic_agent, which will be used to compute the value of each state
        self.tcritic_agent = TemporalAgent(self.critic_agent)

        # 4) Create the workspace of this member of the population
        self.workspace = Workspace()

        # 5) Create the optimizer
        self.optimizer = create_optimizer(cfg, self.action_agent, self.critic_agent)

    
    def train(self, **kwargs):
        return self.train_agent(**kwargs)
    
    def get_creward(self):
        eval_workspace = Workspace() # This is a fresh new workspace that we will use to evaluate the performance of this agent, and we'll discard it afterwards.

        self.eval_agent(eval_workspace, t=0, stop_variable='env/done', stochastic=False) # Run the evaluation agent until it reaches a terminal state on all its environments

        rewards = eval_workspace['env/cumulated_reward'][-1] # Get the last cumulated reward of the agent, which is the reward of the last timestep (terminal state)

        return rewards.mean()
    
    def compute_loss(self, **kwargs):
        return self.action_agent.compute_loss(**kwargs)
    
    def mutate_hyperparameters(self, **kwargs):
        return self.action_agent.mutate_hyperparameters(**kwargs)
    
    def copy(self, other, cfg):
        restore_agent = Agents(self.action_agent, self.critic_agent)
        other_restore_agent = Agents(other.action_agent, other.critic_agent)
        restore_agent.load_state_dict(other_restore_agent.state_dict())
        restore_agent.eval()
        self.optimizer = create_optimizer(cfg, self.action_agent, self.critic_agent)
    
    def load(self, path, cfg):
        restore_agent = Agents(self.action_agent, self.critic_agent)
        restore_agent.load_state_dict(load_model(path))
        restore_agent.eval()
        self.optimizer = create_optimizer(cfg, self.action_agent, self.critic_agent)

def create_population(cfg):
    # Create the required number of agents
    population = []
    workspaces = {} # A dictionary of the workspace of each agent

    for i in range(cfg.algorithm.population_size):
        # 1) Create the necessary agents
        pbt_agent = PBTAgent(cfg)

        # 2) Create the workspace of the agent
        workspace = Workspace()

        # 3) Store them
        population.append(pbt_agent)
        workspaces[pbt_agent] = workspace
        
    return population, workspaces

def create_optimizer(cfg: OmegaConf, action_agent: Agent, critic_agent: Agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer

def train(cfg, population: List[PBTAgent], workspaces: Dict[Agent, Workspace], logger: TFLogger):
    # 1) Prepare the logger and initialize the variables
    epoch_logger = CrewardsLogger()
    total_timesteps = 0

    # 2) Train the agents
    for epoch in range(cfg.algorithm.max_epochs):
        print("Epoch: {}".format(epoch))
        for agent in population: # TODO: Not train the first one, print his model to see if it changes
            consumed_budget = 0
            workspace = workspaces[agent]
            while consumed_budget < cfg.algorithm.train_budget:
                if consumed_budget > 0:
                    workspace.zero_grad()
                    workspace.copy_n_last_steps(1)
                    agent.train(t=1, workspace=workspace, n_steps=cfg.algorithm.num_timesteps - 1) # It computes the critic as well
                else:
                    agent.train(t=0, workspace=workspace, n_steps=cfg.algorithm.num_timesteps) # It computes the critic as well
                steps = (workspace.time_size() - 1) * workspace.batch_size()
                consumed_budget += steps
                
                loss = agent.compute_loss(workspace=workspace, timestep=total_timesteps + consumed_budget, logger=logger)

                optimizer = agent.optimizer
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()
            
            
        total_timesteps += consumed_budget

        all_agents_total_timesteps = total_timesteps * len(population)

        # They have all finished executing
        print('Finished epoch {}'.format(epoch))

        ###########################
        ### PBT EXPLOIT/EXPLORE ###
        ###########################

        crewards = {agent: agent.get_creward() for agent in population}

        mean_crewards = torch.mean(torch.stack(list(crewards.values())))

        logger.add_log("reward", mean_crewards, all_agents_total_timesteps) # TODO: Print the total num of timesteps

        # We sort the agents by their performance
        sort_performance(population, crewards)

        print('Cumulated rewards at epoch {}: {}'.format(epoch, crewards.values()))

        epoch_logger.log_epoch(all_agents_total_timesteps, torch.tensor(list(crewards.values())))
        plot_hyperparams([a.action_agent.a2c_agent for a in population])

        for bad_agent in population[-1 * int(cfg.algorithm.pbt_portion * len(population)) : ]:
            # Select randomly one agent to replace the current one
            agent_to_copy = select_pbt(cfg.algorithm.pbt_portion, population)
            print('Copying agent with creward = {} into agent with creward {}'.format(crewards[agent_to_copy], crewards[bad_agent]))
            bad_agent.copy(agent_to_copy, cfg)
            bad_agent.mutate_hyperparameters()
        
        #for _, workspace in workspaces.items():
        #    workspace.zero_grad()
    
    epoch_logger.show()


@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    # First, build the  logger
    logger = Logger(cfg)
    torch.manual_seed(cfg.algorithm.stochasticity_seed)
    population, workspaces = create_population(cfg)
    #population[0].load('/home/acmc/repos/projet-recherche-lu3in013-2022/saved_agents/agent_28.pickle', cfg)
    train(cfg, population, workspaces, logger=logger)

if __name__ == '__main__':
    main()
