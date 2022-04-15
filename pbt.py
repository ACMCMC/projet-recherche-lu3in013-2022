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
from a2c import A2CParameterizedAgent, CriticAgent

from common import EnvAgent, get_cumulated_reward
from env import AutoResetEnvAgent, NoAutoResetEnvAgent
from plot import CrewardsLogger, Logger, plot_hyperparams

#raise Error(run a2c alone)
#raise Error(use the train agent and the eval agent for the A2C agents, there's a function of sigaud to create them)




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



    
def sort_performance(agents_list: List[TemporalAgent], five_last_rewards: Dict[TemporalAgent, torch.Tensor]):
    agents_list.sort(key=lambda agent: get_cumulated_reward(five_last_rewards,agent), reverse=True)

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

def create_a2c_agents(cfg, train_env_agent, eval_env_agent):
    if train_env_agent.is_action_space_continuous():
        observation_size = train_env_agent.get_observation_size()
        action_size = train_env_agent.get_action_size()
        a2c_agent = A2CParameterizedAgent(cfg.algorithm.hyperparameters, observation_size, cfg.algorithm.neural_network.hidden_layer_sizes, action_size, cfg.algorithm.mutation_rate, discount_factor=cfg.algorithm.discount_factor)
        train_agent = Agents(train_env_agent, a2c_agent)
        eval_agent = Agents(eval_env_agent, a2c_agent)
    else:
        pass # TODO: Implement discrete action space

    critic_agent = CriticAgent(observation_size, cfg.algorithm.neural_network.hidden_layer_sizes)

    train_agent = TemporalAgent(train_agent)
    eval_agent = TemporalAgent(eval_agent)
    train_agent.seed(cfg.algorithm.stochasticity_seed)
    return train_agent, eval_agent, a2c_agent, critic_agent

class PBTAgent():
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
    
    def get_train_agent(self):
        return self.train_agent
    
    def get_creward(self):
        eval_workspace = Workspace() # This is a fresh new workspace that we will use to evaluate the performance of this agent, and we'll discard it afterwards.

        self.eval_agent(eval_workspace, t=0, stop_variable='env/done', stochastic=False) # Run the evaluation agent until it reaches a terminal state on all its environments

        rewards = eval_workspace['env/cumulated_reward'][-1] # Get the last cumulated reward of the agent, which is the reward of the last timestep (terminal state)

        return rewards.mean()


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

def train(cfg, population: List[TemporalAgent], workspaces: Dict[Agent, Workspace], logger: TFLogger):
    # 1) Prepare the logger and initialize the variables
    epoch_logger = CrewardsLogger()
    total_timesteps = 0

    optimizers = {} # A dictionary of the optimizers of each agent
    for agent in population:
        # Configure the optimizer over the a2c agent
        optimizer_args = get_arguments(cfg.optimizer)
        optimizer = get_class(cfg.optimizer)(
            agent.parameters(), **optimizer_args
        )
        optimizers[agent] = optimizer

    # 2) Train the agents
    for epoch in range(cfg.algorithm.max_epochs):
        print("Epoch: {}".format(epoch))
        for agent in population:
            consumed_budget = 0
            while consumed_budget < cfg.algorithm.train_budget:
                workspace = workspaces[agent]
                if consumed_budget > 0:
                    workspace.zero_grad()
                    workspace.copy_n_last_steps(1)
                    agent(t=1, workspace=workspace, n_steps=cfg.algorithm.num_timesteps - 1)
                else:
                    agent(t=0, workspace=workspace, n_steps=cfg.algorithm.num_timesteps)

                steps = (workspace.time_size() - 1) * workspace.batch_size()
                consumed_budget += steps
                
                done = workspace["env/done"]

                loss = agent.agent[1].compute_loss(workspace=workspace, timestep=total_timesteps + consumed_budget, logger=logger)

                optimizer = optimizers[agent]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                creward = workspace["env/cumulated_reward"]
                creward = creward[done]

                five_last_rewards[agent] = torch.cat((five_last_rewards[agent], creward))[-5:]

                if creward.size()[0] > 0:
                    logger.add_log("reward", creward.mean(), total_timesteps + consumed_budget)
            

                # stop = [workspace['env/done'][-1].all() for workspace in workspaces.values()]
                # if all(stop):
                #     break
        
        total_timesteps += consumed_budget

        # They have all finished executing
        print('Finished epoch {}'.format(epoch))

        ###########################
        ### PBT EXPLOIT/EXPLORE ###
        ###########################

        # We sort the agents by their performance
        sort_performance(population, five_last_rewards)

        cumulated_rewards = {agent: get_cumulated_reward(five_last_rewards, agent).item() for agent in population}
        
        print('Cumulated rewards at epoch {}: {}'.format(epoch, cumulated_rewards.values()))

        epoch_logger.log_epoch(total_timesteps, torch.tensor(list(cumulated_rewards.values())))
        plot_hyperparams([a.agent[1].a2c_agent for a in population])

        for bad_agent in population[-1 * int(cfg.algorithm.pbt_portion * len(population)) : ]:
            # Select randomly one agent to replace the current one
            agent_to_copy = select_pbt(cfg.algorithm.pbt_portion, population)
            print('Copying agent with creward = {} into agent with creward {}'.format(cumulated_rewards[agent_to_copy], cumulated_rewards[bad_agent]))
            bad_agent.agent[1].copy(agent_to_copy.agent[1])
            bad_agent.agent[1].mutate_hyperparameters()
        
        for _, workspace in workspaces.items():
            # workspace.clear() # TODO: Is this the right way to do it?
            workspace.zero_grad()
            pass
    
    epoch_logger.show()


@hydra.main(config_path=".", config_name="pbt.yaml")
def main(cfg):
    # First, build the  logger
    logger = Logger(cfg)
    population, workspaces = create_population(cfg)
    train(cfg, population, workspaces, logger=logger)

if __name__ == '__main__':
    main()
