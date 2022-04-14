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
from a2c import A2CParameterizedAgent

from common import EnvAgent, get_cumulated_reward
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

def create_population(cfg):
    # We'll run this number of agents in parallel
    n_cpus = multiprocessing.cpu_count()

    # Create the required number of agents
    population = []
    workspaces = {} # A dictionary of the workspace of each agent

    environment = EnvAgent(cfg)
    # observation_size: the number of features of the observation (in Pendulum-v1, it is 3)
    observation_size = environment.get_observation_size()
    # hidden_layer_size: the number of neurons in the hidden layer
    hidden_layer_sizes = list(cfg.algorithm.neural_network.hidden_layer_sizes)
    # action_size: the number of parameters to output as actions (in Pendulum-v1, it is 1)
    action_size = environment.get_action_size()

    for i in range(cfg.algorithm.population_size):

        workspace = Workspace()
        # raise Error()

        # The agent that we'll train will use the A2C algorithm
        a2c_agent = A2CParameterizedAgent(cfg.algorithm.hyperparameters, observation_size, hidden_layer_sizes, action_size, cfg.algorithm.mutation_rate, discount_factor=cfg.algorithm.discount_factor)
        # To generate the observations, we need a gym agent, which will provide the values in 'env/env_obs'
        environment_agent = EnvAgent(cfg)
        temporal_agent = TemporalAgent(Agents(environment_agent, a2c_agent))
        temporal_agent.seed(cfg.algorithm.stochasticity_seed)
        population.append(temporal_agent)
        
        workspaces[temporal_agent] = workspace
        
        async_agent = AsynchronousAgent(temporal_agent) # TODO: Implement async operation

    return population, workspaces

def train(cfg, population: List[TemporalAgent], workspaces: Dict[Agent, Workspace], logger: TFLogger):
    epoch_logger = CrewardsLogger()

    total_timesteps = 0

    five_last_rewards = {}
    for agent in population:
        five_last_rewards[agent] = torch.tensor([])

    optimizers = {}
    for agent in population:
        # Configure the optimizer over the a2c agent
        optimizer_args = get_arguments(cfg.optimizer)
        optimizer = get_class(cfg.optimizer)(
            agent.parameters(), **optimizer_args
        )
        optimizers[agent] = optimizer

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
