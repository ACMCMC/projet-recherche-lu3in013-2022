import math
import multiprocessing
from copy import deepcopy
from typing import Union

import hydra
import omegaconf
import salina
import torch
import torch.nn as nn
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, TemporalAgent
from salina.agents.asynchronous import AsynchronousAgent

from env import AutoResetEnvAgent, NoAutoResetEnvAgent
from utils import build_nn, load_model, save_model, gae
from plot import CustomLogger, plot_hyperparams, Logger

def create_a2c_agents(cfg, train_env_agent, eval_env_agent, kwargs_action_agent={}):
    if train_env_agent.is_action_space_continuous():
        observation_size = train_env_agent.get_observation_size()
        action_size = train_env_agent.get_action_size()
        a2c_agent = A2CParameterizedAgent(cfg.algorithm.hyperparameters, observation_size, cfg.algorithm.neural_network.hidden_layer_sizes, action_size, cfg.algorithm.mutation_rate, **kwargs_action_agent)
        critic_agent = CriticAgent(observation_size, cfg.algorithm.neural_network.hidden_layer_sizes)
        train_agent = Agents(train_env_agent, a2c_agent, critic_agent)
        eval_agent = Agents(eval_env_agent, a2c_agent)
    else:
        pass # TODO: Implement discrete action space

    train_agent = TemporalAgent(train_agent)
    eval_agent = TemporalAgent(eval_agent)
    train_agent.seed(cfg.algorithm.stochasticity_seed)
    return train_agent, eval_agent, a2c_agent, critic_agent


class CriticAgent(salina.TAgent):
    def __init__(self, observation_size, hidden_layer_sizes):
        super().__init__()
        # Create the critic neural network
        # We use a function that takes a list of layer sizes and returns the neural network
        # TODO: This is for continuous actions, but we should be able to use it for discrete actions as well
        self.critic_model = build_nn([observation_size] + list(hidden_layer_sizes) + [1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        critic = self.critic_model(input).squeeze(-1)
        self.set(("critic", t), critic)

class A2CAgent(salina.TAgent):
    '''This agent implements an Advantage Actor-Critic agent (A2C).
    The hyperparameters of the agent are customizable.'''

    def __init__(self, parameters, observation_size, hidden_layer_sizes, action_size, stochastic=True, std_param=None):
        super().__init__()
        # Create the action neural network
        # We use a function that takes a list of layer sizes and returns the neural network
        self.action_model = build_nn([observation_size] + list(hidden_layer_sizes) + [action_size], activation=nn.ReLU)

        self.observation_size = observation_size # The size of the observations
        self.hidden_layer_sizes = hidden_layer_sizes # The sizes of the hidden layers
        self.action_size = action_size # The size of the actions (output of the action model)
        self.stochastic = stochastic # Whether to use stochastic or deterministic actions
        self.params = omegaconf.DictConfig(content=parameters) # The hyperparameters of the agent
        if std_param is None:
            init_variance = torch.randn(action_size, 1)
            self.std_param = nn.parameter.Parameter(init_variance)
        else:
            self.std_param = std_param.clone()
        self.softplus = torch.nn.Softplus() # Like ReLU, but with a smooth gradient at zero

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        scores = self.action_model(input)
        dist = torch.distributions.Normal(scores, self.softplus(self.std_param))
        self.set(("entropy", t), dist.entropy())

        if self.stochastic:
            action = dist.sample()
        else:
            action = scores

        logprobs = dist.log_prob(action).sum(axis=-1)

        self.set(("action", t), action)
        self.set(("action_logprobs", t), logprobs)

    def compute_a2c_loss(self, action_logprobs: torch.Tensor, td: float) -> float:
        a2c_loss = action_logprobs[:-1] * td.detach()
        return a2c_loss.mean()

    def get_hyperparameter(self, param_name):
        return self.params[param_name]

    def set_hyperparameter(self, param_name, value):
        self.params[param_name] = value

    def compute_critic_loss(self, cfg, reward, must_bootstrap, critic) -> Union[float, float]:
        # Compute temporal difference
        # target = reward[:-1] + cfg.algorithm.discount_factor * critic[1:].detach() * (must_bootstrap.float())
        # td = target - critic[:-1]
        td = gae(critic, reward, must_bootstrap, cfg.algorithm.discount_factor, cfg.algorithm.gae)

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()
        return critic_loss, td
    
    def compute_loss(self, cfg, train_workspace: Workspace, timestep, logger) -> float:
        entropy = train_workspace["entropy"]

        transition_workspace = get_transitions(train_workspace)
        critic, done, action_logprobs, reward, action, truncated = transition_workspace[
            "critic", "env/done", "action_logprobs", "env/reward", "action", "env/truncated"
        ]

        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        critic_loss, td = self.compute_critic_loss(cfg, reward, must_bootstrap, critic)

        entropy_loss = entropy.mean()

        a2c_loss = self.compute_a2c_loss(action_logprobs, td)

        logger.log_losses(timestep, critic_loss, entropy_loss, a2c_loss)

        loss = (
            - self.params.entropy_coef * entropy_loss
            + self.params.critic_coef * critic_loss
            - self.params.a2c_coef * a2c_loss
        )

        return loss

def get_transitions(workspace):
    """
    Takes in a workspace from salina:
    [(step1),(step2),(step3), ... ]
    return a workspace of transitions :
    [
        [step1,step2],
        [step2,step3]
        ...
    ]
    Filters every transitions [step_final,step_initial]
    """
    transitions = {}
    done = workspace["env/done"][:-1]
    for key in workspace.keys():
        array = workspace[key]

        # Now, we remove the transitions of the form (terminal_state -> initial_state)
        x = array[:-1][~done]
        x_next = array[1:][~done]
        transitions[key] = torch.stack([x, x_next])
    
    workspace_without_transitions = Workspace()
    for k,v in transitions.items():
        workspace_without_transitions.set_full(k, v)
    
    return workspace_without_transitions



class A2CParameterizedAgent(salina.TAgent):
    '''
    This agent is in charge of mutating the hyperparameters of an A2C agent.
    '''

    def __init__(self, parameters, observation_size, hidden_layer_sizes, action_size, mutation_rate, stochastic=True, generated_parameters=None):
        super().__init__()

        self.mutation_rate = mutation_rate
        
        simplified_parameters = omegaconf.DictConfig(content={}) # The A2C Agent only sees a dictionary of (param_name, param_value) entries
        self.params_metadata = omegaconf.DictConfig(content={}) # This wrapper will store the metadata for the parameters of the A2C agent, so it knows how to change them when needed
        for param in parameters:
            if generated_parameters is None: # If no initialization for the parameters is provided, we use our own initialization
                simplified_parameters[param] = torch.distributions.Uniform(parameters[param].min, parameters[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            else:
                simplified_parameters[param] = generated_parameters[param]
            self.params_metadata[param] = {'min': parameters[param].min, 'max': parameters[param].max}

        self.a2c_agent = A2CAgent(parameters=simplified_parameters, observation_size=observation_size, hidden_layer_sizes=hidden_layer_sizes, action_size=action_size, stochastic=stochastic)
        
    def mutate_hyperparameters(self):
        '''This function mutates, randomly, all the hyperparameters of this agent, according to the mutation rate'''
        for param in self.params_metadata:
            # We'll generate a completely random value, and mutate the original one according to the mutation rate.
            old_val = self.a2c_agent.get_hyperparameter(param)
            generated_val = torch.distributions.Uniform(self.params_metadata[param].min, self.params_metadata[param].max).sample().item() # We get a 0D tensor, so we do .item(), to get the value
            discriminator = torch.distributions.Uniform(0, 1).sample().item()
            if discriminator > 0.5:
                mutated_val = 0.8 * old_val
            else:
                mutated_val = 1.2 * old_val
            self.a2c_agent.set_hyperparameter(param, mutated_val)
    
    def get_agent(self):
        return self.a2c_agent

    def compute_critic_loss(self, **kwargs):
        return self.a2c_agent.compute_critic_loss(**kwargs)
    
    def compute_a2c_loss(self, **kwargs):
        return self.a2c_agent.compute_a2c_loss(**kwargs)

    def copy(self, other):
        self.a2c_agent = deepcopy(other.get_agent())
    
    def copy_hyperparams(self, other):
        self.get_agent().params = deepcopy(other.get_agent().params)

    def get_cumulated_reward(self):
        return self.a2c_agent.get_cumulated_reward()

    def __call__(self, workspace, t, **kwargs):
        return self.a2c_agent(t=t, workspace=workspace, kwargs=kwargs)
    
    def compute_loss(self, **kwargs):
        return self.a2c_agent.compute_loss(**kwargs)

def get_creward(eval_agent):
    eval_workspace = Workspace() # This is a fresh new workspace that we will use to evaluate the performance of this agent, and we'll discard it afterwards.

    eval_agent(eval_workspace, t=0, stop_variable='env/done', stochastic=False) # Run the evaluation agent until it reaches a terminal state on all its environments

    rewards = eval_workspace['env/cumulated_reward'][-1] # Get the last cumulated reward of the agent, which is the reward of the last timestep (terminal state)

    return rewards.mean()

def create_agent(cfg):
    # 1) Start by creating the environment agents
    train_env_agent = AutoResetEnvAgent(cfg)
    eval_env_agent = NoAutoResetEnvAgent(cfg)

    # 2) Create all the agents that we'll use for training and evaluating
    train_agent, eval_agent, action_agent, critic_agent = create_a2c_agents(cfg, train_env_agent, eval_env_agent)

    # 3) Create the tcritic_agent, which will be used to compute the value of each state
    tcritic_agent = TemporalAgent(critic_agent)

    # 4) Create the workspace of this member of the population
    workspace = Workspace()
    return action_agent, tcritic_agent, train_agent, eval_agent, workspace

def a2c_train(cfg, action_agent: A2CParameterizedAgent, tcritic_agent: TemporalAgent, train_agent: TemporalAgent, eval_agent: TemporalAgent, workspace: Workspace, logger):
    epoch_logger = CustomLogger(cfg)

    total_timesteps = 0
    max_reward = 0

    # Configure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.optimizer)
    optimizer = get_class(cfg.optimizer)(
        nn.Sequential(action_agent, tcritic_agent.agent).parameters(), **optimizer_args
    )

    for epoch in range(cfg.algorithm.max_epochs):
        print("Epoch: {}".format(epoch))
        consumed_budget = 0
        while consumed_budget < cfg.algorithm.train_budget:
            if consumed_budget > 0:
                workspace.zero_grad()
                workspace.copy_n_last_steps(1)
                train_agent(t=1, workspace=workspace, n_steps=cfg.algorithm.num_timesteps - 1)
            else:
                train_agent(t=0, workspace=workspace, n_steps=cfg.algorithm.num_timesteps)

            steps = (workspace.time_size() - 1) * workspace.batch_size()
            consumed_budget += steps
            
            loss = action_agent.compute_loss(cfg=cfg, train_workspace=workspace, timestep=total_timesteps + consumed_budget, logger=logger)


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_agent.parameters(), cfg.algorithm.max_grad_norm)
            optimizer.step()

        total_timesteps += consumed_budget

        print('Finished epoch {}'.format(epoch))

        reward = get_creward(eval_agent)

        logger.add_log("reward", reward, total_timesteps)

        if (reward > max_reward):
            max_reward = reward
            save_agent = Agents(action_agent, tcritic_agent.agent)
            save_model(save_agent.state_dict(), '/home/acmc/repos/projet-recherche-lu3in013-2022/saved_agents/agent_{}.pickle'.format(math.floor(reward)))
            print('Saved agent')
        

@hydra.main(config_path=".", config_name="pbt.yaml")
def run_a2c(cfg):
    # First, build the  logger
    logger = Logger(cfg)
    torch.manual_seed(cfg.algorithm.stochasticity_seed)
    action_agent, tcritic_agent, train_agent, eval_agent, workspace = create_agent(cfg)
    #train_agent.load_state_dict(load_model('/home/acmc/repos/projet-recherche-lu3in013-2022/saved_agents/agent_20.pickle'))
    a2c_train(cfg, action_agent, tcritic_agent, train_agent, eval_agent, workspace, logger=logger)

if __name__ == '__main__':
    run_a2c()
